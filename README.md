# RLHGNN

> **RLHGNN** (Reinforcement Learning-guided Heterogeneous Graph Neural Network) is a framework for **Next Activity Prediction** in Predictive Business Process Monitoring.

## 1. Project Overview

Traditional methods represent event log prefixes with a single fixed graph structure, which cannot adapt to the diversity of different prefixes. The core idea of RLHGNN is:

1. Define **4 heterogeneous graph representations** (graph configurations) for each event log prefix, each focusing on different process semantics;
2. Train a base GNN model for each graph configuration and evaluate its performance on the current prefix;
3. Train a **reinforcement learning (PPO/DQN)** decision model: given the current prefix, **dynamically select the most suitable graph configuration**;
4. Construct a hybrid graph dataset from the RL-selected "optimal graphs" and train the final next-activity prediction model.

### 4 Graph Configurations (`build_graph.py`)

| choice | Configuration | Edge Types |
| :---: | --- | --- |
| 0 | `forward` | Forward edges (activity → next activity) |
| 1 | `Bidirect` | Forward + backward edges |
| 2 | `forward_complex` | Forward + repeated-activity skip edges |
| 3 | `Bidirect_complex` | Bidirectional + repeated-activity skip edges |

Each node represents an event (with multi-dimensional attributes such as `activity` / `duration`), modeled by `HeteroSAGE` (heterogeneous SAGE convolution: `forward` / `backward` edges use LSTM aggregation, `repeat_next` edges use mean aggregation; Post-LN residual blocks + stochastic depth (DropPath) + max-pooling readout), which finally outputs the probability distribution over the next activity.

### Pipeline

```mermaid
flowchart LR
    A[Raw event log CSV] --> B[cut_train_test.py temporal split]
    B --> C[data_process.py prefix samples + 4 graph variants]
    C --> D[pre_main.py pretrain 4 base GNNs]
    D --> E[env_train.py train PPO decision model]
    E --> F[final_policy.py RL graph selection, hybrid graphs]
    F --> G[final_main.py train final prediction model]
    G --> H[metrics_final.py evaluation metrics]
```

---

## 2. Environment Setup

### 2.1 Install Dependencies

**推荐用一键脚本**（会自动检测 CUDA 版本、按正确顺序安装、并做环境自检）：

```bash
chmod +x setup_env.sh
./setup_env.sh                  # 自动检测 CUDA；也可 --cuda cu118|cu121|cu124
```

如果想手动安装，必须严格按下面的顺序，**版本不能改**：

```bash
# 1) 先装 torch，并锁定 CUDA 源（DGL 的 wheel 是针对特定 torch/CUDA 编译的）
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121   # 按你的卡改成 cu118 / cu121 / cu124

# 2) 再装 DGL。必须用官方 wheel 仓库（PyPI 上的 dgl 是 CPU 版），
#    且必须 --no-deps，否则 pip 会把 torch 换成别的版本。
pip install dgl==2.4.0 \
    -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html --no-deps

# 3) 再装 torchdata（DGL 2.x 的硬依赖，版本要与 torch 对齐）
pip install torchdata==0.9.0 --no-deps
pip install urllib3 requests

# 4) 最后装其余依赖
pip install -r requirements.txt
```

> **Why these versions?** The upper bounds are enforced by the code and by upstream packages, not chosen arbitrarily:
> - `torch<=2.4.0` — DGL 2.4.0's `setup.py` declares `install_requires=[..., "torch<=2.4.0"]`, and DGL publishes wheels only up to `torch-2.4`. A newer torch has no matching DGL wheel.
> - `dgl>=2.0` — `model/model.py` calls `dgl.max_nodes(...)`, which only exists from DGL 2.0 (`dgl/readout.py`).
> - `torch>=2.0` — `Pre_Train.py` / `final_main.py` use `torch.optim.NAdam`, added in torch 2.0.
> - `pm4py<=2.7.12.2` and `imbalanced-learn<=0.12.4` — newer releases require Python>=3.11 / >=3.10, which excludes the Python 3.9 environment this project runs on.
> - `numpy<2` — required by DGL 2.3.x and by the pinned pm4py / imbalanced-learn builds.
> - `stable-baselines3==2.3.2` + `gymnasium==0.29.1` — the last pairing that stays on the gymnasium 0.29 API without pulling in newer behavioural changes.

> **Note:** DGL must match the CUDA version of PyTorch. `pip install dgl` from PyPI installs a **CPU-only** build (PyPI's latest is 2.2.1), so the `-f https://data.dgl.ai/wheels/...` source is mandatory.

### 2.2 Verify the Environment

```bash
python -c "import torch, dgl; print(torch.__version__, dgl.__version__, dgl.backend.is_cuda_available())"
```

`setup_env.sh` runs this check (plus API-level checks for every symbol the project imports) and exits non-zero on failure. If `import dgl` raises `ModuleNotFoundError` for `networkx` / `pydantic` / `psutil`, the DGL install lost its dependencies — run `pip install -r requirements.txt` again.

### 2.3 Hardware Requirements

- A **GPU** is required for training (scripts default to `cuda:0`; change it via the `--gpu` argument or the `get_device()` function in the code).
- Running the full pipeline over all datasets × 5 seeds takes a long time; we recommend first validating the workflow on a small subset.

---

## 3. Data Preparation

### 3.1 Datasets

The project supports the following datasets (raw event logs go in `train_test_data/<dataset>/<dataset>.csv` with columns `case, activity, timestamp`; some datasets have extra columns such as `duration`):

- `bpi13_closed_problems`
- `bpi13_problems`
- `bpi13_incidents`
- `bpi12w_complete`
- `bpi12_all_complete`
- `BPI2020_Prepaid`
- `p2p` (source file is the OCEL file `p2p.jsonocel`; conversion required first)

> The event logs can be obtained from 4TU Research Data. Some data is already included under `train_test_data/`.

### 3.2 (Optional) OCEL → Traditional Event Log

`p2p` and `OTC` are in OCEL format and must first be converted to traditional event logs:

```bash
python convert_jsonocel_to_eventlog.py
```

The converted CSVs are written to `train_test_data/<dataset>/<dataset>.csv`.

### 3.3 Temporal Train/Test Split

```bash
python cut_train_test.py
```

- Cases are sorted by start time; the **first 80% of cases form the training set and the last 20% form the test set** (temporal isolation to avoid data leakage).
- Only events occurring before the earliest test-case start time are kept in the training set.
- Outputs: `train_test_data/<dataset>/<dataset>_kfoldcv_0_train.csv` and `<dataset>_kfoldcv_0_test.csv`.

> The `list_event` list in the code controls which datasets are processed; edit it as needed before running.

---

## 4. Full Training Pipeline (5 Steps)

> All scripts below run from the **project root**. The `list_eventlog` / `seed_list` lists in the code control which datasets and random seeds are processed (default seeds `[133, 188, 456, 789, 1666]`); edit them as needed.

### Step 1: Data Preprocessing — `data_process.py`

```bash
python data_process.py
```

- Reads `train_test_data/<dataset>/<dataset>_kfoldcv_0_{train,test}.csv` and produces train/validation/test splits.
- Generates **prefix samples** for each case (predicting the next activity).
- Builds **4 graph configurations** of DGL heterogeneous graphs for every sample, saved under `raw_dir/<dataset>_0/` (two copies `part1/` and `part2/`, plus feature vocabularies, labels, etc.).

### Step 2: Pretrain Base GNN Models — `pre_main.py`

```bash
python pre_main.py
```

- Trains one `HeteroSAGE` model per dataset × seed × graph configuration (choice 0–3).
- Training details: NAdam + 5-epoch linear warmup + cosine annealing scheduler (`eta_min=1e-6`) + AMP mixed precision + early stopping (patience 10).
- Default hyperparameters: hidden 256, epochs 50, lr 1e-3, batch 64, dropout 0.2, layers 2, label smoothing 0.01, weight decay 1e-5, grad clip 1.0.
- Outputs: `Pretrain/action_{choice}_{seed}/<dataset>/<dataset>_fold0_model.pkl`

**Evaluate the pretrained models** (optional):

```bash
python metrics_pre.py
```

Writes metrics (Accuracy / Precision / Recall / F1 / AUC / PRAUC) to `result_Pretrain/action_{choice}_{seed}/<dataset>/`.

### Step 3: Train the RL Decision Model — `env_train.py`

```bash
python env_train.py
```

- Loads the 4 base GNNs from Step 2 as "environment evaluators".
- Builds a Gymnasium environment: state = prefix features (activity sequence, length, entropy, duration statistics, etc.), action = choose one of the 4 graph configurations (0–3), reward = prediction performance of the selected GNN.
- Trains a **PPO** policy (MLP `[256, 128, 64]`, cosine-annealed learning rate `1e-3 → 0`, `n_steps=1024`, batch 128, `ent_coef=0.01`, early-stop patience 5).
- In `--legacy-reward` mode all 4 graph variants for every sample are pre-built and kept resident on GPU; the default path does not do this (it would hold `4 x N` DGL graphs in VRAM for no benefit).
- **Reward table (default since the efficiency pass).** The 4 base GNNs are frozen here, so the reward of a `(prefix, action)` pair is a constant. Instead of running one single-graph GNN forward per environment step (hundreds of thousands of times, with ~19x redundancy), each pair is evaluated exactly once up front and stored as `(loss, correct, pred)`; `env.step()` then just indexes the table. Nothing else changes — batch Macro-F1, the EMA baseline and the final reward are still computed live at every batch boundary, and the PPO policy still chooses actions online. See `reward_table.py`.
  - The table is computed through the *same* `evaluate_gnn`, on the same single graph, in fp32 — so it is **bit-identical** to the old path, not just numerically close.
  - Tables are cached on disk under `reward_cache/` (keyed by dataset, fold, seed, split, the 4 reward-model files, the loss config, and a **content hash of the split's own features + labels** — so a re-split, a re-pretrain, or a change in `activity`/`duration` all invalidate it, while simply rewriting the same `.npy` files does not), so re-running the PPO stage only pays the read cost. Cache build/read time is inside the wall-clock timer, so it is counted in the reported training time.
  - Flags: `--legacy-reward` restores the original per-step forward path, `--verify-reward-table` loads data + models and only runs the equivalence check (every `(prefix, action)` pair is re-forwarded live and compared against the table; add `--verify-sample N` for a quick spot check), `--no-cache` disables the disk cache.
  - Pass criteria for the check: `pred`/`correct` must match **exactly** (they feed macro-F1 and the reward directly, so a single flip changes the training trajectory), and `max|Δloss| < 1e-6`. It also asserts that `_tid` is a 0..N-1 permutation, that every precomputed state equals a fresh `extract_features` call, and that table values are finite and in range.
  - Recommended first run on a new dataset/seed: `python env_train.py --verify-reward-table` — it prints PASS/FAIL and exits non-zero on any discrepancy.
- Prefix state vectors are precomputed once per sample (same pure function, same `float32` cast as SB3 performs), instead of being rebuilt at every step.
- The default `list_eventlog` covers all 7 datasets; edit it to select a subset.
- Outputs the policy model: `RL_model/<dataset>/PPO_best_model_fold0_seed{seed}`.
- After training, `final_policy.py` is invoked automatically: it uses the optimal policy to pick a graph configuration for **every prefix** and generates the final hybrid graph dataset:
  - `graph_data/<dataset>_0_seed{seed}/train_graphs`
  - `graph_data/<dataset>_0_seed{seed}/val_graphs`
  - `graph_data/<dataset>_0_seed{seed}/test_graphs`

### Step 4: Train the Final Prediction Model — `final_main.py`

```bash
python final_main.py
```

- Trains the final `HeteroSAGE` model (next-activity classification) on the RL-generated hybrid graphs.
- Default hyperparameters: hidden 256, epochs 50, lr 1e-3, batch 64, dropout 0.1, layers 3, label smoothing 0.1, weight decay 1e-4, grad clip 2.0, early stopping patience 10.
- Outputs: `final_train/<dataset>/seed{seed}/<dataset>_fold0_seed{seed}_model.pkl`
- Training time is recorded under `train_time/` and `pred_time/`.

### Step 5: Evaluate the Final Model — `metrics_final.py`

```bash
python metrics_final.py
```

- Computes Accuracy / Precision / Recall / F1 / AUC / PRAUC / Gmean on the test set.
- Results are written to `result_final_train/<dataset>/seed{seed}/<dataset>_seed{seed}.txt`.

---

## 5. One-Command Run (Linux / macOS)

`run_pipeline.sh` executes Steps 1→2→3→4→5 in order and stops if any step fails. Logs are saved under `logs/`:

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh                     # default python
./run_pipeline.sh -p /usr/bin/python3 # specify the Python interpreter
./run_pipeline.sh -e myenv            # activate a conda environment
./run_pipeline.sh --skip-data         # skip data preprocessing (when raw_dir already exists)
./run_pipeline.sh --skip-pre          # skip pretraining
./run_pipeline.sh --skip-env          # skip RL training
./run_pipeline.sh --skip-final        # skip final training
./run_pipeline.sh --skip-metrics      # skip evaluation
```

> On Windows, run it in Git Bash / WSL, or simply execute each script from Section 4 (`python xxx.py`).

---

## 6. Ablation & Control Experiments (Optional)

Used to validate the contribution of each graph configuration (and of the RL selection mechanism):

```bash
# Train ablation models: one model per dataset × seed × fixed graph configuration
python ablation_main.py
# Outputs: ablation_exp/<dataset>/action_{choice}/seed{seed}/<dataset>_fold0_seed{seed}_model.pkl

# Evaluate ablation models
python ablation_metrics.py
# Outputs: ablation_exp/result_ablation/<dataset>/action_{choice}/seed{seed}/

# Build oracle ceiling tables from the four ablation GNNs
python ablation_oracle_ceiling.py
# Outputs: ablation_exp/oracle_ceiling/seed133/<dataset>_oracle_table.csv
# Each CSV contains prefix_id, true_label, and the four action predictions.
# The script also writes oracle_accuracy and oracle_macro_f1 into a summary txt.
# oracle_macro_f1 is an approximate upper bound obtained by multi-restart (16)
# coordinate-ascent hill climbing over per-prefix action choices
# (CLI: --f1-restarts / --f1-seed); oracle_macro_f1_naive (fallback on misses)
# is saved alongside for comparison. Multi-seed summaries (mean ± std) are also
# written to the summary file. Other CLI options: --datasets --seeds --batch-size
# --gpu --num-workers --model-root --output-root.
```

### 前缀分组与规则选图对照实验（实验三）

**前缀分组（G1–G4）**：按前缀中的重复活动形态分组 —— G1 无重复活动（A→B→C）、G2 仅相邻重复（A→A→B）、G3 仅非相邻重复（A→B→A）、G4 相邻与非相邻重复同时存在（A→A→B→A）。判定实现在 `prefix_groups.py`（口径：同一活动连续出现 3 次及以上如 A→A→A 只算相邻重复归 G2；结构与名称映射 Basic=0 / Context-aware=1 / Pattern-aware=2 / Comprehensive=3）。

```bash
# 统计每个数据集 train/val/test 的 G1–G4 分布
python prefix_group_stats.py
# Outputs: prefix_group_stats/<dataset>_fold0_group_stats.txt
#          prefix_group_stats/all_datasets_fold0_group_stats.csv
```

**实验三（PPO vs 规则选图）**：用三种规则选择器替代 PPO 为每条 prefix 选图结构，其余流程（混合图数据集 → final HGNN 训练配置 → 测试集评估）与 RLHGNN 主实验完全一致：

| 规则 | 选择方式 |
| --- | --- |
| Rule 1 重复感知 | G1 选 Context-aware；G2/G3/G4 选 Comprehensive |
| Rule 2 重复模式 | G1 选 Basic；G2/G3 选 Pattern-aware；G4 选 Comprehensive |
| Rule 3 验证集最优 | 用 4 个固定结构 HGNN（消融模型）在**验证集**上按组统计 Accuracy，每组取最高者；映射按数据集×种子确定，仅用于选图，不接触测试集 |

```bash
# Rule 3 依赖消融模型，先跑（若还没有）：python ablation_main.py
python rule_control_main.py                       # 全部数据集 × 5 seeds × 3 规则
python rule_control_main.py --datasets p2p --seeds 133   # 快速验证
python rule_control_main.py --rules rule1 rule2   # 只跑静态规则（无需消融模型）
python rule_control_main.py --skip-existing       # 断点续跑
```

Outputs:
- `graph_data/<dataset>_0_seed<seed>_rule{1,2,3}/`（混合图 + `actions_*.npy` + `groups_*.npy`；rule3 另存 `rule3_mapping.txt` 记录验证集 Accuracy 表与组→结构映射）
- `final_train_rule/rule{X}/<dataset>/seed<seed>/`（模型）
- `result_final_train_rule/rule{X}/<dataset>/<dataset>_rule{X}_summary.txt`（5 seeds Mean±SD）
- `result_final_train_rule/<dataset>_rule_vs_ppo_summary.txt`（三种规则 vs RLHGNN 汇总与逐 seed 差值）

Statistical significance analysis (Friedman + Nemenyi test, CD diagram; fill in the metrics of each method manually in the data section of the script):

```bash
python cd_diagram.py
```

Outputs `cd_diagram_accuracy.png` and `cd_diagram_f1.png` (Demsar 2006 style).

### 6.1 Repeat-pattern prefix analysis (实验二)

Checks whether RLHGNN's advantage is concentrated on prefixes that contain repeated activities,
instead of claiming a uniform gain. Grouping is done on the window the model actually sees
(left padding removed, truncated to `mean_trace`), so a prefix longer than the window is
classified by its last `mean_trace` events.

| Group | Definition | Example |
| --- | --- | --- |
| G1 | no repeated activity | A → B → C |
| G2 | adjacent repeats only | A → A → B |
| G3 | non-adjacent repeats only | A → B → A |
| G4 | both kinds present | A → A → B → A |

"Non-adjacent" is tested after compressing runs of equal activities; it does **not** imply business
rework unless there is event-level semantic evidence.

```bash
# 1) test-prefix manifest: case_id + prefix_end_event_index + G1-G4 (CPU only, no GPU/DGL)
python prefix_group_build.py -d bpi12_all_complete -d bpi13_incidents
# 2) per-prefix predictions of RLHGNN and the 4 fixed structures (+ PPO actions)
python prefix_group_export_preds.py -d bpi12_all_complete -d bpi13_incidents
# 3) per-group Accuracy/Macro-F1, McNemar/Wilcoxon tests, DiD bootstrap, action distribution
python prefix_group_metrics.py -d bpi12_all_complete -d bpi13_incidents
```

- Step 1 rebuilds the prefix order of `ProcessEventlog_one_graph.py` — note that the order follows
  the **sorted case key** (because `timestamp_transform()` re-concatenates cases with a sorting
  `groupby`) — and verifies every row against the `raw_dir` npy windows and labels; any mismatch
  aborts instead of silently mislabelling prefixes.
- Step 2 aligns all methods on `prefix_id` (`prefix_index` in the CSV) and refuses to write if a
  method's row count, prefix index or true label differs. Each fixed-structure graph set is loaded
  once and reused across seeds.
- Step 3 writes `per_seed.csv`, `summary.csv`, `mcnemar.csv`, `wilcoxon.csv`,
  `did_bootstrap.csv`, `action_distribution.csv` and `report.md`. Macro-F1 uses the dataset's full
  label set with `zero_division=0`. DiD = [RLHGNN − fixed]@Gx − [RLHGNN − fixed]@G1 with a
  case-level cluster bootstrap, so it directly tests "advantage concentrated on G3/G4".
- Outputs go to `exp2_repeat_groups/{prefix_index,preds,metrics}/`.

---

## 7. Directory Structure & Outputs

| Directory / File | Description |
| --- | --- |
| `train_test_data/` | Raw event logs and split CSVs |
| `raw_dir/<dataset>_0/` | Preprocessed prefix samples, 4 graph variants, feature vocabularies |
| `Pretrain/action_{choice}_{seed}/` | Pretrained base GNN models |
| `RL_model/<dataset>/` | PPO decision models |
| `graph_data/<dataset>_0_seed{seed}/` | Final hybrid graph dataset selected by RL |
| `final_train/<dataset>/seed{seed}/` | Final prediction models |
| `result_Pretrain/` `result_final_train/` | Evaluation metric results (txt) |
| `ablation_exp/` | Ablation models and results |
| `graph_data/<dataset>_0_seed{seed}_random/` | Hybrid graphs under random structure selection (control) |
| `final_train_random/` `result_final_train_random/` | Random-control models and metric results |
| `logs/` | Logs of the one-command pipeline |

## 8. Core Files Overview

| File | Purpose |
| --- | --- |
| `convert_jsonocel_to_eventlog.py` | OCEL → traditional event log |
| `cut_train_test.py` | Temporal train/test split |
| `ProcessEventlog_one_graph.py` | Prefix sample generation |
| `build_graph.py` | Construction of the 4 heterogeneous graph variants |
| `data_process.py` | Data preprocessing entry point |
| `Pre_Train.py` / `pre_main.py` | Base GNN pretraining |
| `metrics_pre.py` | Evaluation of pretrained models |
| `env_train.py` | PPO decision model training |
| `final_policy.py` | Generates the final graph data with the RL policy |
| `final_main.py` | Final prediction model training |
| `metrics_final.py` | Final model evaluation |
| `ablation_main.py` / `ablation_metrics.py` | Ablation study |
| `ablation_oracle_ceiling.py` | Oracle ceiling: per-prefix best-action accuracy + F1 upper-bound search |
| `random_control_main.py` | Random graph-structure selection control (uniform random per prefix → same final HGNN training, 5 seeds) |
| `prefix_groups.py` | Prefix grouping definitions (G1–G4) and rule name → structure mapping (Basic/Context-aware/Pattern-aware/Comprehensive) |
| `prefix_group_stats.py` | G1–G4 distribution statistics per dataset/split |
| `rule_control_main.py` | Experiment 3: PPO vs rule-based graph-structure selection (Rule 1/2/3, 5 seeds, Mean±SD, RLHGNN comparison) |
| `cd_diagram.py` | Friedman / Nemenyi tests and CD diagram (`cd_diagram_accuracy.png`, `cd_diagram_f1.png`) |
| `prefix_group_build.py` | Exp.2 step 1: test-prefix manifest (case_id + prefix_end_event_index) and G1–G4 repeat-pattern grouping |
| `prefix_group_export_preds.py` | Exp.2 step 2: per-prefix predictions of RLHGNN + the 4 fixed structures, plus PPO actions |
| `prefix_group_metrics.py` | Exp.2 step 3: per-group Accuracy/Macro-F1, McNemar/Wilcoxon, DiD cluster bootstrap, action distribution |
| `model/model.py` | `HeteroSAGE` model (hetero SAGEConv + LSTM/mean aggregation, Post-LN + stochastic depth (DropPath), max pooling) |
| `MyDataset*.py` | DGL dataset wrappers |

## 9. FAQ

- **"File not found under `raw_dir/`"**: run `data_process.py` first.
- **`env_train.py` cannot find the pretrained models**: run `pre_main.py` first and make sure `Pretrain/action_{choice}_{seed}/` contains the model for the dataset.
- **Reduce runtime**: keep only one dataset and one seed (e.g., `seed_list = [133]`), and lower `iterations` in `env_train.py`.
- **Recommended (validated) configuration**: on `p2p` / `bpi13_closed_problems` / `bpi13_problems` (5 seeds), hidden 256 / epochs 50 / lr 3e-4 / batch 32 / dropout 0.2 / layers 2 / label smoothing 0.1 / weight decay 3e-4 / clip 1.0 / patience 5 with 3–4 warmup epochs gives stable results (e.g. `p2p` Acc 0.798±0.016 / F1 0.788±0.012) and avoids seed-specific collapses; runs typically early-stop around epoch 9–13 (< 0.03 h).
- **Out of GPU memory**: reduce `--batch-size` (default 64).
- **Garbled characters in `bpi13_closed_problems`**: the code has a built-in `gbk` encoding mapping; other datasets default to `utf-8`. Add entries to `ENCODING_MAP` if needed.

## Tools

- **PyTorch**: deep learning framework
- **DGL**: graph neural networks
- **Stable-Baselines3 / Gymnasium**: reinforcement learning
- **Python**: programming language

## Data

Event logs for predictive business process monitoring can be obtained from [4TU Research Data](https://data.4tu.nl).