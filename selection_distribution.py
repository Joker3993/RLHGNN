# -*- coding: utf-8 -*-
"""
Selection distribution 统计脚本（纯推理，不训练）

统计最终 graph-selection policy（env_train.py 训练并保存的 PPO 模型）在每个数据集上
对 train / val / test 样本选择 4 种图结构动作（action 0~3）的比例。

要点：
- 不重新训练：只加载已有的 PPO 权重做 deterministic 推理（无梯度、无 update）
- 不构建图、不加载 GNN：观测量只由 raw_dir 里的 activity / duration 计算，
  与 env_train.PrefixGraphEnv.extract_features 完全一致
- CPU 即可运行，分钟级

一键全量统计（默认行为，无需任何参数）：
    python selection_distribution.py
        # 自动扫描 RL_model/ 下的全部 dataset x fold x seed，
        # 用每个 seed 的 best RL 模型（PPO_best_model_*.zip）做 deterministic 推理，
        # 统计动作分布；主结果为【最终图数据集】口径：
        #   与 final_policy.py 落盘的 graph_data/<dataset>_<fold>_seed<seed>/ 一一对应
        #   （即 train + val + test 全部样本，用顶层 raw_dir 特征，而非 env_train 的 part2）
        # 同时输出 train / val / test 分项，并汇总成 跨 seed 的 mean ± std

常用变体：
    python selection_distribution.py --source both          # 同时统计 full 与 part2 两种样本口径
    python selection_distribution.py -d OTC -d p2p          # 只统计指定数据集
    python selection_distribution.py -d OTC -s 133 -s 188   # 只统计指定 seed
    python selection_distribution.py --no-fast-obs          # 关闭向量化（逐条，最保守）
    python selection_distribution.py --max-samples 200      # 冒烟测试

性能与内存：
    - 每个 (dataset, source, fold) 的观测只构造一次，被该组内所有 seed 复用；
      处理完立即释放，内存峰值 = 单个 dataset 的全部样本
    - 观测构造默认走向量化路径（结果与逐条实现抽样比对，不一致自动回退）
    - CPU 即可运行

输出：
    <out-dir>/selection_distribution.csv          # 明细（含 source 列）
                                                  # source,dataset,fold,seed,split,action,action_name,count,total,ratio
    <out-dir>/selection_distribution_summary.csv  # 每个 (source,dataset,split) 跨 seed 的 mean/std
    <out-dir>/selection_distribution_summary.json
    <out-dir>/selection_distribution_final_graph_data.csv
                                                  # 【主结果】最终图数据集(train+val+test 合并)
                                                  # 的 action distribution，对应 graph_data/
    <out-dir>/selection_distribution_paper_table.csv  # 论文用紧凑表：一行一个 dataset，列为 4 个 action
    <out-dir>/actions/<source>/<dataset>_seed<seed>_fold<f>.npy
                                                  # 可选(--dump-actions)：逐样本动作，int8，
                                                  # 顺序与 final_policy 一致 train||val||test
"""

import argparse
import json
import os
import pickle
import re
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from utils import calculate_entropy, calculate_longest_consecutive_subsequence

NUM_ACTIONS = 4
ACTION_NAMES = {
    0: "forward",
    1: "bidirect",
    2: "forward_complex",
    3: "bidirect_complex",
}
MODEL_RE = re.compile(r"PPO_best_model_fold(\d+)_seed(\d+)$")


# --------------------------------------------------------------------------
# 与 env_train.py / final_policy.py 完全一致的观测构造
# --------------------------------------------------------------------------
def extract_features(current_prefix_all_attr):
    activity = current_prefix_all_attr.get("activity", [])
    duration = current_prefix_all_attr.get("duration", [])

    length = len(activity)

    unique_activities = len(np.unique(activity))

    top3 = activity[:3].tolist() if len(activity) >= 3 else activity.tolist()
    last3 = activity[-3:].tolist() if len(activity) >= 3 else activity.tolist()

    activity_entropy = calculate_entropy(activity)
    longest_subsequence = calculate_longest_consecutive_subsequence(activity)

    avg_duration = np.mean(duration)
    max_duration = np.max(duration)
    min_duration = np.min(duration)

    features = [
        *activity,
        length,
        unique_activities,
        *top3,
        *last3,
        activity_entropy,
        longest_subsequence,
        avg_duration,
        max_duration,
        min_duration,
    ]

    return features


def _as_2d_float(x):
    """把 np.load(..., allow_pickle=True) 读出的东西统一成 (n, T) 的 float64 矩阵。

    注意：ProcessEventlog_one_graph.save_file 用的是 pickle.dump，所以 np.load 会
    返回 1-D object 数组（元素是 ndarray），而不是 2-D 数值数组。
    不满足等长等前提时返回 None，调用方回退逐条构造。
    """
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        return None

    if arr.dtype != object and arr.ndim == 2:
        try:
            return arr.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            return None

    # object 数组（或 1-D）：检查每个元素长度是否一致
    flat = arr.reshape(-1)
    if flat.size == 0:
        return None
    try:
        lengths = {len(v) for v in flat}
    except TypeError:
        return None
    if len(lengths) != 1:
        return None
    T = lengths.pop()
    if T == 0:
        return None
    try:
        return np.asarray([np.asarray(v, dtype=np.float64) for v in flat]).reshape(
            arr.shape[0], T
        )
    except (TypeError, ValueError):
        return None


def _features_matrix_fast(A, D):
    """向量化构造观测矩阵（A/D 均为 shape=(n, T) 的等长数值矩阵）。

    仅向量化可安全向量化的部分（length / unique / top3 / last3 / duration 统计），
    entropy 与 longest_consecutive 仍调用 utils 中的原函数，保证语义一致。
    返回 (matrix, ok)，ok=False 表示不满足向量化前提，调用方应回退逐条构造。
    """
    Am = _as_2d_float(A)
    Dm = _as_2d_float(D)
    if Am is None or Dm is None:
        return None, False
    if Am.shape != Dm.shape:
        return None, False

    n, T = Am.shape
    if n == 0 or T < 3:  # T<3 时原实现 top3/last3 退化为整行，长度会变，回退
        return None, False

    Af = Am
    Df = Dm

    # length：原实现 len(activity)，含 padding，等长时为常数 T
    lengths = np.full((n, 1), float(T), dtype=np.float64)

    # unique_activities：原实现 len(np.unique(activity))，含 padding 0
    S = np.sort(Af, axis=1)
    neq = np.ones((n, T), dtype=bool)
    neq[:, 1:] = S[:, 1:] != S[:, :-1]
    uniq = neq.sum(axis=1, keepdims=True).astype(np.float64)

    top3 = Af[:, :3]
    last3 = Af[:, -3:]

    entropy = np.empty((n, 1), dtype=np.float64)
    longest = np.empty((n, 1), dtype=np.float64)
    for i in range(n):
        row = flat_row(A, i)  # 用原始元素调用 utils，语义与逐条实现一致
        entropy[i, 0] = calculate_entropy(row)
        longest[i, 0] = calculate_longest_consecutive_subsequence(row)

    avg = Df.mean(axis=1, keepdims=True)
    mx = Df.max(axis=1, keepdims=True)
    mn = Df.min(axis=1, keepdims=True)

    matrix = np.concatenate(
        [Af, lengths, uniq, top3, last3, entropy, longest, avg, mx, mn], axis=1
    ).astype(np.float32)
    return matrix, True


def flat_row(x, i):
    """取第 i 个样本（兼容 object 数组 / 2-D 数值数组）。"""
    arr = x if isinstance(x, np.ndarray) else np.asarray(x)
    if arr.ndim == 2 and arr.dtype != object:
        return arr[i]
    return arr.reshape(-1)[i]


def build_obs(A, D, fast=True, check_n=200):
    """构造观测：默认向量化，并与逐条实现抽样比对；不一致则自动回退。

    返回 np.ndarray(2D) 或 list（等长前提不成立时）。
    """
    if fast:
        matrix, ok = _features_matrix_fast(A, D)
        if ok:
            n = matrix.shape[0]
            idx = np.unique(np.linspace(0, n - 1, min(check_n, n)).astype(int))
            for i in idx:
                ref = np.asarray(
                    extract_features(
                        {"activity": flat_row(A, i), "duration": flat_row(D, i)}
                    ),
                    dtype=np.float32,
                )
                if ref.shape != matrix[i].shape or not np.allclose(
                    ref, matrix[i], rtol=0.0, atol=1e-5
                ):
                    print(f"  [提示] 向量化观测与逐条实现不一致(样本 {i})，回退逐条构造")
                    matrix = None
                    break
            if matrix is not None:
                return matrix

    rows = [
        extract_features({"activity": flat_row(A, i), "duration": flat_row(D, i)})
        for i in range(len(A))
    ]
    try:
        return np.asarray(rows, dtype=np.float32)
    except ValueError:
        return rows


# --------------------------------------------------------------------------
# 数据 / 模型发现
# --------------------------------------------------------------------------
def discover_models(model_root="RL_model"):
    """扫描 RL_model/<dataset>/PPO_best_model_fold<f>_seed<s>[.zip]"""
    found = []
    if not os.path.isdir(model_root):
        return found

    for dataset in sorted(os.listdir(model_root)):
        d = os.path.join(model_root, dataset)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            stem = fn[:-4] if fn.endswith(".zip") else fn
            m = MODEL_RE.match(stem)
            if not m:
                continue
            # sb3 保存时会自动追加 .zip，优先用带 .zip 的路径
            p = os.path.join(d, stem)
            if not os.path.exists(p + ".zip") and not os.path.exists(p):
                continue
            found.append(
                {
                    "dataset": dataset,
                    "fold": int(m.group(1)),
                    "seed": int(m.group(2)),
                    "model_path": p,
                }
            )
    # 去重（同一 stem 同时存在带/不带 .zip 时只留一条）
    dedup = {}
    for m in found:
        dedup[(m["dataset"], m["fold"], m["seed"])] = m
    return [dedup[k] for k in sorted(dedup)]


def load_observations(dataset, fold, split, source="full", raw_root="raw_dir",
                      max_samples=None, fast_obs=True):
    """读取 raw_dir 特征并构造观测（与 env_train 一致）。

    source='full'  -> raw_dir/<dataset>_<fold>/<col>_<fold>_<split>.npy
                      （final_policy 生成最终图数据集时使用的全部样本）
    source='part2' -> raw_dir/<dataset>_<fold>/part2/...（env_train 训练/评估所用的后半份）

    返回 (obs, n, features_name)；obs 为 shape=(n, d) 的 float32 矩阵（不满足等长时退化为 list）。
    """
    path = os.path.join(raw_root, f"{dataset}_{fold}")
    base = os.path.join(path, "part2") if source == "part2" else path

    if not os.path.isdir(base):
        raise FileNotFoundError(f"找不到目录：{base}")

    with open(os.path.join(path, "features_name.npy"), "rb") as f:
        features_name = pickle.load(f)

    label_path = os.path.join(base, f"label_{fold}_{split}.npy")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"找不到标签文件：{label_path}")
    labels = np.load(label_path, allow_pickle=True)

    n = len(labels)
    if max_samples is not None:
        n = min(n, max_samples)

    attrs = {}
    for name in ["activity", "duration"]:  # extract_features 只用到这两列
        fp = os.path.join(base, f"{name}_{fold}_{split}.npy")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"找不到特征文件：{fp}")
        attrs[name] = np.load(fp, allow_pickle=True)[:n]

    obs = build_obs(attrs["activity"], attrs["duration"], fast=fast_obs)
    return obs, n, features_name


# --------------------------------------------------------------------------
# 推理（批量优先，自动回退到逐条，结果与 env_train.evaluate 对齐）
# --------------------------------------------------------------------------
def _predict_one(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return int(np.asarray(action).reshape(-1)[0])


def _predict_batch(model, matrix, batch_size):
    actions = np.empty(len(matrix), dtype=np.int64)
    for start in range(0, len(matrix), batch_size):
        chunk = matrix[start:start + batch_size]
        act, _ = model.predict(chunk, deterministic=True)
        actions[start:start + len(chunk)] = np.asarray(act).reshape(-1)
    return actions


def predict_actions(model, obs, batch_size=2048, check_n=16):
    """对观测批量推理；obs 可以是 (n, d) 矩阵或 list。结果与 env_train.evaluate 对齐。"""
    n = len(obs)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    if isinstance(obs, np.ndarray) and obs.ndim == 2:
        try:
            actions = _predict_batch(model, obs, batch_size)
            # 抽样校验批量推理与逐条推理是否一致（如不一致则回退）
            idx = np.unique(np.linspace(0, n - 1, min(check_n, n)).astype(int))
            mismatched = False
            for i in idx:
                if _predict_one(model, obs[i]) != int(actions[i]):
                    mismatched = True
                    break
            if not mismatched:
                return actions
            print("  [提示] 批量推理与逐条推理结果不一致，回退为逐条推理以保证与训练时一致")
        except Exception as exc:  # noqa: BLE001
            print(f"  [提示] 批量推理失败({exc})，回退为逐条推理")

    return np.asarray([_predict_one(model, o) for o in obs], dtype=np.int64)


def _row(source, dataset, fold, seed, split, action, counts):
    """构造一条明细记录（counts 为该划分 4 个 action 的计数）。"""
    total = int(counts.sum())
    return {
        "source": source,
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "split": split,
        "action": action,
        "action_name": ACTION_NAMES[action],
        "count": int(counts[action]),
        "total": total,
        "ratio": float(counts[action] / total) if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="统计最终 graph-selection policy 的 action distribution（不训练，纯推理）"
    )
    parser.add_argument("-d", "--dataset", action="append", default=None,
                        help="数据集名，可重复；缺省自动扫描 RL_model/ 下所有含模型的 dataset")
    parser.add_argument("-s", "--seed", action="append", type=int, default=None,
                        help="seed，可重复；缺省使用扫到的全部 seed")
    parser.add_argument("--fold", type=int, default=None, help="fold（缺省使用扫到的全部 fold）")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"], help="要统计的划分")
    parser.add_argument("--source", choices=["full", "part2", "both"], default="full",
                        help="full=final_policy 使用的全部样本（默认）；part2=env_train 训练/评估所用的一半；both=两者都统计")
    parser.add_argument("--model-root", default="RL_model", help="RL 模型目录")
    parser.add_argument("--raw-root", default="raw_dir", help="raw_dir 路径")
    parser.add_argument("--out-dir", default="selection_dist", help="结果输出目录")
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    parser.add_argument("--batch-size", type=int, default=2048, help="批量推理大小")
    parser.add_argument("--max-samples", type=int, default=None, help="每个划分最多用多少样本（冒烟测试）")
    parser.add_argument("--no-fast-obs", action="store_true",
                        help="关闭向量化观测构造，完全逐条（最保守，速度较慢）")
    parser.add_argument("--dump-actions", action="store_true",
                        help="额外导出逐样本动作序列（int8），可用于与 graph_data/ 交叉核对")
    args = parser.parse_args()
    fast_obs = not args.no_fast_obs

    models = discover_models(args.model_root)
    if args.dataset:
        models = [m for m in models if m["dataset"] in set(args.dataset)]
    if args.seed:
        models = [m for m in models if m["seed"] in set(args.seed)]
    if args.fold is not None:
        models = [m for m in models if m["fold"] == args.fold]

    if not models:
        print("未找到任何 PPO 模型，请检查 RL_model/<dataset>/PPO_best_model_fold<f>_seed<s>.zip 是否存在")
        return

    datasets = sorted({m["dataset"] for m in models})
    seeds = sorted({m["seed"] for m in models})
    sources = ["full", "part2"] if args.source == "both" else [args.source]
    total_runs = len(models) * len(sources)
    print(f"数据集({len(datasets)}): {', '.join(datasets)}")
    print(f"seed({len(seeds)}): {', '.join(map(str, seeds))}")
    print(f"样本来源: {', '.join(sources)} | 划分: {', '.join(args.splits)}")
    print(f"待推理模型数: {total_runs}（{len(models)} 个权重 × {len(sources)} 种样本口径）\n")

    from stable_baselines3 import PPO  # 延迟导入，便于 --help 快速返回

    rows = []
    skipped = []
    done_runs = 0

    for dataset in datasets:
        ds_models = [m for m in models if m["dataset"] == dataset]
        for source in sources:
            for fold in sorted({m["fold"] for m in ds_models}):
                fold_seeds = sorted(m["seed"] for m in ds_models if m["fold"] == fold)
                print(f"\n########## {dataset} | source={source} | fold={fold} "
                      f"| seeds={fold_seeds} ##########")

                # ---- 1) 该组的观测只构造一次，供本组所有 seed 复用，随后释放 ----
                obs_by_split = {}
                for split in args.splits:
                    try:
                        obs_by_split[split] = load_observations(
                            dataset, fold, split, source, args.raw_root,
                            args.max_samples, fast_obs,
                        )
                    except FileNotFoundError as exc:
                        skipped.append(f"{dataset}/{source}/{split}: {exc}")
                        print(f"  [跳过] {split}: {exc}")
                if not obs_by_split:
                    continue
                for split, (obs, n, _) in obs_by_split.items():
                    shape = getattr(obs, "shape", (n, "ragged"))
                    print(f"  观测就绪 {split:<5} n={n:<8} shape={shape}")

                # ---- 2) 同一组观测上跑完所有 seed ----
                for seed in fold_seeds:
                    m = next(x for x in ds_models
                             if x["fold"] == fold and x["seed"] == seed)
                    model = PPO.load(m["model_path"], device=args.device)

                    seed_split_actions = {}
                    for split, (obs, n, _) in obs_by_split.items():
                        actions = predict_actions(model, obs, args.batch_size)
                        seed_split_actions[split] = actions

                        counts = np.bincount(actions, minlength=NUM_ACTIONS)
                        ratio = counts / max(1, counts.sum())
                        line = "  ".join(
                            f"a{a}({ACTION_NAMES[a]})={ratio[a] * 100:5.1f}%"
                            for a in range(NUM_ACTIONS)
                        )
                        print(f"  seed {seed} {split:<5} n={n:<8} {line}")

                        for a in range(NUM_ACTIONS):
                            rows.append(_row(source, dataset, fold, seed, split, a, counts))

                    # 该 seed 在所选划分上的合并分布
                    if seed_split_actions:
                        all_actions = np.concatenate(list(seed_split_actions.values()))
                        counts = np.bincount(all_actions, minlength=NUM_ACTIONS)
                        ratio = counts / max(1, counts.sum())
                        line = "  ".join(
                            f"a{a}({ACTION_NAMES[a]})={ratio[a] * 100:5.1f}%"
                            for a in range(NUM_ACTIONS)
                        )
                        print(f"  seed {seed} [最终图数据集] n={len(all_actions):<8} {line}")
                        for a in range(NUM_ACTIONS):
                            rows.append(_row(source, dataset, fold, seed, "all", a, counts))

                        if args.dump_actions:
                            dump_dir = os.path.join(args.out_dir, "actions", source)
                            os.makedirs(dump_dir, exist_ok=True)
                            # 顺序与 final_policy 一致：train || val || test
                            np.save(
                                os.path.join(dump_dir, f"{dataset}_seed{seed}_fold{fold}.npy"),
                                all_actions.astype(np.int8),
                            )
                    done_runs += 1

                del obs_by_split  # 释放内存峰值

    print(f"\n完成 {done_runs}/{total_runs} 次推理")
    if skipped:
        print(f"跳过 {len(skipped)} 项：")
        for s in skipped[:20]:
            print(f"  - {s}")
        if len(skipped) > 20:
            print(f"  ... 其余 {len(skipped) - 20} 项略")

    # ---------------- 输出明细与汇总 ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    detail_path = os.path.join(args.out_dir, "selection_distribution.csv")
    fields = ["source", "dataset", "fold", "seed", "split",
              "action", "action_name", "count", "total", "ratio"]
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in fields) + "\n")
    print(f"明细已保存: {detail_path}")

    # 汇总：按 (source, dataset, split) 跨 seed 求 mean/std
    splits = args.splits + ["all"]
    summary_rows = []
    print("\n=========== Selection Distribution (mean ± std over seeds) ===========")
    for source in sources:
        header = f"{'dataset':<24}{'split':<7}" + "".join(
            f"{'a' + str(a) + ' ' + ACTION_NAMES[a]:>26}" for a in range(NUM_ACTIONS)
        )
        print(f"\n--- source = {source} ---")
        print(header)
        print("-" * len(header))
        for dataset in datasets:
            for split in splits:
                per_action = {a: [] for a in range(NUM_ACTIONS)}
                for r in rows:
                    if (r["source"] == source and r["dataset"] == dataset
                            and r["split"] == split):
                        per_action[r["action"]].append(r["ratio"])
                if not any(per_action.values()):
                    continue
                cells = []
                summary_entry = {"source": source, "dataset": dataset,
                                 "split": split, "n_seeds": len(per_action[0])}
                for a in range(NUM_ACTIONS):
                    vals = np.asarray(per_action[a], dtype=float)
                    mean, std = float(vals.mean()), float(vals.std(ddof=0))
                    cells.append(f"{mean * 100:8.1f}% ± {std * 100:4.1f}%")
                    summary_entry[f"action{a}_mean"] = mean
                    summary_entry[f"action{a}_std"] = std
                summary_rows.append(summary_entry)
                print(f"{dataset:<24}{split:<7}" + "".join(f"{c:>26}" for c in cells))

    summary_csv = os.path.join(args.out_dir, "selection_distribution_summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as f:
        cols = ["source", "dataset", "split", "n_seeds"]
        for a in range(NUM_ACTIONS):
            cols += [f"action{a}_mean", f"action{a}_std"]
        f.write(",".join(cols) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n汇总已保存: {summary_csv}")

    summary_json = os.path.join(args.out_dir, "selection_distribution_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sources": sources,
                "splits": args.splits,
                "fast_obs": fast_obs,
                "action_names": ACTION_NAMES,
                "summary": summary_rows,
                "skipped": skipped,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"JSON 已保存: {summary_json}")

    # 论文用紧凑表：一行一个 (source, dataset)，列为 4 个 action 的百分比均值
    paper_split = "all" if "all" in splits else splits[0]
    paper_path = os.path.join(args.out_dir, "selection_distribution_paper_table.csv")
    with open(paper_path, "w", encoding="utf-8") as f:
        head = ["source", "dataset", "split", "n_seeds"] + [
            f"{ACTION_NAMES[a]} (%)" for a in range(NUM_ACTIONS)
        ]
        f.write(",".join(head) + "\n")
        for r in summary_rows:
            if r["split"] != paper_split:
                continue
            vals = [f"{r[f'action{a}_mean'] * 100:.2f}" for a in range(NUM_ACTIONS)]
            f.write(",".join(
                [r["source"], r["dataset"], r["split"], str(r["n_seeds"])] + vals
            ) + "\n")
    print(f"论文用紧凑表已保存: {paper_path}  (split='{paper_split}')")

    # ---------------- 主口径：最终图数据集（graph_data）的 Action Distribution ----------------
    # 即 final_policy.py 用 best RL 模型对 train+val+test 全部样本选择图结构后的分布，
    # 与 dgl.save_graphs 写入 graph_data/<dataset>_<fold>_seed<seed>/{train,val,test}_graphs 一一对应。
    print("\n" + "=" * 78)
    print("最终图数据集（train+val+test 合并）的 Action Distribution —— 直接对应 graph_data/")
    print("=" * 78)
    final_rows = [r for r in summary_rows if r["split"] == "all"]
    header = f"{'source':<8}{'dataset':<24}" + "".join(
        f"{'a' + str(a) + ' ' + ACTION_NAMES[a]:>26}" for a in range(NUM_ACTIONS)
    )
    print(header)
    print("-" * len(header))
    for r in final_rows:
        cells = []
        for a in range(NUM_ACTIONS):
            mean = r[f"action{a}_mean"]
            std = r[f"action{a}_std"]
            cells.append(f"{mean * 100:8.1f}% ± {std * 100:4.1f}%")
        print(f"{r['source']:<8}{r['dataset']:<24}" + "".join(f"{c:>26}" for c in cells))

    final_path = os.path.join(args.out_dir, "selection_distribution_final_graph_data.csv")
    with open(final_path, "w", encoding="utf-8") as f:
        cols = ["source", "dataset", "n_seeds"]
        for a in range(NUM_ACTIONS):
            cols += [f"action{a}_{ACTION_NAMES[a]}_mean", f"action{a}_{ACTION_NAMES[a]}_std"]
        f.write(",".join(cols) + "\n")
        for r in final_rows:
            vals = []
            for a in range(NUM_ACTIONS):
                vals += [f"{r[f'action{a}_mean']:.6f}", f"{r[f'action{a}_std']:.6f}"]
            f.write(",".join(
                [r["source"], r["dataset"], str(r["n_seeds"])] + vals
            ) + "\n")
    print(f"\n最终图数据集分布已保存: {final_path}")


if __name__ == "__main__":
    main()
