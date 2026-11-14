#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验二 / 步骤 2：逐前缀导出 RLHGNN 与 4 个固定结构的预测（外加 PPO 动作）

产出（一行一个测试前缀，任何统计工具都能直接吃）：

    exp2_repeat_groups/preds/<dataset>/seed<seed>.csv
        prefix_index, prefix_id, y_true, ppo_action,
        pred_RLHGNN, pred_fixed_0, pred_fixed_1, pred_fixed_2, pred_fixed_3

对齐口径（关键）
--------------
所有方法都按同一份 raw_dir/<dataset>_0 测试集的索引顺序输出，因此按 prefix_index 一一对齐：
  * 固定结构 action_c : raw_dir/<dataset>_0/Positive_graph_test_choice_c（顺序 == npy 顺序）
  * RLHGNN           : graph_data/<dataset>_0_seed<s>/test_graphs（final_policy 顺次构建）
  * PPO 动作          : 逐条 deterministic 推理，观测函数与 final_policy.extract_features 完全一致
脚本逐个校验：行数 == 清单行数、真实标签 == 清单 label、动作 ∈ [0,3]，任一不满足立即报错。

权重路径（与现有脚本一致）
------------------------
  * RLHGNN    final_train/<dataset>/seed<s>/<dataset>_fold0_seed<s>_model.pkl
  * 固定结构   ablation_exp/<dataset>/action_<c>/seed<s>/<dataset>_fold0_seed<s>_model.pkl
  * PPO       RL_model/<dataset>/PPO_best_model_fold0_seed<s>[.zip]

性能
----
四个固定结构的测试图与随机种子无关：每个 action 的图只加载一次，然后在上面轮换 5 个种子的模型，
避免把同一批（可能上万个）DGL 图重复加载 5 次。

用法
----
    python prefix_group_build.py -d bpi12_all_complete -d bpi13_incidents
    python prefix_group_export_preds.py -d bpi12_all_complete -d bpi13_incidents
    python prefix_group_export_preds.py -d p2p --seeds 133 188 --skip-ppo
"""

from __future__ import annotations

import argparse
import gc
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast  # 与 ablation_main.py / ablation_oracle_ceiling.py 保持一致

DEFAULT_DATASETS = [
    "bpi12_all_complete",
    "bpi13_incidents",
    "bpi13_closed_problems",
    "bpi13_problems",
    "bpi12w_complete",
    "BPI2020_Prepaid",
    "p2p",
    "OTC",  # 与 metrics_final.py 的 8 个数据集保持一致
]
DEFAULT_SEEDS = [133, 188, 456, 789, 1666]

FIXED_METHODS = {
    0: "A0_forward",
    1: "A1_bidirect",
    2: "A2_forward_complex",
    3: "A3_bidirect_complex",
}

OUTPUT_COLUMNS = ["prefix_index", "prefix_id", "y_true", "ppo_action", "pred_RLHGNN"] + [
    f"pred_fixed_{choice}" for choice in FIXED_METHODS
]


def get_device(gpu):
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f"cuda:{gpu}")
    print("警告：CUDA 不可用，回退到 CPU（大测试集逐前缀推理会非常慢）")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_ckpt(path):
    path = Path(path)
    if path.exists():
        return path
    if Path(f"{path}.zip").exists():
        return Path(f"{path}.zip")
    raise FileNotFoundError(f"找不到权重：{path}（也没有 .zip）")


def make_loader(dataset, batch_size, device):
    from dgl.dataloading import GraphDataLoader

    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def predict_loader(model, loader, device):
    """按顺序跑完一个 loader，返回 (labels, preds)，顺序 == 数据集索引顺序。"""
    all_labels = []
    all_preds = []
    with torch.inference_mode():
        for batch in loader:
            batch = [item.to(device) for item in batch]
            graphs, labels = batch
            if device.type == "cuda":
                with autocast():
                    logits = model(graphs)
            else:
                logits = model(graphs)
            all_labels.append(labels.detach().cpu())
            all_preds.append(logits.argmax(1).detach().cpu())
    return torch.cat(all_labels).numpy().astype(np.int64), torch.cat(all_preds).numpy().astype(np.int64)


def load_fixed_model(dataset, fold, seed, choice, args, device):
    ckpt = (
        Path(args.fixed_root)
        / dataset
        / f"action_{choice}"
        / f"seed{seed}"
        / f"{dataset}_fold{fold}_seed{seed}_model.pkl"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"缺少固定结构权重：{ckpt}（先跑 ablation_main.py）")
    model = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def load_rl_hgnn_model(dataset, fold, seed, args, device):
    ckpt = Path(args.final_root) / dataset / f"seed{seed}" / f"{dataset}_fold{fold}_seed{seed}_model.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(f"缺少 RLHGNN 权重：{ckpt}（先跑 final_main.py）")
    model = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def export_ppo_actions(dataset, fold, seed, args, device):
    """逐前缀 deterministic 选出图结构，观测构造与 final_policy.extract_features 完全一致。"""
    from stable_baselines3 import PPO

    from final_policy import final_policy

    raw_path = Path(args.raw_root) / f"{dataset}_{fold}"
    features_path = raw_path / "features_name.npy"
    if not features_path.exists():
        raise FileNotFoundError(f"缺少 {features_path}")

    with open(features_path, "rb") as handle:
        features_name = pickle.load(handle)

    node_features = {
        name: np.load(raw_path / f"{name}_{fold}_test.npy", allow_pickle=True) for name in features_name
    }
    labels = np.load(raw_path / f"label_{fold}_test.npy", allow_pickle=True)

    ckpt = resolve_ckpt(Path(args.rl_root) / dataset / f"PPO_best_model_fold{fold}_seed{seed}")
    policy = PPO.load(str(ckpt), device=device)
    extractor = final_policy(dataset, fold)

    actions = np.empty(len(labels), dtype=np.int64)
    for index in range(len(labels)):
        sample = {name: node_features[name][index] for name in features_name}
        action, _ = policy.predict(extractor.extract_features(sample), deterministic=True)
        action = int(action)
        if action < 0 or action >= len(FIXED_METHODS):
            raise ValueError(f"PPO 输出非法动作 {action}（{dataset} seed{seed} 第 {index} 条）")
        actions[index] = action

    del policy
    gc.collect()
    return actions, np.asarray(labels, dtype=np.int64)


def check_labels(dataset, seed, labels, expected, name):
    if len(labels) != len(expected):
        raise ValueError(f"{dataset} seed{seed}: {name} 样本数 {len(labels)} != 清单 {len(expected)}")
    mismatch = int((np.asarray(labels) != expected).sum())
    if mismatch:
        raise ValueError(
            f"{dataset} seed{seed}: {name} 的真实标签与清单不一致 {mismatch}/{len(expected)} 行；"
            f"raw_dir / graph_data 与本清单不是同一批预处理产物"
        )


def export_dataset(dataset, args, device):
    """一个数据集：图只加载一次，种子内轮换模型，最后按种子落 CSV。"""
    import pandas as pd

    from MyDataset_ablation import MyDataset as AblationDataset
    from MyDataset_final import MyDataset as FinalDataset

    manifest_path = Path(args.output_root) / "prefix_index" / f"{dataset}_fold{args.fold}_test_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少 {manifest_path}；先运行 prefix_group_build.py")
    manifest = pd.read_csv(manifest_path)
    expected_labels = manifest["label"].to_numpy().astype(np.int64)
    n_rows = len(manifest)

    out_dir = Path(args.output_root) / "preds" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer = {
        seed: {
            "prefix_index": manifest["prefix_index"].to_numpy(),
            "prefix_id": manifest["prefix_id"].to_numpy(),
            "ppo_action": np.full(n_rows, -1, dtype=np.int64),
        }
        for seed in args.seeds
    }
    notes = [f"dataset={dataset} fold={args.fold} prefixes={n_rows}"]

    # ---- 1) 固定结构：每个 action 的图加载一次，内部轮换种子 ----
    if not args.skip_fixed:
        for choice in FIXED_METHODS:
            print(f"  [fixed {choice}] 加载 {dataset} 测试图 ...", flush=True)
            dgl_dataset = AblationDataset(name=f"{dataset}_{args.fold}", type="test", choice=choice)
            for seed in args.seeds:
                # 图已在内存里，重建 loader 只是重排 batch，不会再读图；这样不依赖
                # DataLoader 能否重复迭代。
                loader = make_loader(dgl_dataset, args.batch_size, device)
                model = load_fixed_model(dataset, args.fold, seed, choice, args, device)
                labels, preds = predict_loader(model, loader, device)
                check_labels(dataset, seed, labels, expected_labels, f"fixed_{choice}")
                buffer[seed][f"pred_fixed_{choice}"] = preds
                notes.append(f"seed={seed} fixed_{choice}: rows={len(preds)} labels_ok=1")
                del model, loader
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del dgl_dataset
            gc.collect()

    # ---- 2) PPO 动作与 RLHGNN：与种子强相关，逐种子处理 ----
    for seed in args.seeds:
        set_seed(seed)
        if not args.skip_ppo:
            actions, labels = export_ppo_actions(dataset, args.fold, seed, args, device)
            check_labels(dataset, seed, labels, expected_labels, "ppo(raw test)")
            buffer[seed]["ppo_action"] = actions
            notes.append(f"seed={seed} ppo: rows={len(actions)} labels_ok=1")

        if not args.skip_rl_hgnn:
            dgl_dataset = FinalDataset(name=f"{dataset}_{args.fold}_seed{seed}", type="test")
            loader = make_loader(dgl_dataset, args.batch_size, device)
            model = load_rl_hgnn_model(dataset, args.fold, seed, args, device)
            labels, preds = predict_loader(model, loader, device)
            check_labels(dataset, seed, labels, expected_labels, "RLHGNN")
            buffer[seed]["pred_RLHGNN"] = preds
            notes.append(f"seed={seed} RLHGNN: rows={len(preds)} labels_ok=1")
            del model, loader, dgl_dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- 3) 落盘 + 整体准确率对账 ----
    for seed in args.seeds:
        frame = pd.DataFrame(buffer[seed])
        frame["y_true"] = expected_labels
        for column in OUTPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = -1
        frame = frame[OUTPUT_COLUMNS]
        csv_path = out_dir / f"seed{seed}.csv"
        frame.to_csv(csv_path, index=False)

        overall = []
        for method, column in [("RLHGNN", "pred_RLHGNN")] + [
            (FIXED_METHODS[choice], f"pred_fixed_{choice}") for choice in FIXED_METHODS
        ]:
            values = frame[column].to_numpy()
            # -1 是「本方法没导出」的哨兵值；预测值 0 属于标签空间外，是真实预测，
            # 不能因为出现过 0 就把整列当成缺失跳过。
            if np.all(values != -1):
                overall.append(f"{method}={float((values == expected_labels).mean()):.4f}")
        overall_text = "  overall_accuracy: " + ("  ".join(overall) if overall else "skipped")

        action_text = "  ppo_actions=skipped"
        if frame["ppo_action"].min() >= 0:
            counts = np.bincount(frame["ppo_action"].to_numpy(), minlength=4)
            action_text = "  ppo_actions: " + "  ".join(
                f"{FIXED_METHODS[choice]}={int(counts[choice])}" for choice in FIXED_METHODS
            )

        notes.append(f"seed={seed} rows={len(frame)} -> {csv_path}")
        notes.append(overall_text)
        notes.append(action_text)
        print(f"  seed={seed} rows={len(frame)} -> {csv_path}")
        print(overall_text)
        print(action_text)

    with open(out_dir / "_validation.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(notes) + "\n")
    print(f"  validation -> {out_dir / '_validation.txt'}")


def main():
    parser = argparse.ArgumentParser(description="实验二步骤2：导出 RLHGNN / 4 固定结构的逐前缀预测")
    parser.add_argument("-d", "--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("-s", "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--raw-root", default="raw_dir")
    parser.add_argument("--fixed-root", default="ablation_exp", help="消融（固定结构）权重根目录")
    parser.add_argument("--final-root", default="final_train", help="RLHGNN 最终模型根目录")
    parser.add_argument("--rl-root", default="RL_model", help="PPO 权重根目录")
    parser.add_argument("--output-root", default="exp2_repeat_groups")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-fixed", action="store_true", help="不导出 4 个固定结构")
    parser.add_argument("--skip-rl-hgnn", action="store_true", help="不导出 RLHGNN")
    parser.add_argument("--skip-ppo", action="store_true", help="不导出 PPO 动作")
    args = parser.parse_args()

    device = get_device(args.gpu)
    print(f"device={device}")

    for dataset in args.datasets:
        print(f"[{dataset}] 开始导出 ...", flush=True)
        export_dataset(dataset, args, device)


if __name__ == "__main__":
    main()
