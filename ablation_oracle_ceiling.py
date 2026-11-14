import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import precision_recall_fscore_support
from stable_baselines3 import PPO
from torch.cuda.amp import autocast

from final_policy import final_policy
from MyDataset_ablation import MyDataset


DEFAULT_DATASETS = [
    "bpi13_problems",
    "bpi13_incidents",
    "bpi12w_complete",
    "bpi12_all_complete",
    "BPI2020_Prepaid",
    "p2p",
    "bpi13_closed_problems",
]


def get_device(gpu):
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


def predict_one_choice(model, loader, device):
    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for batch in loader:
            batch = [item.to(device) for item in batch]
            positive_graph, labels = batch

            if device.type == "cuda":
                with autocast():
                    logits = model(positive_graph)
            else:
                logits = model(positive_graph)

            preds = logits.argmax(1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    return all_labels, all_preds


def predict_policy_actions(dataset_name, fold, seed, policy_root, device):
    model_path = Path(policy_root) / dataset_name / f"PPO_best_model_fold{fold}_seed{seed}"
    checkpoint_path = model_path if model_path.exists() else model_path.with_suffix(".zip")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing PPO checkpoint: {model_path}(.zip)")

    raw_path = Path("raw_dir") / f"{dataset_name}_{fold}"
    features_path = raw_path / "features_name.npy"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing raw dataset metadata: {features_path}")

    with open(features_path, "rb") as features_file:
        features_name = pickle.load(features_file)

    node_features = {
        name: np.load(raw_path / f"{name}_{fold}_test.npy", allow_pickle=True)
        for name in features_name
    }
    labels = np.load(raw_path / f"label_{fold}_test.npy", allow_pickle=True)
    samples = [
        {**{name: node_features[name][idx] for name in features_name}, "label": labels[idx]}
        for idx in range(len(labels))
    ]

    policy = PPO.load(str(checkpoint_path), device=device)
    feature_extractor = final_policy(dataset_name, fold)
    actions = []
    for sample in samples:
        observation = feature_extractor.extract_features(sample)
        action, _ = policy.predict(observation, deterministic=True)
        action = int(action)
        if action < 0 or action >= 4:
            raise ValueError(f"PPO returned invalid action {action} for {dataset_name}")
        actions.append(action)

    return np.asarray(actions, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def _per_class_f1(a, p, g):
    """标准 F1 = 2*TP / (2*TP + FP + FN) = 2a / (p + g)，a=TP, p=TP+FP, g=TP+FN"""
    denom = p + g
    return (2.0 * a / denom) if denom > 0 else 0.0


def _hill_climb_macro_f1(labels, pred_matrix, init_choices, class_to_idx, g_counts, max_rounds):
    """对给定初始选择做坐标上升爬山，最大化整体 macro-F1。"""
    n = len(labels)
    k_actions = pred_matrix.shape[1]
    m = len(class_to_idx)

    choices = np.asarray(init_choices, dtype=np.int64).copy()
    preds = pred_matrix[np.arange(n), choices]

    a_counts = np.zeros(m, dtype=np.float64)
    p_counts = np.zeros(m, dtype=np.float64)
    for i in range(n):
        p_counts[class_to_idx[preds[i]]] += 1.0
        if labels[i] == preds[i]:
            a_counts[class_to_idx[preds[i]]] += 1.0

    def current_score():
        f1s = np.empty(m, dtype=np.float64)
        for k in range(m):
            f1s[k] = _per_class_f1(a_counts[k], p_counts[k], g_counts[k])
        return float(f1s.mean())

    score = current_score()

    for _ in range(max_rounds):
        improved = False
        for i in range(n):
            cur = choices[i]
            cur_pred = preds[i]
            cur_idx = class_to_idx[cur_pred]
            y = labels[i]

            old_f_cur = _per_class_f1(a_counts[cur_idx], p_counts[cur_idx], g_counts[cur_idx])

            best_delta = 0.0
            best_action = cur

            for a in range(k_actions):
                if a == cur:
                    continue
                new_pred = pred_matrix[i, a]
                if new_pred == cur_pred:
                    continue
                new_idx = class_to_idx[new_pred]

                old_f_new = _per_class_f1(a_counts[new_idx], p_counts[new_idx], g_counts[new_idx])

                a_cur_new = a_counts[cur_idx] - (1.0 if y == cur_pred else 0.0)
                p_cur_new = p_counts[cur_idx] - 1.0
                a_new_new = a_counts[new_idx] + (1.0 if y == new_pred else 0.0)
                p_new_new = p_counts[new_idx] + 1.0

                delta = (
                    _per_class_f1(a_cur_new, p_cur_new, g_counts[cur_idx])
                    + _per_class_f1(a_new_new, p_new_new, g_counts[new_idx])
                    - old_f_cur
                    - old_f_new
                )
                if delta > best_delta + 1e-12:
                    best_delta = delta
                    best_action = a

            if best_action != cur:
                new_pred = pred_matrix[i, best_action]
                new_idx = class_to_idx[new_pred]

                p_counts[cur_idx] -= 1.0
                if y == cur_pred:
                    a_counts[cur_idx] -= 1.0
                p_counts[new_idx] += 1.0
                if y == new_pred:
                    a_counts[new_idx] += 1.0

                choices[i] = best_action
                preds[i] = new_pred
                improved = True

        if not improved:
            break

    return current_score(), preds, choices


def compute_f1_ceiling(labels, pred_matrix, n_restarts=16, max_rounds=300, seed=0):
    """
    计算严格的 macro-F1 天花板：
    每个样本可自由选择 4 个 action 的预测之一，使整体 macro-F1 最大化。
    该组合优化没有多项式精确解，此处用多起点坐标上升爬山搜索近似全局最优。
    返回 (best_f1, best_preds, best_choices)。
    """
    labels = np.asarray(labels, dtype=np.int64)
    pred_matrix = np.asarray(pred_matrix, dtype=np.int64)
    n = len(labels)
    k_actions = pred_matrix.shape[1]

    all_classes = sorted(set(labels.tolist()) | set(pred_matrix.flatten().tolist()))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    m = len(all_classes)

    g_counts = np.zeros(m, dtype=np.float64)
    for y in labels:
        g_counts[class_to_idx[y]] += 1.0

    rng = np.random.default_rng(seed)

    # 起点集合：4 个固定 action + per-sample 最优 + 随机起点
    starts = [np.full(n, a, dtype=np.int64) for a in range(k_actions)]

    acc_choices = np.zeros(n, dtype=np.int64)
    for i in range(n):
        correct = np.where(pred_matrix[i] == labels[i])[0]
        if len(correct) > 0:
            acc_choices[i] = correct[0]
    starts.append(acc_choices)

    for _ in range(max(n_restarts, 0)):
        starts.append(rng.integers(0, k_actions, size=n))

    best_f1 = -1.0
    best_preds = None
    best_choices = None
    for init in starts:
        f1, preds, choices = _hill_climb_macro_f1(labels, pred_matrix, init, class_to_idx, g_counts, max_rounds)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_preds = preds
            best_choices = choices

    return float(best_f1), best_preds, best_choices


def collect_oracle_table(dataset_name, seed, args):
    fold = 0
    device = get_device(args.gpu)

    dataset_key = f"{dataset_name}_{fold}"
    datasets = [
        MyDataset(name=dataset_key, type="test", choice=choice)
        for choice in range(4)
    ]

    lengths = {len(dataset) for dataset in datasets}
    if len(lengths) != 1:
        raise ValueError(f"Test sets are not aligned for {dataset_name}: {sorted(lengths)}")

    num_workers = 0 if os.name == "nt" else min(args.num_workers, os.cpu_count() or 1)
    loaders = [
        GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        for dataset in datasets
    ]

    models = []
    for choice in range(4):
        model_path = (
            Path(args.model_root)
            / dataset_name
            / f"action_{choice}"
            / f"seed{seed}"
            / f"{dataset_name}_fold{fold}_seed{seed}_model.pkl"
        )
        models.append(load_model(str(model_path), device))

    labels_by_choice = []
    preds_by_choice = []
    for choice in range(4):
        labels, preds = predict_one_choice(models[choice], loaders[choice], device)
        labels_by_choice.append(labels)
        preds_by_choice.append(preds)

    base_labels = np.asarray(labels_by_choice[0], dtype=np.int64)
    for idx in range(1, 4):
        current_labels = np.asarray(labels_by_choice[idx], dtype=np.int64)
        if not np.array_equal(base_labels, current_labels):
            raise ValueError(f"Label order mismatch between action 0 and action {idx} for {dataset_name}")

    pred_matrix = np.stack([np.asarray(preds, dtype=np.int64) for preds in preds_by_choice], axis=1)
    any_correct = (pred_matrix == base_labels[:, None]).any(axis=1)

    policy_actions, policy_labels = predict_policy_actions(
        dataset_name,
        fold,
        seed,
        args.policy_root,
        device,
    )
    if len(policy_actions) != len(base_labels):
        raise ValueError(
            f"Policy/test length mismatch for {dataset_name}: "
            f"{len(policy_actions)} vs {len(base_labels)}"
        )
    if not np.array_equal(base_labels, policy_labels):
        raise ValueError(f"Label order mismatch between ablation test data and raw test data for {dataset_name}")
    policy_preds = pred_matrix[np.arange(len(base_labels)), policy_actions]
    policy_correct = policy_preds == base_labels
    candidate_actions = [
        ",".join(str(action) for action in np.flatnonzero(row == label))
        for row, label in zip(pred_matrix, base_labels)
    ]
    candidate_count = int(any_correct.sum())
    selector_success_count = int((policy_correct & any_correct).sum())
    selector_success_rate = (
        float(selector_success_count / candidate_count) if candidate_count else 0.0
    )

    oracle_accuracy = float(any_correct.mean())

    # Accuracy 口径（每样本独立最优）：未命中样本固定用 action_0 的预测
    oracle_preds = pred_matrix[:, 0].copy()
    oracle_preds[any_correct] = base_labels[any_correct]

    # 旧口径 macro-F1（仅作对比）
    _, _, oracle_f1_naive, _ = precision_recall_fscore_support(
        base_labels,
        oracle_preds,
        average="macro",
        zero_division=0,
    )

    # 严格 F1 天花板：每个样本可自由选择任意 action 的预测，
    # 使整体 macro-F1 最大化（多起点坐标上升搜索）
    oracle_f1, oracle_f1_preds, _ = compute_f1_ceiling(
        base_labels,
        pred_matrix,
        n_restarts=args.f1_restarts,
        seed=args.f1_seed,
    )

    print(
        f"  oracle_accuracy={oracle_accuracy:.4f} "
        f"oracle_macro_f1_search={oracle_f1:.4f} "
        f"oracle_macro_f1_naive={oracle_f1_naive:.4f} "
        f"selector_success_rate={selector_success_rate:.4f}"
    )

    table = pd.DataFrame(
        {
            "prefix_id": np.arange(len(base_labels), dtype=np.int64),
            "true_label": base_labels,
            "pred_action_0": pred_matrix[:, 0],
            "pred_action_1": pred_matrix[:, 1],
            "pred_action_2": pred_matrix[:, 2],
            "pred_action_3": pred_matrix[:, 3],
            "correct_candidate_actions": candidate_actions,
            "num_correct_candidate_actions": (pred_matrix == base_labels[:, None]).sum(axis=1),
            "oracle_hit": any_correct.astype(np.int64),
            "oracle_pred_label": oracle_preds,
            "oracle_f1_pred_label": oracle_f1_preds,
            "policy_action": policy_actions,
            "policy_pred_label": policy_preds,
            "policy_correct": policy_correct.astype(np.int64),
        }
    )

    return (
        table,
        oracle_accuracy,
        float(oracle_f1),
        float(oracle_f1_naive),
        selector_success_rate,
        candidate_count,
        selector_success_count,
    )


def save_outputs(
    dataset_name,
    seed,
    table,
    oracle_accuracy,
    oracle_f1,
    oracle_f1_naive,
    selector_success_rate,
    candidate_count,
    selector_success_count,
    args,
):
    output_dir = Path(args.output_root) / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{dataset_name}_oracle_table.csv"
    summary_path = output_dir / f"{dataset_name}_oracle_summary.txt"

    table.to_csv(csv_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"dataset={dataset_name}\n")
        summary_file.write(f"seed={seed}\n")
        summary_file.write(f"rows={len(table)}\n")
        summary_file.write(f"oracle_accuracy={oracle_accuracy:.6f}\n")
        summary_file.write(f"oracle_macro_f1={oracle_f1:.6f}\n")
        summary_file.write(f"oracle_macro_f1_naive={oracle_f1_naive:.6f}\n")
        summary_file.write(f"candidate_prefixes={candidate_count}\n")
        summary_file.write(f"selector_success_count={selector_success_count}\n")
        summary_file.write(f"selector_success_rate={selector_success_rate:.6f}\n")
        summary_file.write(
            "note=accuracy_ceiling: per-sample best among 4 actions; "
            "f1_ceiling: multi-restart coordinate-ascent search over per-sample action choices "
            "(approx. global optimum); selector_success_rate: PPO selected an action whose "
            "fixed-structure prediction is correct, divided by prefixes with at least one "
            "correct candidate structure\n"
        )
        summary_file.write(f"csv_path={csv_path}\n")

    print(
        f"{dataset_name}: rows={len(table)} oracle_accuracy={oracle_accuracy:.4f} "
        f"oracle_macro_f1={oracle_f1:.4f} "
        f"selector_success_rate={selector_success_rate:.4f}"
    )
    print(f"Saved table to: {csv_path}")
    print(f"Saved summary to: {summary_path}")


def save_mean_outputs(dataset_name, seeds, accuracies, macro_f1s, selector_success_rates, args):
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{dataset_name}_oracle_mean.txt"

    acc_mean = float(np.mean(accuracies))
    acc_std = float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
    f1_mean = float(np.mean(macro_f1s))
    f1_std = float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0
    selector_mean = float(np.mean(selector_success_rates))
    selector_std = (
        float(np.std(selector_success_rates, ddof=1))
        if len(selector_success_rates) > 1
        else 0.0
    )

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"dataset={dataset_name}\n")
        summary_file.write(f"num_seeds={len(seeds)}\n")
        for seed, acc, f1, selector_rate in zip(seeds, accuracies, macro_f1s, selector_success_rates):
            summary_file.write(
                f"seed={seed} oracle_accuracy={acc:.6f} oracle_macro_f1={f1:.6f} "
                f"selector_success_rate={selector_rate:.6f}\n"
            )
        summary_file.write(f"oracle_accuracy_mean={acc_mean:.6f}\n")
        summary_file.write(f"oracle_accuracy_std={acc_std:.6f}\n")
        summary_file.write(f"oracle_macro_f1_mean={f1_mean:.6f}\n")
        summary_file.write(f"oracle_macro_f1_std={f1_std:.6f}\n")
        summary_file.write(f"selector_success_rate_mean={selector_mean:.6f}\n")
        summary_file.write(f"selector_success_rate_std={selector_std:.6f}\n")

    print(
        f"{dataset_name}: oracle_accuracy_mean={acc_mean:.4f}+/-{acc_std:.4f} "
        f"oracle_macro_f1_mean={f1_mean:.4f}+/-{f1_std:.4f} "
        f"selector_success_rate_mean={selector_mean:.4f}+/-{selector_std:.4f} (seeds={seeds})"
    )
    print(f"Saved mean summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation oracle ceiling evaluation")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="Datasets to evaluate")
    parser.add_argument("--seeds", nargs="*", type=int, default=[133, 188, 456, 789, 1666],
                        help="Checkpoint seeds to evaluate")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--model-root", type=str, default="ablation_exp", help="Root directory of ablation checkpoints")
    parser.add_argument("--policy-root", type=str, default="RL_model", help="Root directory of PPO checkpoints")
    parser.add_argument("--output-root", type=str, default="ablation_exp/oracle_ceiling", help="Directory for oracle tables")
    parser.add_argument("--f1-restarts", type=int, default=16, help="Random restarts for F1 ceiling search")
    parser.add_argument("--f1-seed", type=int, default=0, help="Random seed for F1 ceiling search")

    args = parser.parse_args()

    for dataset_name in args.datasets:
        accuracies = []
        macro_f1s = []
        selector_success_rates = []
        for seed in args.seeds:
            (
                table,
                oracle_accuracy,
                oracle_f1,
                oracle_f1_naive,
                selector_success_rate,
                candidate_count,
                selector_success_count,
            ) = collect_oracle_table(dataset_name, seed, args)
            save_outputs(
                dataset_name,
                seed,
                table,
                oracle_accuracy,
                oracle_f1,
                oracle_f1_naive,
                selector_success_rate,
                candidate_count,
                selector_success_count,
                args,
            )
            accuracies.append(oracle_accuracy)
            macro_f1s.append(oracle_f1)
            selector_success_rates.append(selector_success_rate)
        save_mean_outputs(dataset_name, args.seeds, accuracies, macro_f1s, selector_success_rates, args)


if __name__ == "__main__":
    main()