#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验二 / 步骤 3：按重复模式分组统计预测指标 + 统计分析

输入
----
    exp2_repeat_groups/prefix_index/<dataset>_fold0_test_manifest.csv   （prefix_group_build.py）
    exp2_repeat_groups/preds/<dataset>/seed<seed>.csv                   （prefix_group_export_preds.py）

输出（exp2_repeat_groups/metrics/）
----------------------------------
    per_seed.csv            每个 (数据集, 分组, 方法, 种子) 的案例数/前缀数/Accuracy/Macro-F1
    summary.csv             跨种子 mean ± std
    mcnemar.csv             每个 (数据集, 分组, 种子, 基线) 的逐前缀配对检验
    wilcoxon.csv            每个 (数据集, 分组, 基线) 的跨种子配对检验
    did_bootstrap.csv       「优势是否集中在复杂重复前缀」的 DiD（按 case 聚类 bootstrap）
    action_distribution.csv PPO 动作分布（含 Forward-only / Bidirectional 拓扑族）
    report.md               上述结果的排版汇总

口径说明
--------
* Macro-F1 的标签空间默认取**数据集完整标签集**（1..N，N=raw_dir 的 activity_info），
  zero_division=0；某分组里没出现的类别以 F1=0 计入均值，因此分组之间、方法之间可比。
  模型若预测出 padding 类（编码 0，不属于标签空间），计入 Accuracy 错误但不进入 Macro-F1，
  条数单独记在 per_seed.csv 的 pred_out_of_space 列（正常应接近 0）。
* 分组只取清单中 verified=1 的行（默认清单已严格校验，等价于全部测试前缀）。
* 配对检验按前缀配对（McNemar 精确二项检验），跨种子检验用 Wilcoxon 符号秩检验。
  种子数=5 时 Wilcoxon 的最小可能 p 为 0.0625，因此「5 个种子方向一致但不显著」属正常，
  报告同时给出逐种子 McNemar 的显著种子数。
* DiD（difference-in-differences）= [RLHGNN − 固定结构] 在复杂分组上的差值
  − 同一差值在 G1 上的取值。DiD>0 且区间不含 0，才说明相对优势确实集中在复杂重复前缀。

用法
----
    python prefix_group_metrics.py -d bpi12_all_complete -d bpi13_incidents
    python prefix_group_metrics.py --include-all --bootstrap 0        # 只要描述统计
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np

GROUP_ORDER = ("G1", "G2", "G3", "G4")
COMPLEX_GROUPS = ("G3", "G4", "G3G4")

METHODS = ("RLHGNN", "A0_forward", "A1_bidirect", "A2_forward_complex", "A3_bidirect_complex")
FIXED_METHODS = METHODS[1:]
PRED_COLUMN = {
    "RLHGNN": "pred_RLHGNN",
    "A0_forward": "pred_fixed_0",
    "A1_bidirect": "pred_fixed_1",
    "A2_forward_complex": "pred_fixed_2",
    "A3_bidirect_complex": "pred_fixed_3",
}

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


# ---------------------------------------------------------------------------
# 指标核心（与 sklearn 的 macro 口径一致，支持权重，便于聚类 bootstrap）
# ---------------------------------------------------------------------------
def weighted_accuracy_macro_f1(y_true, y_pred, n_classes, weights=None):
    """y_true / y_pred 均为 0 基类别下标；y_pred == -1 表示落在标签空间外。"""
    if weights is None:
        weights = np.ones(len(y_true), dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if np.any(y_true < 0) or np.any(y_true >= n_classes):
        raise ValueError("真实标签越界，说明标签空间设置错了")

    valid = (y_pred >= 0) & (y_pred < n_classes)
    yp = np.where(valid, y_pred, -1)
    correct = valid & (yp == y_true)

    total_weight = float(weights.sum())
    accuracy = float(weights[correct].sum() / total_weight) if total_weight > 0 else 0.0

    tp = np.bincount(yp[correct], weights=weights[correct], minlength=n_classes)
    pred_count = np.bincount(yp[valid], weights=weights[valid], minlength=n_classes)
    true_count = np.bincount(y_true, weights=weights, minlength=n_classes)
    denom = pred_count + true_count
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-12), 0.0)
    return accuracy, float(f1.mean())


def mcnemar_exact_p(b, c):
    """McNemar 精确检验（双侧二项检验，p=q=0.5），b/c 是两种不一致方向的计数。

    在对数空间逐项累加：大测试集（如 BPI12C 有 2.7 万前缀）里 b+c 常上千，
    直接算 2.0 ** n 会 OverflowError，math.comb 也会退化成上千位大整数。
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))

    log_terms = np.empty(k + 1, dtype=np.float64)
    log_terms[0] = -n * math.log(2.0)  # ln C(n,0)/2^n
    if k:
        index = np.arange(1, k + 1, dtype=np.float64)
        # C(n,i)/2^n = C(n,i-1)/2^n * (n-i+1)/i
        log_terms[1:] = log_terms[0] + np.cumsum(np.log((n - index + 1.0) / index))

    peak = float(log_terms.max())
    tail = math.exp(peak) * float(np.exp(log_terms - peak).sum())
    return float(min(1.0, 2.0 * tail))


def _average_ranks(values):
    """秩（1 基），并列取平均秩，与 scipy.stats.rankdata 默认行为一致。"""
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and values[order[end + 1]] == values[order[start]]:
            end += 1
        if end > start:
            ranks[order[start:end + 1]] = (start + end + 2) / 2.0
        start = end + 1
    return ranks


def wilcoxon_p(a, b):
    """Wilcoxon 符号秩检验（双侧）的 p 值，零差值按 scipy 的 "wilcox" 方式丢弃。

    不依赖 scipy：本机 scipy 编译于 numpy1，import 直接失败，若退回 nan，
    报告里 p 值就全是 nan 了。m <= 16 时枚举全部 2^m 个符号组合给精确 p
    （n=5 个种子时只有 32 种组合，是真正的精确检验）；更大的 m 用带连续性
    与并列校正的正态近似。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("wilcoxon_p 需要按种子配对的两组等长样本")

    diff = a - b
    diff = diff[np.abs(diff) > 1e-12]
    m = len(diff)
    if m == 0:
        return 1.0

    ranks = _average_ranks(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    center = m * (m + 1) / 4.0
    observed = abs(w_plus - center)

    if m <= 16:
        # 所有符号组合下的 W+ 值：逐个秩做"子集和"递推
        sums = np.zeros(1 << m, dtype=np.float64)
        for k in range(m):
            block = 1 << k
            sums[block:2 * block] = sums[:block] + ranks[k]
        return float(np.mean(np.abs(sums - center) >= observed - 1e-9))

    _, tie_counts = np.unique(np.abs(diff), return_counts=True)
    tie_term = float(np.sum(tie_counts.astype(np.float64) ** 3 - tie_counts))
    variance = m * (m + 1) * (2 * m + 1) / 24.0 - tie_term / 48.0
    if variance <= 0:
        return 1.0
    z = max(observed - 0.5, 0.0) / math.sqrt(variance)
    return float(min(1.0, math.erfc(z / math.sqrt(2.0))))


# ---------------------------------------------------------------------------
# 读入与对齐
# ---------------------------------------------------------------------------
def load_dataset(dataset, root, raw_root, fold, include_all, split="test"):
    """返回清单上下文：真实标签、分组掩码、case 编码，全部与清单行顺序对齐。"""
    import pandas as pd

    manifest_path = Path(root) / "prefix_index" / f"{dataset}_fold{fold}_{split}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少 {manifest_path}；先运行 prefix_group_build.py")
    manifest = pd.read_csv(manifest_path)
    manifest["case_id"] = manifest["case_id"].astype(str)
    if "verified" in manifest.columns:
        dropped = int((manifest["verified"] == 0).sum())
        if dropped:
            print(f"[{dataset}] 剔除 {dropped} 行未通过校验的前缀")
            manifest = manifest[manifest["verified"] == 1].reset_index(drop=True)

    class_path = Path(raw_root) / f"{dataset}_{fold}" / "activity_info.npy"
    if not class_path.exists():
        raise FileNotFoundError(f"缺少 {class_path}（用于确定标签空间）")
    with open(class_path, "rb") as handle:
        n_classes = int(pickle.load(handle))

    group_masks = {group: (manifest["group"] == group).to_numpy() for group in GROUP_ORDER}
    group_masks["G3G4"] = group_masks["G3"] | group_masks["G4"]
    if include_all:
        group_masks["ALL"] = np.ones(len(manifest), dtype=bool)

    case_codes, _ = pd.factorize(manifest["case_id"])
    return {
        "manifest": manifest,
        "prefix_index": manifest["prefix_index"].to_numpy(),
        "group": manifest["group"].to_numpy(),
        "case_codes": case_codes.astype(np.int64),
        "y_true": manifest["label"].to_numpy().astype(np.int64) - 1,
        "group_masks": group_masks,
        "n_classes": n_classes,
    }


def load_predictions(dataset, root, fold, seeds, context):
    """读每个种子的预测，并按 prefix_index 严格对齐到清单顺序。"""
    import pandas as pd

    preds_dir = Path(root) / "preds" / dataset
    prefix_index = context["prefix_index"]
    label = context["manifest"]["label"].to_numpy().astype(np.int64)
    results = {}

    for seed in seeds:
        path = preds_dir / f"seed{seed}.csv"
        if not path.exists():
            print(f"[{dataset}] 跳过 seed{seed}：缺少 {path}")
            continue
        frame = pd.read_csv(path)
        frame = frame.set_index("prefix_index")
        missing = np.setdiff1d(prefix_index, frame.index.to_numpy())
        if len(missing):
            raise ValueError(
                f"{path} 缺少 {len(missing)} 个清单里的 prefix_index（例如 {missing[:5].tolist()}），"
                f"preds 与 manifest 不是同一批产物"
            )
        frame = frame.loc[prefix_index]
        if not np.array_equal(frame["y_true"].to_numpy().astype(np.int64), label):
            raise ValueError(f"{path} 的真实标签与清单不一致")

        present = []
        for method in METHODS:
            column = PRED_COLUMN[method]
            if column not in frame.columns:
                continue
            # 导出脚本用 -1 表示「该方法没跑」；真实预测是 1..N，其中 0 表示模型
            # 落在了标签空间之外（padding 类），仍算「有预测」，不能当成缺失丢掉。
            if np.any(frame[column].to_numpy() != -1):
                present.append(method)
        if not present:
            print(f"[{dataset}] 跳过 seed{seed}：没有任何预测列")
            continue

        data = {"ppo_action": frame["ppo_action"].to_numpy().astype(np.int64)}
        for method in present:
            data[method] = frame[PRED_COLUMN[method]].to_numpy().astype(np.int64)
        results[seed] = {"present": present, **data}

    return results


# ---------------------------------------------------------------------------
# 描述统计
# ---------------------------------------------------------------------------
def compute_per_seed(dataset, seeds, predictions, context):
    rows = []
    y_true = context["y_true"]
    n_classes = context["n_classes"]
    case_codes = context["case_codes"]

    for seed in seeds:
        if seed not in predictions:
            continue
        data = predictions[seed]
        for group, mask in context["group_masks"].items():
            n_prefixes = int(mask.sum())
            if n_prefixes == 0:
                continue
            n_cases = int(len(np.unique(case_codes[mask])))
            for method in data["present"]:
                y_pred = data[method][mask] - 1
                accuracy, macro_f1 = weighted_accuracy_macro_f1(y_true[mask], y_pred, n_classes)
                rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "method": method,
                        "seed": seed,
                        "n_prefixes": n_prefixes,
                        "n_cases": n_cases,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "pred_out_of_space": int((y_pred < 0).sum()),
                    }
                )
    return rows


def summarise(per_seed_rows):
    import pandas as pd

    frame = pd.DataFrame(per_seed_rows)
    if frame.empty:
        return frame

    rows = []
    for (dataset, group, method), sub in frame.groupby(["dataset", "group", "method"], sort=False):
        accuracy = sub["accuracy"].to_numpy()
        macro_f1 = sub["macro_f1"].to_numpy()
        rows.append(
            {
                "dataset": dataset,
                "group": group,
                "method": method,
                "n_seeds": len(sub),
                "n_prefixes": int(sub["n_prefixes"].iloc[0]),
                "n_cases": int(sub["n_cases"].iloc[0]),
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                "macro_f1_mean": float(macro_f1.mean()),
                "macro_f1_std": float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 统计检验
# ---------------------------------------------------------------------------
def paired_tests(dataset, seeds, predictions, context):
    """逐前缀 McNemar（每个种子）+ 跨种子 Wilcoxon。"""
    mcnemar_rows = []
    wilcoxon_rows = []
    y_true = context["y_true"]
    n_classes = context["n_classes"]

    for group, mask in context["group_masks"].items():
        if int(mask.sum()) == 0:
            continue
        for baseline in FIXED_METHODS:
            acc_rl, acc_base, f1_rl, f1_base = [], [], [], []
            for seed in seeds:
                if seed not in predictions:
                    continue
                data = predictions[seed]
                if "RLHGNN" not in data["present"] or baseline not in data["present"]:
                    continue

                y_rl = data["RLHGNN"][mask] - 1
                y_base = data[baseline][mask] - 1
                correct_rl = y_rl == y_true[mask]
                correct_base = y_base == y_true[mask]

                b = int(np.sum(correct_rl & ~correct_base))
                c = int(np.sum(~correct_rl & correct_base))
                mcnemar_rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "baseline": baseline,
                        "seed": seed,
                        "n_prefixes": int(mask.sum()),
                        "rl_correct_baseline_wrong": b,
                        "rl_wrong_baseline_correct": c,
                        "delta_accuracy": (b - c) / max(int(mask.sum()), 1),
                        "p_mcnemar": mcnemar_exact_p(b, c),
                    }
                )

                accuracy_rl, macro_f1_rl = weighted_accuracy_macro_f1(y_true[mask], y_rl, n_classes)
                accuracy_base, macro_f1_base = weighted_accuracy_macro_f1(y_true[mask], y_base, n_classes)
                acc_rl.append(accuracy_rl)
                acc_base.append(accuracy_base)
                f1_rl.append(macro_f1_rl)
                f1_base.append(macro_f1_base)

            if len(acc_rl) < 2:
                continue
            wilcoxon_rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "baseline": baseline,
                    "n_seeds": len(acc_rl),
                    "delta_accuracy_mean": float(np.mean(acc_rl) - np.mean(acc_base)),
                    "delta_macro_f1_mean": float(np.mean(f1_rl) - np.mean(f1_base)),
                    "p_wilcoxon_accuracy": wilcoxon_p(acc_rl, acc_base),
                    "p_wilcoxon_macro_f1": wilcoxon_p(f1_rl, f1_base),
                }
            )
    return mcnemar_rows, wilcoxon_rows


def bootstrap_did(dataset, seeds, predictions, context, n_boot, rng_seed):
    """按 case 聚类的 bootstrap：同一 case 的前缀同进同出，保留案例内相关性。"""
    rows = []
    if n_boot <= 0:
        return rows

    y_true = context["y_true"]
    n_classes = context["n_classes"]
    case_codes = context["case_codes"]
    n_cases = int(case_codes.max()) + 1
    needed = {group: context["group_masks"][group] for group in ("G1", "G3", "G4", "G3G4")}
    empty = [group for group, mask in needed.items() if int(mask.sum()) == 0]
    if empty:
        print(f"[{dataset}] 跳过 DiD：分组 {empty} 没有前缀")
        return rows

    for seed in seeds:
        if seed not in predictions:
            continue
        data = predictions[seed]
        if "RLHGNN" not in data["present"]:
            continue
        baselines = [method for method in FIXED_METHODS if method in data["present"]]
        if not baselines:
            continue

        methods = ["RLHGNN", *baselines]
        yp = {method: data[method] - 1 for method in methods}

        point = {
            (group, method): weighted_accuracy_macro_f1(y_true[mask], yp[method][mask], n_classes)
            for group, mask in needed.items()
            for method in methods
        }

        rng = np.random.default_rng(rng_seed + seed)
        probability = np.full(n_cases, 1.0 / n_cases)
        acc_draws = {(group, method): np.empty(n_boot) for group in needed for method in methods}
        f1_draws = {(group, method): np.empty(n_boot) for group in needed for method in methods}

        for draw in range(n_boot):
            weights = rng.multinomial(n_cases, probability)[case_codes].astype(np.float64)
            for group, mask in needed.items():
                for method in methods:
                    accuracy, macro_f1 = weighted_accuracy_macro_f1(
                        y_true[mask], yp[method][mask], n_classes, weights[mask]
                    )
                    acc_draws[(group, method)][draw] = accuracy
                    f1_draws[(group, method)][draw] = macro_f1

        for baseline in baselines:
            for group in COMPLEX_GROUPS:
                for metric, draws, column in (
                    ("accuracy", acc_draws, 0),
                    ("macro_f1", f1_draws, 1),
                ):
                    did_draws = (draws[(group, "RLHGNN")] - draws[(group, baseline)]) - (
                        draws[("G1", "RLHGNN")] - draws[("G1", baseline)]
                    )
                    low, high = np.percentile(did_draws, [2.5, 97.5])
                    did_point = (
                        point[(group, "RLHGNN")][column]
                        - point[(group, baseline)][column]
                        - point[("G1", "RLHGNN")][column]
                        + point[("G1", baseline)][column]
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "group": group,
                            "reference_group": "G1",
                            "baseline": baseline,
                            "metric": metric,
                            "delta_on_group": point[(group, "RLHGNN")][column]
                            - point[(group, baseline)][column],
                            "delta_on_reference": point[("G1", "RLHGNN")][column]
                            - point[("G1", baseline)][column],
                            "did": did_point,
                            "did_bootstrap_mean": float(did_draws.mean()),
                            "ci_low": float(low),
                            "ci_high": float(high),
                            "p_bootstrap": float(min(1.0, 2.0 * min((did_draws <= 0).mean(), (did_draws >= 0).mean()))),
                            "n_boot": n_boot,
                        }
                    )
    return rows


def action_distribution(dataset, seeds, predictions, context):
    rows = []
    group_of_prefix = context["group"]

    for seed in seeds:
        if seed not in predictions:
            continue
        actions = predictions[seed]["ppo_action"]
        if np.any(actions < 0):
            continue
        for group in GROUP_ORDER:
            values = actions[group_of_prefix == group]
            total = len(values)
            if total == 0:
                continue
            counts = np.bincount(values, minlength=4)
            for action in range(4):
                rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "seed": seed,
                        "action": action,
                        "method": METHODS[action + 1],
                        "count": int(counts[action]),
                        "total": total,
                        "ratio": counts[action] / total,
                    }
                )
            for family, members in (("Forward-only(0+2)", (0, 2)), ("Bidirectional(1+3)", (1, 3))):
                count = int(counts[list(members)].sum())
                rows.append(
                    {
                        "dataset": dataset,
                        "group": group,
                        "seed": seed,
                        "action": "family",
                        "method": family,
                        "count": count,
                        "total": total,
                        "ratio": count / total,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def write_report(path, datasets, summary, mcnemar, wilcoxon, did, actions):
    import pandas as pd

    summary_frame = pd.DataFrame(summary)
    mcnemar_frame = pd.DataFrame(mcnemar)
    wilcoxon_frame = pd.DataFrame(wilcoxon)
    did_frame = pd.DataFrame(did)
    action_frame = pd.DataFrame(actions)

    lines = ["# 实验二：不同重复模式下的前缀性能", ""]
    lines.append(
        "分组基于模型实际输入窗口（去掉左侧 padding，超过 mean_trace 截断）："
        "G1 无重复 / G2 仅相邻重复 / G3 仅非相邻重复 / G4 两者兼有。"
    )
    lines.append("")

    for dataset in datasets:
        sub = summary_frame[summary_frame["dataset"] == dataset] if not summary_frame.empty else pd.DataFrame()
        if sub.empty:
            lines.append(f"## {dataset}")
            lines.append("")
            lines.append("（无预测文件，跳过）")
            lines.append("")
            continue

        present_groups = set(sub["group"])
        keep = list(GROUP_ORDER) + (["ALL"] if "ALL" in present_groups else [])
        sizes = sub.drop_duplicates("group")
        sizes = sizes[sizes["group"].isin(keep)][["group", "n_prefixes", "n_cases"]]
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(
            "分组规模：" + "，".join(f"{r.group}={r.n_prefixes} 前缀/{r.n_cases} 案例" for r in sizes.itertuples())
        )
        lines.append("")
        lines.append("（同一案例可同时贡献多个分组的前缀，案例数不可跨组相加；G3G4=G3∪G4 只用于 DiD）")
        lines.append("")

        columns = [group for group in GROUP_ORDER if group in set(sub["group"])]
        if "ALL" in present_groups:
            columns.append("ALL")

        for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
            lines.append(f"**{label}（各种子 mean±std）**")
            lines.append("")
            lines.append("| 方法 | " + " | ".join(columns) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in columns) + " |")
            for method in METHODS:
                cells = []
                for group in columns:
                    row = sub[(sub["group"] == group) & (sub["method"] == method)]
                    cells.append("-" if row.empty else _fmt(row[f"{metric}_mean"].iloc[0], row[f"{metric}_std"].iloc[0]))
                lines.append(f"| {method} | " + " | ".join(cells) + " |")
            lines.append("")

        if not mcnemar_frame.empty:
            subset = mcnemar_frame[mcnemar_frame["dataset"] == dataset]
            if not subset.empty:
                lines.append("**RLHGNN vs 固定结构：逐前缀 McNemar（显著种子数/种子数，括号内为 ΔAccuracy 均值）**")
                lines.append("")
                lines.append("| 基线 | " + " | ".join(columns) + " |")
                lines.append("| --- | " + " | ".join("---" for _ in columns) + " |")
                for baseline in FIXED_METHODS:
                    cells = []
                    for group in columns:
                        rows = subset[(subset["group"] == group) & (subset["baseline"] == baseline)]
                        if rows.empty:
                            cells.append("-")
                        else:
                            significant = int((rows["p_mcnemar"] < 0.05).sum())
                            cells.append(f"{significant}/{len(rows)} ({rows['delta_accuracy'].mean():+.4f})")
                    lines.append(f"| {baseline} | " + " | ".join(cells) + " |")
                lines.append("")

        if not wilcoxon_frame.empty:
            subset = wilcoxon_frame[wilcoxon_frame["dataset"] == dataset]
            if not subset.empty:
                lines.append("**跨种子 Wilcoxon 符号秩检验（n=5 时最小 p=0.0625）**")
                lines.append("")
                lines.append("| 分组 | 基线 | ΔAcc | p | ΔMacro-F1 | p |")
                lines.append("| --- | --- | --- | --- | --- | --- |")
                for r in subset.itertuples():
                    lines.append(
                        f"| {r.group} | {r.baseline} | {r.delta_accuracy_mean:+.4f} | {r.p_wilcoxon_accuracy:.3f} "
                        f"| {r.delta_macro_f1_mean:+.4f} | {r.p_wilcoxon_macro_f1:.3f} |"
                    )
                lines.append("")

        if not did_frame.empty:
            subset = did_frame[did_frame["dataset"] == dataset]
            if not subset.empty:
                lines.append(
                    "**优势集中性 DiD = [RLHGNN−基线]@复杂组 − [RLHGNN−基线]@G1（按 case 聚类 bootstrap）**"
                )
                lines.append("")
                lines.append("| 分组 | 基线 | 指标 | DiD（均值） | 精度区间不含0的种子数 |")
                lines.append("| --- | --- | --- | --- | --- |")
                for (group, baseline, metric), rows in subset.groupby(["group", "baseline", "metric"], sort=False):
                    excludes_zero = int(((rows["ci_low"] > 0) | (rows["ci_high"] < 0)).sum())
                    lines.append(
                        f"| {group} | {baseline} | {metric} | {rows['did'].mean():+.4f} | {excludes_zero}/{len(rows)} |"
                    )
                lines.append("")

        if not action_frame.empty:
            subset = action_frame[action_frame["dataset"] == dataset]
            if not subset.empty:
                ratio = subset.groupby(["group", "method"])["ratio"].agg(["mean"]).reset_index()
                lines.append("**PPO 动作选择比例（跨种子均值）**")
                lines.append("")
                header = [METHODS[i + 1] for i in range(4)] + ["Forward-only(0+2)", "Bidirectional(1+3)"]
                lines.append("| 分组 | " + " | ".join(header) + " |")
                lines.append("| --- | " + " | ".join("---" for _ in header) + " |")
                for group in GROUP_ORDER:
                    cells = []
                    for method in header:
                        row = ratio[(ratio["group"] == group) & (ratio["method"] == method)]
                        cells.append("-" if row.empty else f"{row['mean'].iloc[0]:.3f}")
                    lines.append(f"| {group} | " + " | ".join(cells) + " |")
                lines.append("")

    lines.extend(
        [
            "## 注意事项",
            "",
            "- 每个数据集只有 5 个种子：跨种子 Wilcoxon 的最小可能 p 为 0.0625，"
            "不要把「不显著」当成「无差异」；逐前缀 McNemar 的显著种子数更能反映稳定性。",
            "- 逐前缀 McNemar 把同一案例的多个前缀当独立样本，p 值偏乐观；要严格推断"
            "「优势是否集中在复杂重复前缀」，看 did_bootstrap.csv 里按 case 聚类的 DiD 区间。",
            "- Macro-F1 使用数据集完整标签集（zero_division=0），分组之间、方法之间可直接比较；"
            "预测落在 padding 类（编码 0）的条数记在 per_seed.csv 的 pred_out_of_space 列。",
            "- 结果依赖 raw_dir / final_train / ablation_exp / RL_model 是同一批产物；"
            "前缀清单与预测导出两步都会做一致性校验。",
            "",
        ]
    )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="实验二步骤3：分组指标与统计分析")
    parser.add_argument("-d", "--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("-s", "--seeds", nargs="+", type=int, default=[133, 188, 456, 789, 1666])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="清单的 split（预测导出脚本目前只导 test）")
    parser.add_argument("--root", default="exp2_repeat_groups", help="前两步的输出根目录")
    parser.add_argument("--raw-root", default="raw_dir", help="用于读取 activity_info（标签空间）")
    parser.add_argument("--bootstrap", type=int, default=1000, help="DiD 聚类 bootstrap 次数，0 表示跳过")
    parser.add_argument("--bootstrap-seed", type=int, default=2024)
    parser.add_argument(
        "--include-all", action="store_true", help="额外输出整体（ALL）一行，便于与 metrics_final.py 对账"
    )
    args = parser.parse_args()

    import pandas as pd

    output_dir = Path(args.root) / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_all, mcnemar_all, wilcoxon_all, did_all, actions_all = [], [], [], [], []
    used_datasets = []

    for dataset in args.datasets:
        try:
            context = load_dataset(dataset, args.root, args.raw_root, args.fold, args.include_all, args.split)
        except FileNotFoundError as error:
            print(f"跳过 {dataset}：{error}")
            continue

        predictions = load_predictions(dataset, args.root, args.fold, args.seeds, context)
        if not predictions:
            print(f"跳过 {dataset}：没有可用的预测文件")
            continue

        per_seed_rows = compute_per_seed(dataset, args.seeds, predictions, context)
        if not per_seed_rows:
            print(f"跳过 {dataset}：预测列不完整")
            continue

        used_datasets.append(dataset)
        per_seed_all.extend(per_seed_rows)

        mcnemar_rows, wilcoxon_rows = paired_tests(dataset, args.seeds, predictions, context)
        mcnemar_all.extend(mcnemar_rows)
        wilcoxon_all.extend(wilcoxon_rows)
        did_all.extend(
            bootstrap_did(dataset, args.seeds, predictions, context, args.bootstrap, args.bootstrap_seed)
        )
        actions_all.extend(action_distribution(dataset, args.seeds, predictions, context))

        print(f"[{dataset}] 完成：{len(per_seed_rows)} 条 (分组×方法×种子) 指标，"
              f"{len(mcnemar_rows)} 条 McNemar")

    if not per_seed_all:
        raise SystemExit("没有任何可统计的结果：请先跑完前两步")

    per_seed_frame = pd.DataFrame(per_seed_all)
    summary_frame = summarise(per_seed_all)

    per_seed_frame.to_csv(output_dir / "per_seed.csv", index=False)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(mcnemar_all).to_csv(output_dir / "mcnemar.csv", index=False)
    pd.DataFrame(wilcoxon_all).to_csv(output_dir / "wilcoxon.csv", index=False)
    pd.DataFrame(did_all).to_csv(output_dir / "did_bootstrap.csv", index=False)
    pd.DataFrame(actions_all).to_csv(output_dir / "action_distribution.csv", index=False)

    write_report(
        output_dir / "report.md",
        used_datasets,
        summary_frame.to_dict("records"),
        mcnemar_all,
        wilcoxon_all,
        did_all,
        actions_all,
    )

    print(f"\n输出目录：{output_dir}")
    for name in (
        "per_seed.csv",
        "summary.csv",
        "mcnemar.csv",
        "wilcoxon.csv",
        "did_bootstrap.csv",
        "action_distribution.csv",
        "report.md",
    ):
        print(f"  {name}")


if __name__ == "__main__":
    main()
