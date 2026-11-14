#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验二 / 步骤 1：测试前缀清单（prefix manifest）与重复模式分组 G1-G4

用途
----
为「不同重复模式下的前缀性能分析」建立唯一、可复现的前缀索引：每个测试前缀用
(dataset, case_id, prefix_end_event_index) 唯一标识，并给出该前缀在**模型实际输入窗口**
上的重复模式分组。后续两步都只认这份清单：

    prefix_group_build.py     -> exp2_repeat_groups/prefix_index/    （本脚本，无 GPU 依赖）
    prefix_group_export_preds.py -> exp2_repeat_groups/preds/        （需要 torch + dgl + 权重）
    prefix_group_metrics.py   -> exp2_repeat_groups/metrics/         （只吃上面两步的 CSV）

分组口径
--------
window = 模型真正消费的 activity 向量去掉左侧 padding(=0) 之后的部分，即
min(前缀长度, mean_trace) 个事件（mean_trace 取自 raw_dir/<dataset>_0/mean_trace.npy）。
前缀长度超过 mean_trace 时窗口会截断，分组按截断后的窗口判定，因为那才是模型看到的东西。

    G1  无重复活动                      A -> B -> C
    G2  仅相邻重复                      A -> A -> B
    G3  仅非相邻重复                    A -> B -> A
    G4  相邻与非相邻重复同时存在        A -> A -> B -> A

判定方式：
    has_adjacent_repeat     = 窗口中存在 w[i] == w[i-1]
    has_nonadjacent_repeat  = 先把连续相同活动压缩成一段，压缩序列中仍存在重复活动
说明：非相邻重复只表示「同一活动在非相邻位置再次出现」，不等同于业务返工，
除非另有事件语义证据。

一致性校验（为什么需要）
----------------------
上游 ProcessEventlog_one_graph.py 按 case 顺序 + case 内事件顺序生成前缀，但不落盘
case_id；这里按同一顺序重建，并逐行验证：
    1. 前缀总数 = sum(len(case) - 1)
    2. npy 窗口去掉 padding 后 == CSV 中该前缀最后 min(len, mean_trace) 个事件的编码
    3. npy 标签 == 该前缀下一个事件的编码
任何一行不满足都会报错（默认中断，--allow-mismatch 可降级为告警并把 verified 记为 0）。
这一步失败通常意味着 raw_dir 不是当前代码/数据生成的，需先重跑 data_process.py。

用法
----
    python prefix_group_build.py                       # 默认 7 个数据集，fold 0，test
    python prefix_group_build.py -d bpi12_all_complete -d bpi13_incidents
    python prefix_group_build.py --output-root my_exp
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np

# 与 ProcessEventlog_one_graph.ENCODING_MAP 保持一致。
# 这里复制一份而不是 import，是为了避免本步骤连带导入 dgl（本步骤纯 CPU、无需 GPU）。
ENCODING_MAP = {
    "bpi13_closed_problems": "gbk",
}

GROUP_ORDER = ("G1", "G2", "G3", "G4")

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
# 纯逻辑：窗口 -> 重复模式
# ---------------------------------------------------------------------------
def strip_padding(window):
    """去掉左侧 padding(0)。编码从 1 开始（见 utils.encode_map），因此 0 只可能是填充。"""
    values = [int(v) for v in window]
    start = 0
    while start < len(values) and values[start] == 0:
        start += 1
    return values[start:]


def classify_window(window):
    """按模型实际看得见的窗口（已去除 padding）判定重复模式分组。"""
    visible = strip_padding(window)
    if not visible:
        raise ValueError("encountered an all-padding window; window length is 0")

    has_adjacent = any(visible[i] == visible[i - 1] for i in range(1, len(visible)))

    compressed = []
    max_run = 0
    current_run = 0
    for value in visible:
        if compressed and compressed[-1] == value:
            current_run += 1
        else:
            compressed.append(value)
            current_run = 1
        if current_run > max_run:
            max_run = current_run

    has_nonadjacent = len(compressed) != len(set(compressed))

    if has_adjacent and has_nonadjacent:
        group = "G4"
    elif has_adjacent:
        group = "G2"
    elif has_nonadjacent:
        group = "G3"
    else:
        group = "G1"

    return {
        "group": group,
        "has_adjacent_repeat": int(has_adjacent),
        "has_nonadjacent_repeat": int(has_nonadjacent),
        "window_len": len(visible),
        "num_distinct": len(set(visible)),
        "max_run": max_run,
        "visible": visible,
    }


# ---------------------------------------------------------------------------
# 清单构建（纯逻辑，便于单独测试）
# ---------------------------------------------------------------------------
def _register(encode, token, code):
    code = int(code)
    previous = encode.get(token)
    if previous is None:
        encode[token] = code
        return True
    return previous == code


def build_manifest_rows(
    case_sequences,
    activity_matrix,
    labels,
    mean_trace,
    n_classes,
    dataset,
    fold=0,
    split="test",
    strict=True,
    max_reported_mismatch=5,
):
    """由 (case_id, 活动序列) 列表 + npy 矩阵重建带 case_id 的前缀清单。

    case_sequences: [(case_id, [activity_token, ...]), ...]，顺序必须与上游生成 npy 时一致
    activity_matrix: raw_dir 里的 activity_<fold>_<split>.npy（定长 padding 窗口）
    labels:          raw_dir 里的 label_<fold>_<split>.npy
    n_classes:       raw_dir 里 activity_info.npy 的取值（标签空间 1..n_classes）
    """
    mean_trace = int(mean_trace)

    expected_rows = sum(max(len(seq) - 1, 0) for _, seq in case_sequences)

    label_values = np.asarray([int(value) for value in labels], dtype=np.int64)
    if len(label_values) and (label_values.min() < 1 or label_values.max() > n_classes):
        raise ValueError(
            f"{dataset}: 标签取值 [{label_values.min()}, {label_values.max()}] 超出 1..{n_classes}，"
            f"activity_info.npy 与标签不是同一批产物"
        )

    if len(activity_matrix) != expected_rows or len(labels) != expected_rows:
        raise ValueError(
            f"{dataset}: prefix count mismatch — case sequences imply {expected_rows}, "
            f"activity npy has {len(activity_matrix)}, label npy has {len(labels)}. "
            f"raw_dir 与本脚本读到的 split CSV 不是同一次预处理产物。"
        )

    # ---- 第一遍：从对齐位置反推 activity 字符串 -> 编码（编码是双射，可完整恢复） ----
    encode = {}
    cursor = 0
    for _, seq in case_sequences:
        for j in range(len(seq) - 1):
            visible = strip_padding(activity_matrix[cursor])
            if not visible:
                raise ValueError(f"{dataset}: row {cursor} is an all-padding window")
            if not _register(encode, seq[j], visible[-1]):
                raise ValueError(
                    f"{dataset}: activity encoding conflict for {seq[j]!r} "
                    f"(expected {encode[seq[j]]}, npy says {int(visible[-1])}) at row {cursor}. "
                    f"前缀顺序与 npy 不对齐。"
                )
            if not _register(encode, seq[j + 1], labels[cursor]):
                raise ValueError(
                    f"{dataset}: label encoding conflict for {seq[j + 1]!r} at row {cursor}. "
                    f"前缀顺序与 npy 不对齐。"
                )
            cursor += 1

    # ---- 第二遍：逐行严格校验（窗口内容 + 截断 + padding + 标签） ----
    rows = []
    mismatches = []
    cursor = 0
    for case_index, (case_id, seq) in enumerate(case_sequences):
        case_len = len(seq)
        for j in range(case_len - 1):
            window = activity_matrix[cursor]
            pattern = classify_window(window)
            visible = pattern.pop("visible")

            prefix_len = j + 1
            expected_window_len = min(prefix_len, mean_trace)
            problems = []

            if len(window) != mean_trace:
                problems.append(f"window length {len(window)} != mean_trace {mean_trace}")
            if len(visible) != expected_window_len:
                problems.append(f"visible length {len(visible)} != {expected_window_len}")
            else:
                start = prefix_len - expected_window_len
                tokens = seq[start:prefix_len]
                unknown = [token for token in tokens if token not in encode]
                expected_codes = [encode.get(token) for token in tokens]
                if unknown:
                    problems.append(f"活动 {unknown} 未出现在任何窗口末位，无法还原编码")
                elif expected_codes != visible:
                    problems.append(f"window {visible} != {expected_codes}")
            expected_label = encode.get(seq[j + 1])
            if expected_label is None:
                problems.append(f"下一活动 {seq[j + 1]!r} 未出现在任何标签位，无法还原编码")
            elif int(labels[cursor]) != expected_label:
                problems.append(f"label {int(labels[cursor])} != {expected_label}")

            if problems:
                if strict:
                    raise ValueError(
                        f"{dataset}: row {cursor} (case {case_id}, end index {j}) failed validation: "
                        + "; ".join(problems)
                    )
                if len(mismatches) < max_reported_mismatch:
                    mismatches.append((cursor, case_id, j, "; ".join(problems)))

            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "split": split,
                    "prefix_index": cursor,
                    "prefix_id": f"{dataset}|{case_id}|{j}",
                    "case_id": case_id,
                    "case_index": case_index,
                    "case_len": case_len,
                    "prefix_end_event_index": j,
                    "prefix_len": prefix_len,
                    "window_len": pattern["window_len"],
                    "padding_len": mean_trace - pattern["window_len"],
                    "truncated": int(prefix_len > mean_trace),
                    "group": pattern["group"],
                    "has_adjacent_repeat": pattern["has_adjacent_repeat"],
                    "has_nonadjacent_repeat": pattern["has_nonadjacent_repeat"],
                    "num_distinct": pattern["num_distinct"],
                    "max_run": pattern["max_run"],
                    "window_activities": " > ".join(seq[prefix_len - pattern["window_len"]:prefix_len]),
                    "label": int(labels[cursor]),
                    "label_activity": seq[j + 1],
                    "verified": int(not problems),
                }
            )
            cursor += 1

    return rows, {"expected_rows": expected_rows, "mismatches": mismatches, "verification_errors": int(sum(1 - r["verified"] for r in rows))}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def read_case_sequences(csv_path, encoding):
    """按上游同样的顺序读回 (case_id, 活动序列)。

    顺序很关键：ProcessEventlog_one_graph.timestamp_transform() 用
    ``event_log.groupby('case')``（默认 sort=True）逐案例处理后 pd.concat 回去，
    因此后续 groupby('case', sort=False) 生成前缀时，案例顺序其实是**按 case 键排序**后的
    顺序，而不是 CSV 里的时间顺序。这里用默认 sort=True 的 groupby 复刻同一顺序；
    案例内事件顺序不动（CSV 内已按时间排好）。
    """
    import pandas as pd  # 延迟导入：只有 IO 需要 pandas，便于纯逻辑部分单独复用

    df = pd.read_csv(csv_path, sep=",", header=0, index_col=False, encoding=encoding)
    for column in ("case", "activity"):
        if column not in df.columns:
            raise ValueError(f"{csv_path} 缺少列 {column!r}")

    sequences = []
    for case_id, group in df.groupby("case"):
        sequences.append((str(case_id), [str(value) for value in group["activity"].tolist()]))
    return sequences


def build_dataset_manifest(dataset, fold, split, raw_root, strict=True):
    raw_path = os.path.join(raw_root, f"{dataset}_{fold}")
    activity_path = os.path.join(raw_path, f"activity_{fold}_{split}.npy")
    label_path = os.path.join(raw_path, f"label_{fold}_{split}.npy")
    csv_path = os.path.join(raw_path, f"{dataset}_{split}.csv")
    mean_trace_path = os.path.join(raw_path, "mean_trace.npy")
    class_path = os.path.join(raw_path, "activity_info.npy")

    for path in (activity_path, label_path, csv_path, mean_trace_path, class_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少 {path}；请先运行 data_process.py")

    activity_matrix = np.load(activity_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    with open(mean_trace_path, "rb") as handle:
        mean_trace = int(pickle.load(handle))
    with open(class_path, "rb") as handle:
        n_classes = int(pickle.load(handle))

    sequences = read_case_sequences(csv_path, ENCODING_MAP.get(dataset, "utf-8"))
    rows, report = build_manifest_rows(
        sequences,
        activity_matrix,
        labels,
        mean_trace,
        n_classes,
        dataset=dataset,
        fold=fold,
        split=split,
        strict=strict,
    )
    report.update(
        {
            "dataset": dataset,
            "fold": fold,
            "split": split,
            "n_cases": len(sequences),
            "n_prefixes": len(rows),
            "mean_trace": mean_trace,
            "n_classes": n_classes,
        }
    )
    return rows, report


def summarise(rows, report):
    counts = {group: 0 for group in GROUP_ORDER}
    for row in rows:
        counts[row["group"]] += 1
    total = len(rows) or 1
    return {
        "dataset": report["dataset"],
        "split": report["split"],
        "n_cases": report["n_cases"],
        "n_prefixes": report["n_prefixes"],
        "mean_trace": report["mean_trace"],
        "n_classes": report["n_classes"],
        "truncated": sum(row["truncated"] for row in rows),
        "adjusted": sum(row["has_adjacent_repeat"] for row in rows),
        **{f"{group}_count": counts[group] for group in GROUP_ORDER},
        **{f"{group}_ratio": round(counts[group] / total, 6) for group in GROUP_ORDER},
        "verification_errors": report["verification_errors"],
    }


def main():
    parser = argparse.ArgumentParser(description="实验二步骤1：构建测试前缀清单与 G1-G4 分组")
    parser.add_argument("-d", "--datasets", nargs="+", default=DEFAULT_DATASETS, help="数据集名（默认 7 个）")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="清单的 split（步骤2/3 的逐前缀导出目前只针对 test）",
    )
    parser.add_argument("--raw-root", default="raw_dir")
    parser.add_argument("--output-root", default="exp2_repeat_groups")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="校验失败时继续（不推荐）：verified=0 的行后续会被指标脚本剔除统计",
    )
    args = parser.parse_args()

    import pandas as pd

    index_dir = os.path.join(args.output_root, "prefix_index")
    os.makedirs(index_dir, exist_ok=True)

    summaries = []
    lines = []
    for dataset in args.datasets:
        rows, report = build_dataset_manifest(
            dataset=dataset,
            fold=args.fold,
            split=args.split,
            raw_root=args.raw_root,
            strict=not args.allow_mismatch,
        )
        frame = pd.DataFrame(rows)
        csv_path = os.path.join(index_dir, f"{dataset}_fold{args.fold}_{args.split}_manifest.csv")
        frame.to_csv(csv_path, index=False)

        summary = summarise(rows, report)
        summaries.append(summary)

        header = (
            f"[{dataset}] cases={summary['n_cases']} prefixes={summary['n_prefixes']} "
            f"mean_trace={summary['mean_trace']} classes={summary['n_classes']} "
            f"truncated={summary['truncated']} verification_errors={summary['verification_errors']}"
        )
        print(header)
        print(
            "  "
            + "  ".join(
                f"{group}={summary[f'{group}_count']} ({summary[f'{group}_ratio']:.1%})" for group in GROUP_ORDER
            )
        )
        print(f"  manifest -> {csv_path}")
        lines.append(header)
        lines.append(
            "  " + "  ".join(f"{group}={summary[f'{group}_count']}" for group in GROUP_ORDER)
        )
        if report["mismatches"]:
            for cursor, case_id, j, message in report["mismatches"]:
                lines.append(f"  mismatch row={cursor} case={case_id} end={j}: {message}")
                print(f"  WARNING mismatch row={cursor} case={case_id} end={j}: {message}")

    summary_path = os.path.join(index_dir, "prefix_group_summary.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    with open(os.path.join(index_dir, "prefix_group_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.write("\n备注：group 判定基于模型实际窗口（去掉左侧 padding，超过 mean_trace 截断）。\n")
        handle.write("prefix_id = dataset|case_id|prefix_end_event_index（0 基，指窗口最后一个事件的案例内下标）。\n")

    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
