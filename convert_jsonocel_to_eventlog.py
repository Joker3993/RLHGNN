from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd


try:
    from pm4py.objects.ocel.importer.jsonocel import importer as jsonocel_importer
    from pm4py.objects.ocel.util import flattening
except ImportError as exc:  # pragma: no cover
    # 如果本地没有安装 pm4py，这里直接给出明确提示，避免后面运行时报更难懂的错误。
    raise ImportError(
        "未安装 pm4py。请先执行: pip install pm4py"
    ) from exc


@dataclass
class DatasetConfig:
    # 一个数据集的转换配置：输入文件、输出文件、以及主对象类型。
    name: str
    input_path: Path
    output_path: Path
    case_object_type: Optional[str] = None


def _detect_case_object_type(ocel, preferred: Optional[str] = None) -> str:
    # 先读取 OCEL 里所有对象类型，再决定最终用哪一种作为 case 视角。
    object_types = sorted(list(ocel.objects["ocel:type"].dropna().unique()))

    # 如果外部已经指定了主对象类型，并且它确实存在，就直接采用。
    if preferred and preferred in object_types:
        return preferred

    # 没有显式指定时，按常见的业务主对象类型优先尝试。
    priority = ["orders", "purchase_order", "case", "application", "order"]
    for p in priority:
        if p in object_types:
            return p

    if not object_types:
        raise ValueError("无法在 OCEL 中识别对象类型。")

    return object_types[0]


def _normalize_traditional_log(df_flat: pd.DataFrame) -> pd.DataFrame:
    # pm4py flatten 后通常会得到传统事件日志所需的三列：case / activity / timestamp。
    col_case = "case:concept:name"
    col_activity = "concept:name"
    col_time = "time:timestamp"

    if col_case not in df_flat.columns:
        raise ValueError("flatten 后数据缺少 case:concept:name 列")
    if col_activity not in df_flat.columns:
        raise ValueError("flatten 后数据缺少 concept:name 列")
    if col_time not in df_flat.columns:
        raise ValueError("flatten 后数据缺少 time:timestamp 列")

    # 复制一份，避免直接改动原始 flatten 结果。
    base = df_flat.copy()

    # 将时间戳统一格式化，方便后续排序和写入 CSV。
    ts = pd.to_datetime(base[col_time], errors="coerce", utc=True)
    base[col_time] = ts.dt.tz_convert(None).dt.strftime("%Y/%m/%d %H:%M:%S.%f").str.slice(0,-4)

    # 将核心列放在前面，其余列保留原名称与原顺序，不做额外重命名。
    ordered_cols = [col_case, col_activity, col_time] + [
        c for c in base.columns if c not in {col_case, col_activity, col_time}
    ]
    base = base[ordered_cols]

    # 按 case 和时间排序，保证同一轨迹内事件顺序正确。
    base = base.sort_values([col_case, col_time, col_activity], kind="stable").reset_index(drop=True)

    # 对三个核心列进行重命名
    rename_map = {
        col_case: "case",
        col_activity: "activity",
        col_time: "timestamp",
        "case:ocel:type": "type",
        "case:eid": "eid",
        "case: Document Type(EKKO - BSART)":"Document Type",
        "case:Purchasing Group (EKKO-EKGRP)":"Purchasing Group",
        "case:Release Status (EKKO-FRGZU)":"Release Status"     ,
        "case:Purchasing Organization (EKKO-EKORG)":"Purchasing Organization",
        "case:Vendor (EKKO-LIFNR)":"Vendor"
    }

    base = base.rename(columns=rename_map)

    # 删除所有值都为空的列，减少无用数据
    base = base.dropna(axis=1, how="all")
    return base


def convert_one_dataset(cfg: DatasetConfig) -> Dict[str, str]:
    # 单个数据集的完整转换流程：读取 -> flatten -> 整理 -> 保存。
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {cfg.input_path}")

    # 读取 JSONOCEL。
    ocel = jsonocel_importer.apply(str(cfg.input_path))
    # 决定当前数据集使用哪个对象类型作为 case 视角。
    case_object_type = _detect_case_object_type(ocel, cfg.case_object_type)

    # 按指定 case 对象类型将 OCEL 压平为传统事件日志表。
    df_flat = flattening.flatten(ocel, case_object_type)

    # 统一格式、排序并整理输出列。
    df_eventlog = _normalize_traditional_log(df_flat)

    # 保存 CSV。
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    df_eventlog.to_csv(cfg.output_path, index=False, encoding="utf-8-sig")

    return {
        "dataset": cfg.name,
        "input": str(cfg.input_path),
        "output": str(cfg.output_path),
        "case_object_type": case_object_type,
        "rows": str(len(df_eventlog)),
        "columns": str(len(df_eventlog.columns)),
    }


def build_default_configs(project_root: Path) -> Sequence[DatasetConfig]:
    # 当前项目默认要处理的两个数据集，以及它们对应的主对象类型。
    return [
        DatasetConfig(
            name="OTC",
            input_path=project_root / "train_test_data" / "OTC"  / "OTC.jsonocel",
            output_path=project_root / "train_test_data" / "OTC"  / "OTC.csv",
            case_object_type="items",
        ),
        DatasetConfig(
            name="p2p",
            input_path=project_root / "train_test_data" / "p2p"  / "p2p.jsonocel",
            output_path=project_root / "train_test_data" / "p2p"  / "p2p.csv",
            case_object_type="purchase_order",
        ),
    ]




def main() -> None:
    # 脚本入口：顺序处理每个数据集并打印结果摘要。
    project_root = Path(__file__).resolve().parent
    configs = build_default_configs(project_root)

    print("开始进行 jsonOCEL -> 传统事件日志 CSV 转换...\n")
    for cfg in configs:
        result = convert_one_dataset(cfg)
        print(
            f"[{result['dataset']}] 完成: "
            f"case对象={result['case_object_type']}, "
            f"行数={result['rows']}, 列数={result['columns']}\n"
            f"输入: {result['input']}\n"
            f"输出: {result['output']}\n"
        )


if __name__ == "__main__":
    main()
