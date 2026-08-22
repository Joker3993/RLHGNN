# 导入所需的库
import gzip

import pandas as pd

import numpy as np
import os
import pm4py

from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom



"""普通划分方式"""
# 新增：不同数据集编码映射，需要gbk的添加在这里即可，默认使用utf-8
ENCODING_MAP = {
    'bpi13_closed_problems': 'gbk'
}
def cut_csv():

    """做了事件级别截断"""

    # 读取csv文件中的数据集
    encoding = ENCODING_MAP.get(event_name, 'utf-8')
    dataset= pd.read_csv("train_test_data/" + event_name + "/" + event_name + ".csv",
                           sep=',',
                           header=0, index_col=False,encoding=encoding)

    # 转换时间字段，便于按案例起始时间排序
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], errors='coerce')

    # 统计每个案例的起止时间；时间缺失的案例不参与切分
    case_time_df = dataset.groupby('case')['timestamp'].agg(['min', 'max']).rename(
        columns={'min': 'case_start_time', 'max': 'case_end_time'}
    )
    valid_case_time_df = case_time_df.dropna(subset=['case_start_time', 'case_end_time']).copy()

    # 按案例起始时间升序，最近20%案例划为测试集
    valid_case_time_df = valid_case_time_df.sort_values(by=['case_start_time', 'case_end_time'])
    ordered_cases = valid_case_time_df.index.tolist()
    case_count = len(ordered_cases)
    if dataset['case'].nunique()-case_count > 0:
        raise ValueError(f"{event_name} 中有{dataset['case'].nunique()-case_count}个无效案例（timestamp 全部缺失）")

    split_index = int(case_count * 0.8)
    train_cases_before_filter = ordered_cases[:split_index]
    test_cases = ordered_cases[split_index:]
    if len(test_cases) == 0:
        test_cases = ordered_cases[-1:]
        train_cases_before_filter = ordered_cases[:-1]

    # 以测试集最早开始时间作为边界，训练集只保留边界之前的事件
    test_start_time = valid_case_time_df.loc[test_cases, 'case_start_time'].min()

    # 同一案例内保持原始事件顺序不变
    grouped = dataset.groupby('case', sort=False)
    ordered_groups = [grouped.get_group(case_id) for case_id in ordered_cases]
    log_sorted = pd.concat(ordered_groups, ignore_index=True)

    # 仅在每个案例内部前向填充，避免跨案例污染
    log_sorted = log_sorted.groupby('case', sort=False, group_keys=False).apply(lambda group: group.ffill())

    train_before_filter = log_sorted[log_sorted['case'].isin(set(train_cases_before_filter))].copy()
    test = log_sorted[log_sorted['case'].isin(set(test_cases))].copy()

    train = train_before_filter[train_before_filter['timestamp'] < test_start_time].copy()

    train.index = range(len(train))
    test.index = range(len(test))

    # 输出切分统计
    train_before_count = len(train_cases_before_filter)
    train_after_count = train['case'].nunique()
    deleted_train_count = train_before_count - train_after_count
    deleted_event_count = len(train_before_filter) - len(train)
    print(f"\n[{event_name}] 切分统计")
    print("有效案例总数:", case_count)
    print("测试集案例数(最近20%):", len(test_cases))
    print("训练集案例数-删除前:", train_before_count)
    print("训练集案例数-删除后:", train_after_count)
    print("训练集删除案例数:", deleted_train_count)
    print("训练集删除事件数:", deleted_event_count)

    # 校验时间隔离：max(train event_time) < min(test case_start_time)
    if len(train) > 0 and len(test_cases) > 0:
        train_max_end = train['timestamp'].max()
        test_min_start = valid_case_time_df.loc[test_cases, 'case_start_time'].min()
        is_valid_split = train_max_end < test_min_start
        print("时间边界校验(max train_event_time < min test_start):", is_valid_split)
        print("max train_event_time:", train_max_end)
        print("min test_start:", test_min_start)
    else:
        print("时间边界校验跳过：训练集或测试集案例数为0")

    # 将切分后的数据集保存成新的excel文件
    output_dir = f"train_test_data/{event_name}"
    os.makedirs(output_dir, exist_ok=True)

    train['timestamp'] = train['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S.%f')
    test['timestamp'] = test['timestamp'].dt.strftime('%Y/%m/%d %H:%M:%S.%f')

    # Save the train and test datasets
    train.to_csv(f"{output_dir}/{event_name}_kfoldcv_0_train.csv", index=False)
    test.to_csv(f"{output_dir}/{event_name}_kfoldcv_0_test.csv", index=False)




if __name__ == '__main__':
    """切分数据集, 按案例数量切分前80%为训练集，后20%为测试集.不含验证集的划分"""
    list_event = [
        'bpi13_closed_problems',
        "bpi12_all_complete",
        "bpi12w_complete",
        'bpi13_incidents',
        'bpi13_problems',
        'BPI2020_Prepaid',
        'OTC',
        'p2p',
        # 'TestDatasetSmall'
    ]

    for eventlog in tqdm(list_event):

        event_name = eventlog
        cut_csv()

        print(f"\n{eventlog}完成切分")




