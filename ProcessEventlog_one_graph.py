import dgl

import warnings

warnings.filterwarnings(action='ignore')
from utils import *
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
import pandas as pd
import pickle
from tqdm import tqdm

# 新增：不同数据集编码映射，需要gbk的添加在这里即可，默认使用utf-8
ENCODING_MAP = {
    'bpi13_closed_problems': 'gbk'
}

class PreProcess:
    def __init__(self, event_log):
        self.eventlog = event_log
        self.n_bins = 0

    def get_seq_view(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])

                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(new_seq[-mean_trace:])

                j = j + 1
            i = i + 1
        return list_seq

    def get_sequence_label(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        list_label = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])

                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(new_seq[-mean_trace:])

                list_label.append(sequence.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_seq, list_label

    def timestamp_transform(self, event_log):

        groupProcessList = []

        for groupId, group in tqdm(event_log.groupby('case')):
            group = pd.concat([group.iloc[0:1], group])
            group['timestamp'] = pd.to_datetime(group['timestamp'])

            group['start_duration'] = (group['timestamp'].sub(group['timestamp'].min(), axis=0)).dt.total_seconds()[1:]
            group['start_duration'] = group['start_duration'].values / 86400

            group['duration'] = group['timestamp'].diff().dt.total_seconds()[1:]
            group['duration'] = group['duration'].values / 86400

            group = group.shift(periods=-1)

            group = group.iloc[:-1, :]
            group['duration'] = group['duration'].fillna(1)

            groupProcessList.append(group)

        edges_raw = pd.concat(groupProcessList)

        edges_raw.index = range(len(edges_raw))

        return edges_raw

    def time_features_process(self, train_df, val_df, test_df, time_col):

        X1 = train_df[time_col].to_numpy().reshape(-1, 1)
        X2 = val_df[time_col].to_numpy().reshape(-1, 1)
        X3 = test_df[time_col].to_numpy().reshape(-1, 1)

        change = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
        change.fit(X1)

        train_df[time_col] = change.transform(X1)
        val_df[time_col] = change.transform(X2)
        test_df[time_col] = change.transform(X3)

    def save_file(self, type, col, data, label, fold):

        with open(self.path + "/" + col + '_' + str(fold) + f"_{type}.npy", 'wb') as file:
            pickle.dump(data, file)

        if col == 'activity':
            with open(self.path + "/" + 'label_' + str(fold) + f"_{type}.npy", 'wb') as file:
                pickle.dump(label, file)

    def create_graph(self, fold, choice):
        print("-------------------开始创建顺序图---------------------")

        for type in ["train", "val", "test"]:
            print("-------前缀长度统一 ,异构图模式--------")

            node_feature = {}

            for name in self.list_cat_cols:
                feature_path = os.path.join(self.path, name + '_' + str(fold) + f"_{type}.npy")
                att_list = np.load(feature_path, allow_pickle=True)
                node_feature[name] = att_list

            label_path = os.path.join(self.path, 'label' + '_' + str(fold) + f"_{type}.npy")
            label = np.load(label_path, allow_pickle=True)

            PrefixList = node_feature['activity']
            Positive_graph_list = []

            edge_dict = {}

            for prefix in tqdm(PrefixList):
                nodeID = list(range(len(prefix)))

                if len(prefix) == 1:
                    src = [0]
                    dst = [0]
                    edge_dict[('node', 'forward', 'node')] = (src, dst)

                else:
                    src = nodeID[:-1]
                    dst = nodeID[1:]
                    edge_dict[('node', 'forward', 'node')] = (src, dst)

                index_one = get_index_of_duplicate_elements(prefix)

                if choice == 0:

                    edge_dict[('node', 'repeat_next', 'node')] = ([], [])

                    edge_dict[('node', 'backward', 'node')] = ([], [])
                    pass
                elif choice == 1:

                    edge_dict[('node', 'backward', 'node')] = (dst, src)

                    edge_dict[('node', 'repeat_next', 'node')] = ([], [])


                elif choice == 2:

                    edge_dict[('node', 'backward', 'node')] = ([], [])

                    edge_dict[('node', 'repeat_next', 'node')] = ([], [])

                    for index in index_one:
                        if (len(index) == 1):
                            continue

                        for i in range(len(index) - 1):
                            src = index[i]
                            for j in range(i + 1, len(index)):
                                dst = index[j]

                                if (dst + 1 >= len(prefix)):
                                    continue
                                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)

                        for i in range(len(index) - 1, -1, -1):
                            src = index[i]
                            for j in range(i - 1, -1, -1):
                                dst = index[j]

                                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)


                elif choice == 3:

                    edge_dict[('node', 'backward', 'node')] = (dst, src)

                    edge_dict[('node', 'repeat_next', 'node')] = ([], [])

                    for index in index_one:
                        if (len(index) == 1):
                            continue

                        for i in range(len(index) - 1):
                            src = index[i]
                            for j in range(i + 1, len(index)):
                                dst = index[j]

                                if (dst + 1 >= len(prefix)):
                                    continue

                                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)

                        for i in range(len(index) - 1, -1, -1):
                            src = index[i]
                            for j in range(i - 1, -1, -1):
                                dst = index[j]

                                edge_dict[('node', 'repeat_next', 'node')][0].append(src)
                                edge_dict[('node', 'repeat_next', 'node')][1].append(dst + 1)

                graph = dgl.heterograph(edge_dict)
                Positive_graph_list.append(graph)

            for feture_name in self.list_cat_cols:
                for i in range(len(Positive_graph_list)):
                    Positive_graph_list[i].ndata[feture_name] = torch.tensor(node_feature[feture_name][i])

            dgl.save_graphs(self.path + "/" + f"Positive_graph_{type}_choice_{choice}", Positive_graph_list,
                            labels={"labels": torch.tensor(label)})

    def main_process(self):

        # 单次时间拆分，不需要多折交叉验证，fold固定为0
        fold = 0
        encoding = ENCODING_MAP.get(self.eventlog, 'utf-8')

        self.list_num_cols = []
        self.list_cat_cols = []

        self.path = "raw_dir/" + self.eventlog + "_" + str(fold)

        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)

        # 切分结果保存目录
        save_dir = f"Dataset/{self.eventlog}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # # 1. 读取Dataset目录下的总数据集CSV文件
        # full_df = pd.read_csv(f"Dataset/{self.eventlog}/{self.eventlog}.csv", sep=',', header=0, index_col=False,encoding=encoding)

        # 1. 读取已切割好的训练集和测试集，拼接成完整数据集
        output_dir = f"train_test_data/{self.eventlog}"
        os.makedirs(output_dir, exist_ok=True)

        df_train_existing = pd.read_csv(f"{output_dir}/{self.eventlog}_kfoldcv_0_train.csv", sep=',', header=0, index_col=False, encoding=encoding)
        df_test_existing = pd.read_csv(f"{output_dir}/{self.eventlog}_kfoldcv_0_test.csv", sep=',', header=0, index_col=False, encoding=encoding)
        full_df = pd.concat([df_train_existing, df_test_existing], ignore_index=True)

        # 2. 统一转换时间戳格式
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')

        # 3. 按案例第一个事件的时间升序排序，保证训练集是较早历史，测试集是较晚未来
        case_start_time = full_df.groupby('case')['timestamp'].min().reset_index()
        case_start_time = case_start_time.sort_values('timestamp', ascending=True)
        sorted_case_ids = case_start_time['case'].tolist()

        # 按排序后的案例重新拼接，保持每个案例内部事件顺序不变
        sorted_groups = []
        for case_id in sorted_case_ids:
            # 保持案例内部事件按时间排序
            case_df = full_df[full_df['case'] == case_id].sort_values('timestamp', ascending=True)
            # 4. 同一个案例内缺失值前向填充
            case_df = case_df.ffill()
            sorted_groups.append(case_df)
        full_df = pd.concat(sorted_groups, ignore_index=True)

        # 5. 按案例数量8:2切分总数据集：前80%为训练验证集，后20%为测试集
        total_cases = len(sorted_case_ids)
        train_val_cases_count = int(total_cases * 0.8)
        train_val_case_ids = sorted_case_ids[:train_val_cases_count]
        test_case_ids = sorted_case_ids[train_val_cases_count:]

        df_train_val = full_df[full_df['case'].isin(train_val_case_ids)].copy()
        df_test = full_df[full_df['case'].isin(test_case_ids)].copy()

        # 训练验证集再按9:1切分出训练集和验证集，保持时间顺序不变
        train_val_case_count = len(train_val_case_ids)
        train_count = int(train_val_case_count * 0.9)
        train_case_ids = train_val_case_ids[:train_count]
        val_case_ids = train_val_case_ids[train_count:]
        df_train = df_train_val[df_train_val['case'].isin(train_case_ids)].copy()
        val_df = df_train_val[df_train_val['case'].isin(val_case_ids)].copy()

        # 6. 保存切分结果到指定目录
        df_train.to_csv(f"{save_dir}/{self.eventlog}_train.csv", index=False)
        val_df.to_csv(f"{save_dir}/{self.eventlog}_valid.csv", index=False)
        df_test.to_csv(f"{save_dir}/{self.eventlog}_test.csv", index=False)
        print(
            f"数据集切分完成！\n总案例数: {total_cases}, 训练案例数: {len(train_case_ids)}, 验证案例数: {len(val_case_ids)}, 测试案例数: {len(test_case_ids)}")

        # 后续原有预处理流程保持不变
        full_df = pd.concat([df_train, df_test])
        cont_trace = full_df['case'].value_counts(dropna=False)
        max_trace = max(cont_trace)
        mean_trace = int(round(np.mean(cont_trace)))
        self.n_bins = mean_trace

        with open(self.path + "/" + "max_trace" + ".npy", 'wb') as file:
            pickle.dump(max_trace, file)

        with open(self.path + "/" + "mean_trace" + ".npy", 'wb') as file:
            pickle.dump(mean_trace, file)

        print(f"最大长度：{max_trace}\n")
        print(f"平均长度：{mean_trace}\n")

        df_train.to_csv(self.path + "/" + self.eventlog + "_train.csv", index=False)
        val_df.to_csv(self.path + "/" + self.eventlog + "_valid.csv", index=False)
        df_test.to_csv(self.path + "/" + self.eventlog + "_test.csv", index=False)

        train_df = pd.read_csv(self.path + "/" + self.eventlog + "_train.csv", sep=',', header=0, index_col=False,encoding=encoding)
        val_df = pd.read_csv(self.path + "/" + self.eventlog + "_valid.csv", sep=',', header=0, index_col=False,encoding=encoding)
        test_df = pd.read_csv(self.path + "/" + self.eventlog + "_test.csv", sep=',', header=0, index_col=False,encoding=encoding)

        train_df = self.timestamp_transform(train_df)
        val_df = self.timestamp_transform(val_df)
        test_df = self.timestamp_transform(test_df)

        self.time_features_process(train_df, val_df, test_df, 'duration')
        self.time_features_process(train_df, val_df, test_df, 'start_duration')

        for col in train_df.columns.tolist():

            if (col == 'timestamp' or col == 'case'):
                pass

            else:

                train_df[col].fillna(method='ffill', inplace=True)
                val_df[col].fillna(method='ffill', inplace=True)
                test_df[col].fillna(method='ffill', inplace=True)

                total_data = pd.concat([train_df, val_df, test_df])

                att_encode_map = encode_map(set(total_data[col].values))

                train_df[col] = train_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
                val_df[col] = val_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
                test_df[col] = test_df[col].apply(lambda e: att_encode_map.get(str(e), -1))

                view_train = train_df.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_val = val_df.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_test = test_df.groupby('case', sort=False).agg({col: lambda x: list(x)})

                if col == "activity":

                    train_repeat = count_cases_with_repeated_activities(view_train['activity'].tolist())
                    val_repeat = count_cases_with_repeated_activities(view_val['activity'].tolist())
                    test_repeat = count_cases_with_repeated_activities(view_test['activity'].tolist())
                    with open(self.path + "/" + "total_repeat_case" + ".npy", 'wb') as file:
                        pickle.dump(train_repeat + val_repeat + test_repeat, file)
                    print(f"存在重复执行活动案例数：\n"
                          f"train :{train_repeat},val : {val_repeat} , test ：{test_repeat} ; \n"
                          f"total:{train_repeat + val_repeat + test_repeat}")

                    view_train, label_train = self.get_sequence_label(view_train, max_trace, mean_trace)
                    view_val, label_val = self.get_sequence_label(view_val, max_trace, mean_trace)
                    view_test, label_test = self.get_sequence_label(view_test, max_trace, mean_trace)

                    self.save_file('train', col, view_train, label_train, fold)
                    self.save_file('val', col, view_val, label_val, fold)
                    self.save_file('test', col, view_test, label_test, fold)

                    self.list_cat_cols.append(col)

                else:

                    view_train = self.get_seq_view(view_train, max_trace, mean_trace)
                    view_val = self.get_seq_view(view_val, max_trace, mean_trace)
                    view_test = self.get_seq_view(view_test, max_trace, mean_trace)

                    self.save_file('train', col, view_train, '', fold)
                    self.save_file('val', col, view_val, '', fold)
                    self.save_file('test', col, view_test, '', fold)

                    self.list_cat_cols.append(col)

        for col in train_df.columns.tolist():
            total_data = pd.concat([train_df, val_df, test_df])
            att_count = len(total_data[col].unique())
            print(f"{col}:{att_count}")

            with open(self.path + "/" + col + '_' + "info" + ".npy", 'wb') as file:
                pickle.dump(att_count, file)

        with open(self.path + "/" + "features_name" + ".npy", 'wb') as file:
            pickle.dump(self.list_cat_cols, file)

        print("特征名列表：\n", self.list_cat_cols)

        print("开始方案0构建")
        print("-" * 80)
        self.create_graph(fold, 0)
        print("开始方案1构建")
        print("-" * 80)
        self.create_graph(fold, 1)
        print("开始方案2构建")
        print("-" * 80)
        self.create_graph(fold, 2)
        print("开始方案3构建")
        print("-" * 80)
        self.create_graph(fold, 3)
