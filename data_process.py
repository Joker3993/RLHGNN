import os
import pickle
import dgl
import numpy as np
import torch
from tqdm import tqdm

from ProcessEventlog_one_graph import PreProcess


def split_dataset(graph_list, labels):
    total_samples = len(graph_list)
    half_samples = total_samples // 2

    graph_part1 = graph_list[:half_samples]
    graph_part2 = graph_list[half_samples:]

    label_part1 = labels[:half_samples]
    label_part2 = labels[half_samples:]

    return graph_part1, label_part1, graph_part2, label_part2


def split_attr(graph_list):
    total_samples = len(graph_list)
    half_samples = total_samples // 2

    graph_part1 = graph_list[:half_samples]
    graph_part2 = graph_list[half_samples:]

    return graph_part1, graph_part2


def process_datasets(path):
    data_types = ["train", "val", "test"]
    for choice in range(4):

        for data_type in data_types:

            graph_path = os.path.join(path, f"Positive_graph_{data_type}_choice_{choice}")
            graph_list, labels_dict = dgl.load_graphs(graph_path)
            labels = labels_dict["labels"]

            graph_part1, label_part1, graph_part2, label_part2 = split_dataset(graph_list, labels)

            save_path_part1 = os.path.join(path, f"part1")
            save_path_part2 = os.path.join(path, f"part2")

            os.makedirs(save_path_part1, exist_ok=True)
            os.makedirs(save_path_part2, exist_ok=True)

            dgl.save_graphs(os.path.join(save_path_part1, f"Positive_graph_{data_type}_choice_{choice}"), graph_part1,
                            labels={"labels": torch.tensor(label_part1)})

            dgl.save_graphs(os.path.join(save_path_part2, f"Positive_graph_{data_type}_choice_{choice}"), graph_part2,
                            labels={"labels": torch.tensor(label_part2)})

            for name in features_name:
                feature_path = os.path.join(path, name + '_' + str(fold) + f"_{data_type}.npy")
                att_list = np.load(feature_path, allow_pickle=True)

                seq_part1, seq_part2 = split_attr(att_list)
                np.save(os.path.join(save_path_part1, name + '_' + str(fold) + f"_{data_type}.npy"), seq_part1)
                np.save(os.path.join(save_path_part2, name + '_' + str(fold) + f"_{data_type}.npy"), seq_part2)

            label_path = os.path.join(path, 'label' + '_' + str(fold) + f"_{data_type}.npy")
            label_list = np.load(label_path, allow_pickle=True)
            seq_part1, seq_part2 = split_attr(label_list)
            np.save(os.path.join(save_path_part1, 'label' + '_' + str(fold) + f"_{data_type}.npy"), seq_part1)
            np.save(os.path.join(save_path_part2, 'label' + '_' + str(fold) + f"_{data_type}.npy"), seq_part2)


if __name__ == '__main__':
    list_eventlog = [
        'bpi13_closed_problems',
        'bpi13_problems',
        'bpi13_incidents',
        'bpi12w_complete',
        'bpi12_all_complete',
        'BPI2020_Prepaid',
    ]

    for eventlog in tqdm(list_eventlog):
        print(f"--------------数据预处理------------")
        PreProcess(event_log=eventlog).main_process()

        for fold in range(3):
            path = "./raw_dir/" + eventlog + "_" + str(fold)

            features_name_path = path + "/" + "features_name" + ".npy"
            with open(features_name_path, 'rb') as file:
                features_name = pickle.load(file)

            process_datasets(path)
