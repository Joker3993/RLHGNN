import os
import pickle
import numpy as np
from stable_baselines3 import DQN, PPO
import dgl
import torch
from tqdm import tqdm
from build_graph import build_graph
from utils import calculate_entropy, calculate_longest_consecutive_subsequence


class final_policy:
    def __init__(self, dataset, fold):
        self.eventlog = dataset
        self.fold = fold

    def get_device(self, gpu):

        if torch.cuda.is_available() and gpu < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu}')
        else:
            return torch.device('cpu')

    def extract_features(self, current_prefix_all_attr):

        activity = current_prefix_all_attr.get('activity', [])
        duration = current_prefix_all_attr.get('duration', [])

        length = len(activity)

        unique_activities = len(np.unique(activity))

        unique, counts = np.unique(activity, return_counts=True)
        activity_counts = counts.tolist()

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
            min_duration
        ]

        return features

    def generate_optimal_graphs(self, dataset, features_name, rl_model, device):

        optimal_graphs = []
        labels = []
        action_counts = [0] * 4

        rl_model.policy.eval()

        with torch.no_grad():
            for sample in tqdm(dataset):
                obs = self.extract_features(sample)

                action, _ = rl_model.predict(obs, deterministic=True)
                action = int(action)
                action_counts[action] += 1

                current_prefix = sample['activity']
                P_graph = build_graph(current_prefix, action, sample, features_name)

                optimal_graphs.append(P_graph)
                labels.append(sample['label'])

        return optimal_graphs, labels, action_counts

    def main(self):
        eventlog = self.eventlog
        fold = self.fold

        path = "./raw_dir/" + eventlog + "_" + str(fold)

        features_name_path = path + "/" + "features_name" + ".npy"
        with open(features_name_path, 'rb') as file:
            self.features_name = pickle.load(file)

        self.vocab_sizes = [np.load(path + "/" + features + "_info.npy",
                                    allow_pickle=True)
                            for features in self.features_name]

        print(f"feature name:{self.features_name}")

        train_data = []
        val_data = []
        test_data = []

        for type in ["train", "val", "test"]:

            node_feature = {}
            for name in self.features_name:
                feature_path = os.path.join(path, name + '_' + str(fold) + f"_{type}.npy")
                att_list = np.load(feature_path, allow_pickle=True)
                node_feature[name] = att_list

            label_path = os.path.join(path, 'label' + '_' + str(fold) + f"_{type}.npy")
            labels = np.load(label_path, allow_pickle=True)

            for i in range(len(labels)):
                data_dict = {}
                for name in self.features_name:
                    data_dict[name] = node_feature[name][i]

                data_dict['label'] = labels[i]
                if type == 'train':
                    train_data.append(data_dict)
                elif type == 'val':
                    val_data.append(data_dict)
                elif type == 'test':
                    test_data.append(data_dict)

        self.device = self.get_device(0)

        rl_model = DQN.load(f"./RL_model/{eventlog}/DQN_best_model_{fold}", device=self.device)

        train_graphs, train_labels, train_actions = self.generate_optimal_graphs(train_data, self.features_name,
                                                                                 rl_model, self.device)
        val_graphs, val_labels, val_actions = self.generate_optimal_graphs(val_data, self.features_name, rl_model,
                                                                           self.device)
        test_graphs, test_labels, test_actions = self.generate_optimal_graphs(test_data, self.features_name, rl_model,
                                                                              self.device)

        def print_action_distribution(name, action_counts):
            total = sum(action_counts)
            print(f"\n{name} Action Distribution:")
            for action, count in enumerate(action_counts):
                percentage = count / total * 100 if total > 0 else 0
                print(f"  Action {action}: {percentage:.1f}% ({count}/{total})")

        print_action_distribution("Train", train_actions)
        print_action_distribution("Validation", val_actions)
        print_action_distribution("Test", test_actions)

        dgl.save_graphs(f"./graph_data/{eventlog}_{fold}/train_graphs",
                        train_graphs,
                        {"label": torch.tensor(train_labels)})
        dgl.save_graphs(f"./graph_data/{eventlog}_{fold}/val_graphs",
                        val_graphs,
                        {"label": torch.tensor(val_labels)})
        dgl.save_graphs(f"./graph_data/{eventlog}_{fold}/test_graphs",
                        test_graphs,
                        {"label": torch.tensor(test_labels)})
