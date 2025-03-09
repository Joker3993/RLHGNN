import contextlib
import os
import pickle
import random
import time

import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np

from stable_baselines3 import DQN

from stable_baselines3.common.running_mean_std import RunningMeanStd
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from build_graph import build_graph
from final_policy import final_policy
from utils import calculate_entropy, calculate_longest_consecutive_subsequence


def get_device(gpu):
    
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def extract_features(current_prefix_all_attr):
    

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


def evaluate_gnn(P_graph, attr_label, action):
    
    gnn_model = model_list[action]
    gnn_model.eval()

    with torch.no_grad():
        P_graph = P_graph.to(device)
        label = torch.tensor([attr_label['label']], device=device, dtype=torch.long)

        logits = gnn_model(P_graph)

        loss = loss_func(logits, label)
        correct = (logits.argmax(1) == label).sum().item()
    return loss, correct


def calculate_reward(prefix, action, attr_label):
    

    P_graph = build_graph(prefix=prefix, action=action, attr_label=attr_label, features_name=features_name)

    loss, correct = evaluate_gnn(P_graph, attr_label, action)

    return loss, correct


class PrefixGraphEnv(gym.Env):
    def __init__(self, data_prefix):
        super(PrefixGraphEnv, self).__init__()
        self.data = data_prefix
        self.current_prefix = None
        self.current_label = None
        self.current_prefix_all_attr = None
        self.current_index = 0
        self.step_count = 0
        self.baseline_loss = None
        self.alpha = 0.3
        self.action_performance = {a: {'count': 0, 'correct': 0} for a in range(4)}

        self.reward_stats = RunningMeanStd(shape=())

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=0, high=activity_info, shape=(mean_len + 7 + 6,), dtype=np.float32)

    def reset(self, **kwargs):
        

        if self.current_index >= len(self.data):
            self.current_index = 0
            random.shuffle(self.data)

        node_feature_dict = self.data[self.current_index]
        self.current_index += 1

        self.current_prefix = node_feature_dict['activity']
        self.current_label = node_feature_dict['label']
        self.current_prefix_all_attr = node_feature_dict

        obs = extract_features(node_feature_dict)
        info = {}
        return obs, info

    def step(self, action):
        

        loss, correct = calculate_reward(self.current_prefix, action, self.current_prefix_all_attr)
        current_loss = loss.item()

        if self.baseline_loss is None:
            self.baseline_loss = current_loss
        else:
            self.baseline_loss = (1 - self.alpha) * self.baseline_loss + self.alpha * current_loss

        improvement = self.baseline_loss - current_loss

        base_reward = -current_loss + 0.1
        if correct:
            base_reward += 2.0
        else:
            base_reward -= 1.0

        improvement_reward = max(0, improvement / (self.baseline_loss + 1e-8))

        reward = base_reward + 2.0 * improvement_reward

        self.reward_stats.update(np.array([reward]))

        mean = self.reward_stats.mean
        std = np.sqrt(self.reward_stats.var)
        normalized_reward = (reward - mean) / (std + 1e-8)

        terminated = True
        truncated = False

        next_state = extract_features(self.current_prefix_all_attr)
        info = {}

        self.action_performance[action]['count'] += 1
        if correct:
            self.action_performance[action]['correct'] += 1

        return next_state, normalized_reward, terminated, truncated, info


def evaluate(model, data):
    total_loss = 0.0
    total_correct = 0
    action_counts = [0] * 4

    with torch.no_grad():
        for sample in data:
            obs = extract_features(sample)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1

            current_prefix = sample['activity']
            attr_label = sample
            P_graph = build_graph(current_prefix, action, attr_label, features_name)
            loss, correct = evaluate_gnn(P_graph, attr_label, action)

            total_loss += loss.item()
            total_correct += correct

        avg_loss = total_loss / len(data)
        accuracy = total_correct / len(data)
        action_dist = [count / len(data) for count in action_counts]

    return avg_loss, accuracy, action_dist


def method_name(start_time, end_time, fold):
    total_training_time_seconds = end_time - start_time

    total_training_time_hours = total_training_time_seconds / 3600

    time_file_path = f'train_time/{eventlog}/second_training_time_{fold}.txt'
    os.makedirs(f'train_time/{eventlog}', exist_ok=True)

    with open(time_file_path, 'w') as time_file:
        time_file.write(f"training time: {total_training_time_hours:.3f} hours\n")

    print("-" * 90)
    print("\n")

    print(f"{fold} fold---Total training time: {total_training_time_hours:.3f} hours")


if __name__ == '__main__':

    list_eventlog = [
        'bpi13_closed_problems',
        'bpi13_problems',
        'bpi13_incidents',
        'bpi12w_complete',
        'bpi12_all_complete',
        'BPI2020_Prepaid',
    ]

    for eventlog in list_eventlog:

        print(f"--------------开始-记录时间------------")

        start_total = time.perf_counter()

        print(f"-------------{eventlog}日志开始---------------")
        for fold in range(3):
            path = "./raw_dir/" + eventlog + "_" + str(fold)

            features_name_path = path + "/" + "features_name" + ".npy"
            with open(features_name_path, 'rb') as file:
                features_name = pickle.load(file)

            vocab_sizes = [np.load(path + "/" + features + "_info.npy",
                                   allow_pickle=True)
                           for features in features_name]

            
            path = f"raw_dir/{eventlog}_{fold}/"
            with open(path + "mean_trace.npy", 'rb') as f:
                mean_len = pickle.load(f)

            
            activity_info = np.load(path + "activity_info.npy", allow_pickle=True)

            print(f"feature name:{features_name}")

            train_data = []
            val_data = []
            test_data = []

            path2 = "./raw_dir/" + eventlog + "_" + str(fold) + "/part2"

            for type in ["train", "val", "test"]:

                node_feature = {}
                for name in features_name:
                    feature_path = os.path.join(path2, name + '_' + str(fold) + f"_{type}.npy")
                    att_list = np.load(feature_path, allow_pickle=True)
                    node_feature[name] = att_list

                label_path = os.path.join(path2, 'label' + '_' + str(fold) + f"_{type}.npy")
                labels = np.load(label_path, allow_pickle=True)
                print(f"{type} 样本数：{len(labels)}")

                for i in range(len(labels)):
                    data_dict = {}
                    for name in features_name:
                        data_dict[name] = node_feature[name][i]

                    data_dict['label'] = labels[i]
                    if type == 'train':
                        train_data.append(data_dict)
                    elif type == 'val':
                        val_data.append(data_dict)
                    elif type == 'test':
                        test_data.append(data_dict)

            device = get_device(0)
            model_list = []
            for action in range(4):
                model_path_1 = f"./Pretrain/action_{action}/" + eventlog
                model_path = model_path_1 + '/' + str(eventlog) + f'_fold{fold}' + '_model.pkl'

                GNN_model = torch.load(model_path, map_location=device)
                GNN_model.to(device)
                model_list.append(GNN_model)

            loss_func = nn.CrossEntropyLoss()
            train_env = PrefixGraphEnv(train_data)

            policy_kwargs = dict(
                net_arch=[256, 128, 128],
                activation_fn=nn.ReLU,
                normalize_images=False,
                optimizer_class=optim.NAdam,
            )

            model = DQN(
                "MlpPolicy",
                train_env,
                learning_rate=0.0001,
                buffer_size=50000,
                learning_starts=10000,
                batch_size=64,
                gamma=0.99,
                exploration_fraction=0.7,
                exploration_final_eps=0.1,
                target_update_interval=2000,
                verbose=1,
                device=device,
                policy_kwargs=policy_kwargs,
            )

            best_val_accuracy = 0
            patience = 10
            no_improve_epochs = 0
            best_epoch = 0

            iterations = 50
            initial_steps = 50000
            final_steps = 5000

            print("-----------------开始训练模型------------------")
            for epoch in range(iterations):
                if epoch == 0:
                    steps = initial_steps
                else:
                    steps = final_steps
                model.learn(total_timesteps=steps, reset_num_timesteps=False, log_interval=10000)

                if epoch % 5 == 0:
                    print("Train: ")
                    print(f"Action Performance:")
                    for a in range(4):
                        count = train_env.action_performance[a]['count']
                        correct = train_env.action_performance[a]['correct']
                        acc = correct / (count + 1e-8)
                        print(f"Action {a}: count={count} acc={acc:.2f}")
                    train_env.action_performance = {a: {'count': 0, 'correct': 0} for a in range(4)}

                val_loss, val_acc, val_actions = evaluate(model, val_data)
                print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
                print("Action Distribution:")
                for action, prob in enumerate(val_actions):
                    print(f"  Action {action}: {prob * 100:.1f}%")

                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    no_improve_epochs = 0
                    best_epoch = epoch
                    model.save(f"./RL_model/{eventlog}/DQN_best_model_{fold}")

                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f"Early stopping at epoch {epoch + 1}!")
                        break

            print(f"Best epoch :{best_epoch + 1}")
            print("-----------------训练结束------------------")

            test_env = PrefixGraphEnv(test_data)

            model_test = DQN.load(f"./RL_model/{eventlog}/DQN_best_model_{fold}", env=test_env)

            print("-----------------开始测试模型------------------")

            test_loss, test_acc, test_actions = evaluate(model_test, test_data)

            with open(f'./RL_model/{eventlog}/log.txt', 'a') as log_file:

                with contextlib.redirect_stdout(log_file):
                    print("\n" + "=" * 50)
                    print(f"{'Test Results':^50}")
                    print("=" * 50)
                    print(f"Average Loss: {test_loss:.4f}")
                    print(f"Accuracy:    {test_acc:.4f}")
                    print("Action Distribution:")
                    for action, prob in enumerate(test_actions):
                        print(f"  Action {action}: {prob * 100:.1f}%")

                    print("\n" + "=" * 50)
                    print(f"{'Action-wise Performance (Test Set)':^50}")
                    print("=" * 50)
                    action_performance = []

                    for action in range(4):
                        total_loss = 0.0
                        total_correct = 0

                        for sample in test_data:
                            current_prefix = sample['activity']
                            attr_label = sample
                            P_graph = build_graph(current_prefix, action, attr_label, features_name)
                            loss, correct = evaluate_gnn(P_graph, attr_label, action)

                            total_loss += loss
                            total_correct += correct

                        avg_loss = total_loss / len(test_data)
                        accuracy = total_correct / len(test_data)
                        action_performance.append((avg_loss, accuracy))

                    headers = ["Action", "Avg Loss", "Accuracy", "RL Selection%"]
                    print(f"\n{' | '.join(headers):^50}")
                    print("-" * 50)
                    for action, (loss, acc) in enumerate(action_performance):
                        print(f"{action:^6} | {loss:^8.4f} | {acc:^8.4f} | {test_actions[action] * 100:^12.1f}%")

            print("-" * 50)
            print("预测最终的图数据集: \n")
            final_policy(dataset=eventlog, fold=fold).main()
            print("图数据集预测完成！！！")

        end_total = time.perf_counter()
        method_name(start_total, end_total, 123)
        print(f"--------------结束-记录时间------------")
