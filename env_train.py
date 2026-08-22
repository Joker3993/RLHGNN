import contextlib
import math
import os
import pickle
import random
import time

import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
from sklearn.metrics import f1_score

from stable_baselines3 import PPO  # 改这里

from torch import nn, optim

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

    with torch.inference_mode():
        if P_graph.device != device:
            P_graph = P_graph.to(device)
        label = torch.tensor([attr_label['label']], device=device, dtype=torch.long)

        logits = gnn_model(P_graph)
        pred = logits.argmax(1).item()
        loss = loss_func(logits, label)
        correct = (logits.argmax(1) == label).sum().item()
    return loss, correct, pred


def calculate_reward(prefix, action, attr_label):
    P_graph = attr_label['graph_cache'][action]

    loss, correct, pred = evaluate_gnn(P_graph, attr_label, action)

    return loss, correct, pred


class PrefixGraphEnv(gym.Env):
    def __init__(self, data_prefix, batch_size=32):
        super(PrefixGraphEnv, self).__init__()
        self.data = data_prefix
        self.current_prefix = None
        self.current_label = None
        self.current_prefix_all_attr = None
        self.current_index = 0
        self.step_count = 0
        self.baseline_loss = None
        self.alpha = 0.1
        self.action_performance = {a: {'count': 0, 'correct': 0} for a in range(4)}

        # 新增batch相关配置
        self.batch_size = batch_size
        self.batch_buffer = []

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

        loss, correct, pred = calculate_reward(self.current_prefix, action, self.current_prefix_all_attr)
        current_loss = loss.item()

        # 将当前样本结果加入batch缓存
        self.batch_buffer.append((current_loss, pred, self.current_label))

        # 记录当前动作性能
        self.action_performance[action]['count'] += 1
        if correct:
            self.action_performance[action]['correct'] += 1

        beta = beta_value

        # 缓存未满，检查是否已经遍历完所有训练样本
        if len(self.batch_buffer) < self.batch_size:
            # 如果已经遍历完所有样本，即使不满也计算奖励
            if self.current_index >= len(self.data):
                # 直接用现有buffer计算奖励，结束episode
                batch_losses = [item[0] for item in self.batch_buffer]
                batch_preds = [item[1] for item in self.batch_buffer]
                batch_labels = [item[2] for item in self.batch_buffer]

                avg_batch_loss = np.mean(batch_losses)
                macro_f1 = f1_score(batch_labels, batch_preds, average='macro')

                # 更新基线损失
                if self.baseline_loss is None:
                    self.baseline_loss = avg_batch_loss
                else:
                    self.baseline_loss = (1 - self.alpha) * self.baseline_loss + self.alpha * avg_batch_loss

                # 计算奖励
                improvement = max(0.0, self.baseline_loss - avg_batch_loss)
                improvement_score = np.clip(improvement / (self.baseline_loss + 1e-8), 0.0, 1.0)

                loss_score = np.exp(-avg_batch_loss)
                correct_score = macro_f1

                reward = loss_score + correct_score + beta * improvement_score

                # 清空缓存，重置索引开始下一轮
                self.batch_buffer = []
                self.current_index = 0
                random.shuffle(self.data)

                terminated = True
                truncated = False
                next_state = extract_features(self.current_prefix_all_attr)
                info = {}
                return next_state, reward, terminated, truncated, info

            # 没遍历完，继续收集下一个样本
            node_feature_dict = self.data[self.current_index]
            self.current_index += 1
            self.current_prefix = node_feature_dict['activity']
            self.current_label = node_feature_dict['label']
            self.current_prefix_all_attr = node_feature_dict
            next_state = extract_features(node_feature_dict)
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            return next_state, reward, terminated, truncated, info

        # batch满了，计算批量指标
        batch_losses = [item[0] for item in self.batch_buffer]
        batch_preds = [item[1] for item in self.batch_buffer]
        batch_labels = [item[2] for item in self.batch_buffer]

        avg_batch_loss = np.mean(batch_losses)
        macro_f1 = f1_score(batch_labels, batch_preds, average='macro')

        # 更新基线损失
        if self.baseline_loss is None:
            self.baseline_loss = avg_batch_loss
        else:
            self.baseline_loss = (1 - self.alpha) * self.baseline_loss + self.alpha * avg_batch_loss

        # 计算奖励
        improvement = max(0.0, self.baseline_loss - avg_batch_loss)
        improvement_score = np.clip(improvement / (self.baseline_loss + 1e-8), 0.0, 1.0)

        loss_score = np.exp(-avg_batch_loss)  # 使用批量平均损失计算损失分数
        correct_score = macro_f1  # 使用宏F1作为分类性能奖励

        reward = loss_score + correct_score + beta * improvement_score

        # 清空缓存，准备下一个batch
        self.batch_buffer = []

        terminated = True
        truncated = False
        next_state = extract_features(self.current_prefix_all_attr)
        info = {}

        return next_state, reward, terminated, truncated, info


def evaluate(model, data):
    total_loss = 0.0
    total_correct = 0
    action_counts = [0] * 4

    with torch.inference_mode():
        for sample in data:
            obs = extract_features(sample)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1

            attr_label = sample
            P_graph = attr_label['graph_cache'][action]
            loss, correct, pred = evaluate_gnn(P_graph, attr_label, action)

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_list = [133, 188, 456, 789, 1666]
    list_eventlog = [
        'bpi13_closed_problems',
        'bpi13_problems',
        'bpi13_incidents',
        'p2p',
        'BPI2020_Prepaid',
        'bpi12w_complete',
        'bpi12_all_complete',
    ]
    beta_value = 1
    print(f"beta:{beta_value}")

    for eventlog in list_eventlog:
        for seed in seed_list:

            print(f"--------------开始-记录时间------------")
            start_total = time.perf_counter()

            print(f"-------------{eventlog}日志开始---------------")
            print(f"seed: {seed}")

            set_seed(seed)
            fold = 0

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

            # ========== 预构建图缓存 ==========
            print("预构建图缓存（约4x加速）...")
            for dataset_name, dataset in [("train", train_data), ("val", val_data), ("test", test_data)]:
                for data_dict in dataset:
                    graph_cache = {}
                    for action in range(4):
                        g = build_graph(data_dict['activity'], action, data_dict, features_name)
                        graph_cache[action] = g.to(device)
                    data_dict['graph_cache'] = graph_cache
                print(f"  {dataset_name}: {len(dataset)} 个样本图已缓存")

            model_list = []
            for action in range(4):
                model_path_1 = f"./Pretrain/action_{action}_{seed}/" + eventlog
                model_path = model_path_1 + '/' + str(eventlog) + f'_fold{fold}' + '_model.pkl'

                GNN_model = torch.load(model_path, map_location=device)
                GNN_model.to(device)
                model_list.append(GNN_model)

            loss_func = nn.CrossEntropyLoss()
            train_env = PrefixGraphEnv(train_data)

            policy_kwargs = dict(
                net_arch=[256, 128, 64],
                activation_fn=nn.ReLU,
                normalize_images=False,
                optimizer_class=optim.Adam,
            )

            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=lambda f: 1e-3 * 0.5 * (1 + math.cos(f * math.pi)),  # 余弦退火衰减
                n_steps=1024,
                batch_size=128,
                n_epochs=3,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                verbose=1,
                device=device,
                policy_kwargs=policy_kwargs,
            )

            best_val_accuracy = 0
            patience = 5
            no_improve_epochs = 0
            best_epoch = 0

            iterations = 50
            train_size = len(train_data)
            print(f"训练集大小: {train_size}")
            if train_size < 3000:
                initial_steps = 8000
                final_steps = 4000
            elif train_size < 20000:
                initial_steps = 30000
                final_steps = 15000
            else:
                initial_steps = 50000
                final_steps = 30000

            model_dir = f"./RL_model/{eventlog}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/PPO_best_model_fold{fold}_seed{seed}"
            log_path = f"{model_dir}/log_seed{seed}.txt"

            print("-----------------开始训练模型------------------")
            for epoch in range(iterations):
                if epoch == 0:
                    steps = initial_steps
                else:
                    steps = final_steps

                model.learn(total_timesteps=steps, reset_num_timesteps=False, log_interval=100)

                if epoch % 10 == 0:
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
                    model.save(model_path)
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f"Early stopping at epoch {epoch + 1}!")
                        break

            print(f"Best epoch :{best_epoch + 1}")
            print("-----------------训练结束------------------")

            test_env = PrefixGraphEnv(test_data)
            model_test = PPO.load(model_path, env=test_env)

            print("-----------------开始测试模型------------------")

            test_loss, test_acc, test_actions = evaluate(model_test, test_data)

            with open(log_path, 'a') as log_file:
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
                            attr_label = sample
                            P_graph = attr_label['graph_cache'][action]
                            loss, correct, pred = evaluate_gnn(P_graph, attr_label, action)

                            total_loss += loss.item()
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
            final_policy(dataset=eventlog, fold=fold, model_path=model_path, output_suffix=f"seed{seed}").main()
            print("图数据集预测完成！！！")

            end_total = time.perf_counter()
            method_name(start_total, end_total, seed)
            print(f"--------------结束-记录时间------------")