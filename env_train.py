import argparse
import contextlib
import math
import os
import pickle
import random
import sys
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
from reward_table import build_or_load, table_fingerprint
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


def calculate_reward(prefix, action, attr_label, table=None):
    # 快路径：reward 表已在启动时按 (样本, 动作) 算好，这里只做一次数组索引。
    # table 里存的就是 evaluate_gnn 的原返回值（逐位相同），见 reward_table.py。
    if table is not None:
        return table.get(attr_label['_tid'], action)

    P_graph = attr_label['graph_cache'][action]

    loss, correct, pred = evaluate_gnn(P_graph, attr_label, action)

    return loss, correct, pred


def get_obs(attr_label):
    """取观测。预计算过就直接返回，否则走原路径。"""
    obs = attr_label.get('_obs')
    if obs is not None:
        return obs
    return extract_features(attr_label)


class PrefixGraphEnv(gym.Env):
    def __init__(self, data_prefix, batch_size=32, table=None):
        super(PrefixGraphEnv, self).__init__()
        self.data = data_prefix
        self.table = table
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

        obs = get_obs(node_feature_dict)
        info = {}
        return obs, info

    def step(self, action):

        loss, correct, pred = calculate_reward(self.current_prefix, action, self.current_prefix_all_attr, self.table)
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
                next_state = get_obs(self.current_prefix_all_attr)
                info = {}
                return next_state, reward, terminated, truncated, info

            # 没遍历完，继续收集下一个样本
            node_feature_dict = self.data[self.current_index]
            self.current_index += 1
            self.current_prefix = node_feature_dict['activity']
            self.current_label = node_feature_dict['label']
            self.current_prefix_all_attr = node_feature_dict
            next_state = get_obs(node_feature_dict)
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
        next_state = get_obs(self.current_prefix_all_attr)
        info = {}

        return next_state, reward, terminated, truncated, info


def evaluate(model, data, table=None):
    total_loss = 0.0
    total_correct = 0
    action_counts = [0] * 4

    with torch.inference_mode():
        for sample in data:
            obs = get_obs(sample)

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1

            # 与 calculate_reward 同一条路径：有表查表，没表走原始 GNN 前向
            loss, correct, pred = calculate_reward(sample['activity'], action, sample, table)

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


def verify_reward_table(tables, model_list, device, features_name, splits, k=None):
    """比对 reward 表与"现场重算"的 evaluate_gnn，给出可判定的结论。

    判定规则（两条同时满足才算 PASS）：
      1. pred 与 correct 必须**完全一致**。这两个量直接进 batch macro-F1 和
         reward，翻转一个就会改变训练轨迹，没有容差空间。
      2. max|Δloss| < 1e-6。loss 允许末位级别的差异（GPU 归约顺序所致，原实现
         自己对同一张图反复前向时也可能有同量级抖动），但不允许量级偏差。

    顺带校验（任一失败即 FAIL）：
      - 每个 split 的 _tid 恰好是 0..N-1 的一个排列（决定"查哪一行"）
      - 预计算观测与 extract_features 逐位一致
      - 表内数值有限、pred 落在合法类别范围内

    k=None 表示全量校验；给整数则每个 split 随机抽 k 个样本。
    """
    rng = random.Random(20240101)
    ok = True
    n_actions = len(model_list)
    n_classes = getattr(model_list[0], 'n_classes', None)

    print("\n" + "=" * 78)
    print("reward 表等价性校验")
    print("=" * 78)

    # ---- 0. 索引完整性 ----
    for name, dataset in splits:
        tids = [d.get('_tid') for d in dataset]
        good = tids == list(range(len(dataset)))
        print(f"  [{'PASS' if good else 'FAIL'}] _tid  {name:5s}: "
              f"{'0..N-1 完整连续' if good else '索引缺失或错位，查表行会串位'}")
        ok &= good

    # ---- 1. 观测 ----
    for name, dataset in splits:
        if dataset and '_obs' not in dataset[0]:
            print(f"  [SKIP] obs   {name:5s}: legacy 模式未做观测预计算")
            continue
        bad = sum(1 for d in dataset
                  if not np.array_equal(np.asarray(extract_features(d), dtype=np.float32), d['_obs']))
        print(f"  [{'PASS' if bad == 0 else 'FAIL'}] obs   {name:5s}: "
              f"{len(dataset) - bad}/{len(dataset)} 逐位一致")
        ok &= (bad == 0)

    # ---- 2. 表自身取值合理性 ----
    for name, _ in splits:
        t = tables[name]
        finite = bool(np.isfinite(t.loss).all())
        if n_classes is not None:
            pred_ok = bool((t.pred >= 0).all() and (t.pred < n_classes).all())
        else:
            pred_ok = True
        ok &= finite and pred_ok
        rng_txt = (f"pred∈[0,{n_classes})={pred_ok}" if n_classes is not None
                   else "pred 范围：模型未暴露 n_classes，跳过")
        print(f"  [{'PASS' if finite and pred_ok else 'FAIL'}] 取值  {name:5s}: "
              f"loss 有限={finite}  {rng_txt}")

    # ---- 2b. 表内自洽：correct 必须恰好等于 (pred == label) ----
    # 这是一条不依赖"现场重算"的独立不变式，能抓住"行列写反""_tid 整体偏移一位"
    # 这类错误——它们会让每一条 loss 都错位，但只要错位是整体的，抽查也可能
    # 看到"差得不多"的假象。这里做的是全量比对，代价 O(N×4)。
    for name, dataset in splits:
        t = tables[name]
        y = np.asarray([int(d['label']) for d in dataset], dtype=np.int64).reshape(-1, 1)
        good = bool((t.pred == y).shape == t.correct.shape and ((t.pred == y) == t.correct).all())
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] 自洽  {name:5s}: "
              f"{'correct == (pred == label) 全量成立' if good else 'correct 与 pred/label 不一致，查表行列可能错位'}")

    # ---- 3. reward 逐条比对 ----
    for name, dataset in splits:
        n = len(dataset)
        if n == 0:
            continue
        idxs = list(range(n)) if k is None else rng.sample(range(n), min(k, n))
        tab = tables[name]

        exact = pred_bad = loss_bad = 0
        max_abs = 0.0
        for cnt, i in enumerate(idxs):
            d = dataset[i]
            for a in range(n_actions):
                g = build_graph(d['activity'], a, d, features_name).to(device)
                ref_loss, ref_correct, ref_pred = evaluate_gnn(g, d, a)
                del g

                ref_loss_val = ref_loss.item()
                diff = abs(ref_loss_val - float(tab.loss[i, a]))
                max_abs = max(max_abs, diff)
                if diff == 0.0:
                    exact += 1
                elif diff >= 1e-6:
                    loss_bad += 1

                if (int(ref_pred) != int(tab.pred[i, a])
                        or int(ref_correct) != int(tab.correct[i, a])):
                    pred_bad += 1
            if (cnt + 1) % 5000 == 0:
                print(f"    ... {name} {cnt + 1}/{len(idxs)} 样本", flush=True)

        total = len(idxs) * n_actions
        good = (pred_bad == 0 and loss_bad == 0)
        print(f"  [{'PASS' if good else 'FAIL'}] reward {name:5s}: {total} 条"
              f" | 完全相等 {exact} ({exact / total:.4%})"
              f" | pred/correct 翻转 {pred_bad}"
              f" | loss 偏差>=1e-6 {loss_bad}"
              f" | max|Δloss|={max_abs:.3e}")
        ok &= good

    print("=" * 78)
    print("结论：" + ("PASS —— 表与现场前向等价，可用于正式实验" if ok
                     else "FAIL —— 存在 pred 翻转或量级偏差，请勿用表跑正式实验"))
    print("=" * 78 + "\n")
    return 0 if ok else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--legacy-reward", action="store_true",
                        help="关闭 reward 表，走原始的逐步 GNN 前向（逐位对照 / 兜底用）")
    parser.add_argument("--verify-reward-table", action="store_true",
                        help="载入数据与模型后只做等价性校验，然后退出（默认全量）")
    parser.add_argument("--verify-sample", type=int, default=None,
                        help="配合 --verify-reward-table：每个 split 只抽查 N 个样本")
    parser.add_argument("--cache-dir", default="reward_cache",
                        help="reward 表的磁盘缓存目录")
    parser.add_argument("--no-cache", action="store_true", help="不读写磁盘缓存")
    ARGS, _ = parser.parse_known_args()

    seed_list = [133, 188, 456, 789, 1666]
    list_eventlog = [
        'bpi13_closed_problems',
        'bpi13_problems',
        'OTC',
        'bpi13_incidents',
        'bpi12w_complete',
        'p2p',
        'bpi12_all_complete',
        'BPI2020_Prepaid',
    ]
    beta_value = 2
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

            model_list = []
            model_paths = []
            for action in range(4):
                model_path_1 = f"./Pretrain/action_{action}_{seed}/" + eventlog
                model_path = model_path_1 + '/' + str(eventlog) + f'_fold{fold}' + '_model.pkl'
                model_paths.append(model_path)

                # .pkl 里存的是完整模型对象（含 dgl 子模块），必须 weights_only=False，
                # 否则 PyTorch>=2.6 的默认 weights_only=True 会拒绝还原 dgl.nn.pytorch 的类。
                GNN_model = torch.load(model_path, map_location=device, weights_only=False)
                GNN_model.to(device)
                model_list.append(GNN_model)

            loss_func = nn.CrossEntropyLoss()

            splits = [("train", train_data), ("val", val_data), ("test", test_data)]

            # ========== 观测预计算 ==========
            # extract_features 只依赖样本、是纯函数，每个样本算一次即可。
            # 存成 np.float32 与 SB3 内部 np.asarray(obs, dtype=np.float32) 的结果逐位一致。
            for name, dataset in splits:
                for i, d in enumerate(dataset):
                    d['_tid'] = i
                    # legacy 模式下不预计算观测，让 --legacy-reward 成为可以对照的原始路径
                    if not ARGS.legacy_reward:
                        d['_obs'] = np.asarray(extract_features(d), dtype=np.float32)
                if not ARGS.legacy_reward:
                    print(f"  {name}: {len(dataset)} 个样本观测已预计算")

            # ========== reward 表 ==========
            tables = {}
            written = []   # 本次新写入的缓存文件（校验失败时要删掉，否则下次会静默复用坏表）
            # 注意 --legacy-reward + --verify-reward-table 同时开时**不建图缓存**：
            # 校验模式一定会建表，calculate_reward 全程走表，graph_cache 一次都不会被读到。
            # 此时若还按 legacy 建 4N 张图，等于白白多占一份显存（大数据集直接把卡撑爆）。
            if ARGS.legacy_reward and not ARGS.verify_reward_table:
                print("预构建图缓存（legacy 模式：保持原始逐步前向）...")
                for dataset_name, dataset in splits:
                    for data_dict in dataset:
                        graph_cache = {}
                        for action in range(4):
                            g = build_graph(data_dict['activity'], action, data_dict, features_name)
                            graph_cache[action] = g.to(device)
                        data_dict['graph_cache'] = graph_cache
                    print(f"  {dataset_name}: {len(dataset)} 个样本图已缓存")

            if (not ARGS.legacy_reward) or ARGS.verify_reward_table:
                print("预计算 reward 表（每个 (样本, 动作) 只算一次，逐位等价）...")
                cache_dir = None if ARGS.no_cache else ARGS.cache_dir
                for name, dataset in splits:
                    t0 = time.perf_counter()
                    # 指纹覆盖三样东西：4 个 reward 模型、loss 配置、**该 split 的数据内容**。
                    # 数据用内容哈希而不是 label 文件 mtime：data_process.py 每次重跑都会重写
                    # npy，用 mtime 会让缓存永远命不中；而只查 label 又会漏掉 activity/duration
                    # 变化（标签和样本数都没变，表却是错的，且没有任何提示）。
                    fp = table_fingerprint(model_paths, loss_func, dataset)
                    cache_path = None
                    if cache_dir:
                        cache_path = os.path.join(
                            cache_dir, f"{eventlog}_fold{fold}_seed{seed}_{name}_{fp}.npz")
                    existed = bool(cache_path) and os.path.exists(cache_path)
                    tables[name] = build_or_load(
                        data=dataset,
                        reward_fn=evaluate_gnn,
                        build_fn=lambda d, a: build_graph(d['activity'], a, d, features_name).to(device),
                        n_actions=len(model_list),
                        cache_path=cache_path,
                        fingerprint=fp,
                        verbose=True,
                    )
                    if cache_path and not existed and os.path.exists(cache_path):
                        written.append(cache_path)
                    print(f"  {name}: {len(dataset)} 样本 × {len(model_list)} 动作，"
                          f"耗时 {time.perf_counter() - t0:.1f}s")

            if ARGS.verify_reward_table:
                rc = verify_reward_table(tables, model_list, device, features_name, splits,
                                         k=ARGS.verify_sample)
                if rc != 0 and written:
                    # 校验没过的表留在磁盘上，下次训练会"命中缓存"直接用——这是最坏的
                    # 失败模式：错误的表 + 没有任何提示。直接删掉，宁可重算。
                    for p in written:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    print(f"  已删除本次新写入的 {len(written)} 个缓存文件，避免下次静默复用")
                sys.exit(rc)

            train_env = PrefixGraphEnv(train_data, table=tables.get("train"))

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
                # 一个 reward batch 内的 32 个 trace prefixes 相互独立，
                # 其在 batch 中的先后位置仅为数据排列顺序，不存在前后 decision 的时序依赖，
                # 故 gamma=gae_lambda=1.0，避免离 batch terminal reward 较远的 decision
                # 获得更弱反馈而引入与 batch 位置有关的额外衰减。
                gamma=1.0,
                gae_lambda=1.0,
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

                val_loss, val_acc, val_actions = evaluate(model, val_data, tables.get("val"))
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

            test_env = PrefixGraphEnv(test_data, table=tables.get("test"))
            model_test = PPO.load(model_path, env=test_env)

            print("-----------------开始测试模型------------------")

            test_loss, test_acc, test_actions = evaluate(model_test, test_data, tables.get("test"))

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
                            loss, correct, pred = calculate_reward(
                                sample['activity'], action, sample, tables.get("test"))

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