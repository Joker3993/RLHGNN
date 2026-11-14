"""
Random Graph-Structure Selection Control（随机图结构选择对照实验）
====================================================================

实验目的
--------
RLHGNN 用 PPO 策略为每个 trace prefix 动态选择最合适的图结构。
本脚本提供对应的对照（control）baseline：**用均匀随机选择替代 RL 策略**，
即对每个 trace prefix 从 4 种候选图结构（action 0-3）中等概率随机选一种。

除"选择机制"外，后续流程与 RLHGNN 完全一致：
    build_graph() 构建被选中的图
      -> dgl.save_graphs() 保存混合图数据集（与 final_policy.py 产物同格式）
      -> 用与 final_main.py 完全相同的配置训练 final HGNN
      -> 测试集评估（指标计算与 metrics_final.py 一致）

对照意义
--------
若随机选择也能达到接近 RLHGNN 的效果，说明"选哪个图结构"本身影响有限；
反之，则说明 RL 的动态选择机制带来了真实收益。

图结构定义（build_graph.py，与 RLHGNN 相同）
    0 : forward             前向边
    1 : Bidirect            前向 + 后向
    2 : forward_complex     前向 + 重复活动跳连
    3 : Bidirect_complex    双向 + 重复活动跳连

输出（与 RLHGNN 主实验路径隔离，不会覆盖其结果）
    graph_data/<dataset>_0_seed<seed>_random/{train,val,test}_graphs
    graph_data/<dataset>_0_seed<seed>_random/actions_{train,val,test}.npy
    final_train_random/<dataset>/seed<seed>/<dataset>_fold0_seed<seed>_model.pkl
    result_final_train_random/<dataset>/seed<seed>/<dataset>_seed<seed>.txt
    result_final_train_random/<dataset>/<dataset>_random_summary.txt   （5 seed 的 mean±std）

前置依赖
    只需先完成 Step 1 数据预处理（python data_process.py），因为要读取
    raw_dir/<dataset>_0/ 下的 prefix 样本，以及 HeteroSAGE 所需的
    特征词表（features_name.npy / *_info.npy / mean_trace.npy / 类别配置）。
    本脚本是自包含的：不需要 Pretrain/ 与 RL_model/，也不需要先跑 final_policy.py。

用法
    python random_control_main.py                                  # 全部数据集 × 5 seeds
    python random_control_main.py --datasets p2p --seeds 133        # 单数据集单 seed 快速验证
    python random_control_main.py --skip-existing                   # 断点续跑
"""

import argparse
import copy
import os
import pickle
import random
import time
import warnings

import dgl
import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from MyDataset_final import MyDataset
from build_graph import build_graph
from model.model import HeteroSAGE

warnings.filterwarnings("ignore", category=UserWarning)

# ====================== 实验配置 ======================
NUM_ACTIONS = 4                     # 候选图结构数量
GRAPH_SUFFIX = "random"             # 图数据集后缀，用于与 RL 版本隔离开
MODEL_ROOT = "final_train_random"
RESULT_ROOT = "result_final_train_random"
GRAPH_ROOT = "graph_data"

SEED_LIST = [133, 188, 456, 789, 1666]
DATASET_LIST = [
    'bpi13_closed_problems',
    'bpi13_problems',
    'OTC',
    'bpi13_incidents',
    'bpi12w_complete',
    'p2p',
    'bpi12_all_complete',
    'BPI2020_Prepaid',
]


def get_device(gpu):
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====================== 第一部分：随机选择图结构并构建数据集 ======================

def load_prefix_samples(dataset, fold):
    """
    读取 raw_dir/<dataset>_<fold>/ 下的 prefix 样本（与 final_policy.py 的读取方式一致）。

    返回
        features_name : list[str]         特征名列表
        splits        : dict[str, list]   {'train'|'val'|'test': [sample_dict, ...]}
    """
    path = f"./raw_dir/{dataset}_{fold}"

    with open(os.path.join(path, "features_name.npy"), 'rb') as file:
        features_name = pickle.load(file)

    splits = {}
    for split in ["train", "val", "test"]:
        node_feature = {}
        for name in features_name:
            feature_path = os.path.join(path, name + '_' + str(fold) + f"_{split}.npy")
            node_feature[name] = np.load(feature_path, allow_pickle=True)

        label_path = os.path.join(path, 'label' + '_' + str(fold) + f"_{split}.npy")
        labels = np.load(label_path, allow_pickle=True)

        samples = []
        for i in range(len(labels)):
            data_dict = {name: node_feature[name][i] for name in features_name}
            data_dict['label'] = labels[i]
            samples.append(data_dict)

        splits[split] = samples

    return features_name, splits


def build_random_graphs(samples, features_name, rng):
    """
    对每条 trace prefix 均匀随机选择一个图结构（action），并构建对应图。

    rng 使用 random.Random 实例（非全局），因此图结构的选择
    不会消耗训练阶段全局 random 状态的随机数，且完全可复现。
    """
    graphs, labels, actions = [], [], []

    for sample in tqdm(samples, desc="  building random graphs", leave=False):
        # ★ 核心：每个 prefix 等概率随机选一种候选图结构
        action = rng.randrange(NUM_ACTIONS)

        current_prefix = sample['activity']
        graphs.append(build_graph(current_prefix, action, sample, features_name))
        labels.append(sample['label'])
        actions.append(action)

    return graphs, labels, actions


def generate_random_graph_dataset(dataset, fold, seed):
    """
    生成"随机选择"版本的混合图数据集，并保存到
    graph_data/<dataset>_<fold>_seed<seed>_random/。

    为每个 split 使用独立的 RNG（由 (dataset, seed, split) 派生），
    保证各 split 的随机分配互不影响且可复现。
    """
    graph_root = f"./{GRAPH_ROOT}/{dataset}_{fold}_seed{seed}_{GRAPH_SUFFIX}"
    os.makedirs(graph_root, exist_ok=True)

    features_name, splits = load_prefix_samples(dataset, fold)

    print(f"  随机图结构选择：{dataset} (seed={seed})")
    print(f"  feature name: {features_name}")

    for split in ["train", "val", "test"]:
        # 用字符串做种子：random.seed 对 str 走 sha512 派生（与 PYTHONHASHSEED 无关），
        # 因此跨进程 / 跨运行稳定可复现；且不同 split 的分配互不相关。
        rng = random.Random(f"{dataset}_{fold}_{seed}_{split}")

        graphs, labels, actions = build_random_graphs(splits[split], features_name, rng)
        actions_arr = np.asarray(actions, dtype=np.int64)

        dgl.save_graphs(f"{graph_root}/{split}_graphs",
                        graphs,
                        {"label": torch.tensor(labels)})
        # 额外保存选择记录，便于分析随机分布
        np.save(f"{graph_root}/actions_{split}.npy", actions_arr)

        total = len(actions_arr)
        print(f"    [{split}] n={total}, action distribution:")
        for a in range(NUM_ACTIONS):
            count = int(np.sum(actions_arr == a))
            percentage = count / total * 100 if total > 0 else 0.0
            print(f"      Action {a}: {percentage:5.1f}% ({count}/{total})")

    print(f"  [OK] random graph dataset saved: {graph_root}")
    return graph_root


# ====================== 第二部分：训练 final HGNN（与 final_main.py 同配置） ======================
# 说明：以下 train / validate 与 final_main.py 逐行一致（仅在 precision_recall_fscore_support
#       中显式写了 zero_division=0，与 sklearn 的默认行为等价，只为消除告警，不影响数值），
#       以确保对照实验与 RLHGNN 主实验的唯一差异只有"图结构的选择机制"。

def train(model, train_loader, loss_func, optimizer, device, data_length, scaler=None, clip_grad_norm=2.0):
    model.train()

    total_loss = 0

    total_accuracy = 0
    total_step = 0

    for batch in train_loader:
        batch = [item.to(device) for item in batch]

        Positive_graph, labels = batch

        if scaler is not None:
            with autocast():
                logits = model(Positive_graph)
                logits = logits.to(device)
                loss = loss_func(logits, labels)
        else:
            logits = model(Positive_graph)
            logits = logits.to(device)
            loss = loss_func(logits, labels)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        total_accuracy += (logits.argmax(1) == labels).sum().item()
        total_step += 1

    return total_loss / total_step, total_accuracy / data_length


def validate(model, loss_func, val_loader, device, data_length):
    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_test_step = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in val_loader:
            batch = [item.to(device) for item in batch]

            Positive_graph, labels = batch

            if device.type == 'cuda':
                with autocast():
                    logits = model(Positive_graph)
                    logits = logits.to(device)
                    loss = loss_func(logits, labels)
            else:
                logits = model(Positive_graph)
                logits = logits.to(device)
                loss = loss_func(logits, labels)

            total_loss += loss.item()
            total_accuracy += (logits.argmax(1) == labels).sum().item()
            total_test_step += 1

            all_labels.append(labels)
            all_preds.append(logits.argmax(1))

    y_true = torch.cat(all_labels, 0).to('cpu')
    y_pred = torch.cat(all_preds, 0).to('cpu')
    _, _, val_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', pos_label=None,
                                                      zero_division=0)

    return total_loss / total_test_step, total_accuracy / data_length, val_f1


def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)


def evaluate_test(model, test_loader, device, data_length):
    """
    测试集评估：指标计算方式与 metrics_final.py 完全一致
    （不启用 autocast，避免 fp16 推理在类别边界上带来与论文指标脚本的差异），
    以便随机对照实验的结果能与 RLHGNN 的最终指标直接对比。
    """
    model.eval()

    total_loss = 0
    total_accuracy = 0
    all_labels = []
    all_preds = []

    loss_func = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for batch in test_loader:
            batch = [item.to(device) for item in batch]

            Positive_graph, labels = batch

            logits = model(Positive_graph)
            loss = loss_func(logits, labels)

            total_loss += loss.item()
            total_accuracy += (logits.argmax(1) == labels).sum().item()

            all_labels.append(labels)
            all_preds.append(logits.argmax(1))

    Y_test_int = torch.cat(all_labels, 0).to('cpu')
    preds_a = torch.cat(all_preds, 0).to('cpu')

    accuracy = total_accuracy / data_length
    _, _, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                      pos_label=None, zero_division=0)

    auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
    prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")
    g_mean = geometric_mean_score(Y_test_int, preds_a, average="macro")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(fscore),
        "auc_macro": float(auc_score_macro),
        "prauc_macro": float(prauc_score_macro),
        "g_mean": float(g_mean),
        "test_loss": float(total_loss / max(len(test_loader), 1)),
    }


def train_final_model(args, dataset, seed, fold=0):
    """在随机选择的图数据集上训练 final HGNN，并在测试集上评估。"""
    device = get_device(args.gpu)

    dataset_name = f"{dataset}_{fold}_seed{seed}_{GRAPH_SUFFIX}"

    print(f"\n{'=' * 70}")
    print(f"[Random Control] dataset={dataset}  seed={seed}  graphs={dataset_name}")
    print(f"{'=' * 70}")
    print(f"hidden_dim: {args.hidden_dim}, num_epochs: {args.num_epochs}, lr: {args.lr}, "
          f"batch_size: {args.batch_size}, dropout: {args.dropout}, num_layers: {args.num_layers}")
    print(f"label_smoothing: {args.label_smoothing}, weight_decay: {args.weight_decay}, "
          f"clip_grad_norm: {args.clip_grad_norm}, patience: {args.early_stop_patience}")

    dataset_train = MyDataset(name=dataset_name, type="train")
    dataset_val = MyDataset(name=dataset_name, type="val")
    dataset_test = MyDataset(name=dataset_name, type="test")

    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HeteroSAGE(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        dataname=dataset,
        fold=fold,
        num_layers=args.num_layers
    )

    num_workers = 0 if os.name == 'nt' else min(args.num_workers, os.cpu_count() or 1)
    train_loader = GraphDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=num_workers)
    val_loader = GraphDataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_workers)

    model.to(device)

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.NAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print("AMP (Automatic Mixed Precision) is enabled.")

    # 学习率调度：预热 + 余弦退火（与 final_main.py 一致）
    # 保护：若 num_epochs<=5 而 warmup 仍取 5，CosineAnnealingLR 的 T_max 会 <=0。
    # 默认配置 (num_epochs=50) 下 warmup_epochs 仍为 5，与 final_main.py 完全一致。
    warmup_epochs = min(5, max(1, args.num_epochs - 1))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=args.num_epochs - warmup_epochs, eta_min=1e-6)
        ],
        milestones=[warmup_epochs]
    )

    patience = args.early_stop_patience

    no_improvement_count = 0
    best_epoch = 0
    best_state = None
    best_val_score = float('-inf')

    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = train(
            model,
            train_loader,
            loss_func,
            optimizer,
            device,
            len(dataset_train),
            scaler,
            clip_grad_norm=args.clip_grad_norm,
        )
        val_loss, val_accuracy, val_f1 = validate(model, loss_func, val_loader, device, len(dataset_val))
        val_score = val_accuracy
        current_lr = scheduler.get_last_lr()[0]
        print(
            f'Epoch [{epoch + 1}/{args.num_epochs}], '
            f'Train Loss: {train_loss:.3f},'
            f'Train Acc: {train_accuracy:.3f},'
            f' Val Loss: {val_loss:.3f},'
            f' Val Acc: {val_accuracy:.3f},'
            f' Val F1: {val_f1:.3f},'
            f' Val Score: {val_score:.3f},'
            f' LR: {current_lr:.6f}')

        if round(val_score, 4) >= round(best_val_score, 4):
            best_val_score = val_score
            best_epoch = epoch + 1
            no_improvement_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping: Validation score has not improved for {} epochs.".format(patience))
                break

        scheduler.step()

    # ---- 保存最优模型 ----
    save_dir = os.path.join(MODEL_ROOT, dataset, f"seed{seed}")
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'{dataset}_fold{fold}_seed{seed}_model.pkl')

    model.load_state_dict(best_state)
    torch.save(model, model_path)

    check_model = torch.load(model_path, weights_only=False)
    check_model.to(device)
    val_loss, val_accuracy, val_f1 = validate(check_model, loss_func, val_loader, device, len(dataset_val))
    print('-' * 89)
    print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}]. In best model: '
          f'Validation Loss: {val_loss:.3f}, Val Acc: {val_accuracy:.3f}, Val F1: {val_f1:.3f}')

    # ---- 测试集评估 ----
    metrics = evaluate_test(check_model, test_loader, device, len(dataset_test))
    print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}]. Test -> '
          f"loss={metrics['test_loss']:.3f} accuracy={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} "
          f"AUC={metrics['auc_macro']:.4f} PRAUC={metrics['prauc_macro']:.4f} "
          f"Gmean={metrics['g_mean']:.4f}")
    print('-' * 89)

    metrics['best_epoch'] = best_epoch
    metrics['model_path'] = model_path
    return metrics


def write_seed_result(dataset, seed, metrics):
    result_dir = os.path.join(RESULT_ROOT, dataset, f"seed{seed}")
    os.makedirs(result_dir, exist_ok=True)

    result_summary = (
        f"Result -> test_loss={metrics['test_loss']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"AUC={metrics['auc_macro']:.4f} PRAUC={metrics['prauc_macro']:.4f} "
        f"Gmean={metrics['g_mean']:.4f}"
    )
    with open(os.path.join(result_dir, f"{dataset}_seed{seed}.txt"), 'a') as outfile:
        outfile.write(f"# Random Graph-Structure Selection Control (best_epoch={metrics['best_epoch']})\n")
        outfile.write(result_summary)
        outfile.write('\n\n')


def write_dataset_summary(dataset, seeds, metrics_list):
    os.makedirs(os.path.join(RESULT_ROOT, dataset), exist_ok=True)
    summary_path = os.path.join(RESULT_ROOT, dataset, f"{dataset}_random_summary.txt")

    accuracies = [m['accuracy'] for m in metrics_list]
    macro_f1s = [m['macro_f1'] for m in metrics_list]
    ddof = 1 if len(accuracies) > 1 else 0  # 单 seed 时 ddof=1 会得到 nan

    with open(summary_path, 'w') as summary_file:
        summary_file.write("Random Graph-Structure Selection Control (uniform random over 4 structures)\n")
        summary_file.write(f"Dataset: {dataset}\n")
        summary_file.write(f"Seeds: {seeds}\n\n")
        for seed, m in zip(seeds, metrics_list):
            summary_file.write(
                f"seed={seed} accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
                f"AUC={m['auc_macro']:.4f} PRAUC={m['prauc_macro']:.4f} "
                f"Gmean={m['g_mean']:.4f} test_loss={m['test_loss']:.4f} "
                f"best_epoch={m['best_epoch']}\n")

        summary_file.write(f"\naccuracy_mean={float(np.mean(accuracies)):.4f}\n")
        summary_file.write(f"accuracy_std={float(np.std(accuracies, ddof=ddof)):.4f}\n")
        summary_file.write(f"macro_f1_mean={float(np.mean(macro_f1s)):.4f}\n")
        summary_file.write(f"macro_f1_std={float(np.std(macro_f1s, ddof=ddof)):.4f}\n")

    print(f"\n[Summary] {dataset}: "
          f"accuracy={np.mean(accuracies):.4f}±{np.std(accuracies, ddof=ddof):.4f}, "
          f"macro_f1={np.mean(macro_f1s):.4f}±{np.std(macro_f1s, ddof=ddof):.4f}")
    print(f"[Summary] saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Random Graph-Structure Selection Control')

    parser.add_argument("--datasets", type=str, nargs='+', default=DATASET_LIST,
                        help="datasets to run")
    parser.add_argument("--seeds", type=int, nargs='+', default=SEED_LIST,
                        help="random seeds")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip (dataset, seed) whose final model already exists")
    parser.add_argument("--rebuild-graphs", action="store_true",
                        help="rebuild random graphs even if they already exist")

    parser.add_argument("--hidden-dim", type=int, default=256, help="dim of hidden")
    parser.add_argument("--num-epochs", type=int, default=50, help="number of epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--num_layers", type=int, default=3, help="num_layers")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="label smoothing")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--clip-grad-norm", type=float, default=2, help="gradient clipping norm")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    args = parser.parse_args()

    fold = 0

    for dataset in args.datasets:
        print(f"\n{'#' * 70}")
        print(f"# Random Control — dataset: {dataset}")
        print(f"{'#' * 70}")

        metrics_list = []
        seeds_done = []

        for seed in args.seeds:
            set_seed(seed)

            model_path = os.path.join(MODEL_ROOT, dataset, f"seed{seed}",
                                      f'{dataset}_fold{fold}_seed{seed}_model.pkl')
            if args.skip_existing and os.path.exists(model_path):
                print(f"[Skip] existing model for {dataset} seed={seed}")
                continue

            start_total = time.perf_counter()

            # ---- 1) 随机选择图结构并构建混合图数据集 ----
            graph_root = f"./{GRAPH_ROOT}/{dataset}_{fold}_seed{seed}_{GRAPH_SUFFIX}"
            graphs_ready = all(
                os.path.exists(f"{graph_root}/{split}_graphs")
                for split in ["train", "val", "test"]
            )
            if args.rebuild_graphs or not graphs_ready:
                generate_random_graph_dataset(dataset, fold, seed)
            else:
                print(f"[Skip] existing random graph dataset: {graph_root}")

            # ---- 2) 训练 final HGNN 并评估 ----
            metrics = train_final_model(args, dataset, seed, fold=fold)

            write_seed_result(dataset, seed, metrics)
            metrics_list.append(metrics)
            seeds_done.append(seed)

            elapsed_hours = (time.perf_counter() - start_total) / 3600
            print(f"[Time] {dataset} seed={seed}: {elapsed_hours:.3f} hours")

        if metrics_list:
            write_dataset_summary(dataset, seeds_done, metrics_list)
        else:
            print(f"[Warn] no run recorded for {dataset} (all seeds skipped by --skip-existing?)")

    print("\n[Done] Random Graph-Structure Selection Control finished.")


if __name__ == '__main__':
    main()
