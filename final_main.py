import argparse
import copy
import os
import random
import time

import numpy as np
import torch

from dgl.dataloading import GraphDataLoader
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

import warnings
from torch.cuda.amp import autocast, GradScaler

from MyDataset_final import MyDataset
from model.model import HeteroSAGE

warnings.filterwarnings("ignore", category=UserWarning)


def get_device(gpu):
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


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
    _, _, val_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', pos_label=None)

    return total_loss / total_test_step, total_accuracy / data_length, val_f1


def test(model, loss_func, val_loader, device, data_length):
    model.eval()

    total_loss = 0
    total_accuracy = 0

    total_step = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in val_loader:
            batch = [item.to(device) for item in batch]

            Positive_graph, labels = batch

            if device.type == 'cuda':
                with autocast():
                    logits = model(Positive_graph)
                    loss = loss_func(logits, labels)
            else:
                logits = model(Positive_graph)
                loss = loss_func(logits, labels)

            total_loss += loss.item()
            total_accuracy += (logits.argmax(1) == labels).sum().item()

            total_step += 1

            y_pred = logits.argmax(1)
            all_labels.append(labels)
            all_preds.append(y_pred)

    average_loss = total_loss / total_step
    average_accuracy = total_accuracy / data_length

    Y_test_int = torch.cat(all_labels, 0).to('cpu')

    preds_a = torch.cat(all_preds, 0).to('cpu')

    precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                   pos_label=None)

    print(f"precision:{precision:.3f} recall:{recall:.3f} F1-score: {fscore:.3f}")

    return average_loss, average_accuracy


def train_val(args, seed):
    print("start training...")
    print("Training with the following arguments:")
    print(f"dataset: {args.dataset}")
    print(f"hidden_dim: {args.hidden_dim}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"lr: {args.lr}")
    print(f"batch_size: {args.batch_size}")
    print(f"dropout: {args.dropout}")
    print(f"num_layers: {args.num_layers}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"clip_grad_norm: {args.clip_grad_norm}")
    print(f"gpu: {args.gpu}")

    fold = 0
    dataset_name = args.dataset + "_" + str(fold) + f"_seed{seed}"

    print(
        f"--------------------------------------第{fold}折开始 -------------------------------------------")

    dataset_train = MyDataset(name=dataset_name, type="train")
    dataset_val = MyDataset(name=dataset_name, type="val")
    dataset_test = MyDataset(name=dataset_name, type="test")

    device = get_device(args.gpu)

    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = HeteroSAGE(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        dataname=args.dataset,
        fold=fold,
        num_layers=args.num_layers
    )

    num_workers = 0 if os.name == 'nt' else min(args.num_workers, os.cpu_count() or 1)
    train_loader = GraphDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = GraphDataLoader(dataset_val, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, num_workers=num_workers)

    model.to(device)

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.NAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )

    # AMP: GradScaler for mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print("AMP (Automatic Mixed Precision) is enabled.")

    # 学习率调度：预热 + 余弦退火（无重启，最终收敛更平稳）
    warmup_epochs = 5
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
        print(f"------第{epoch + 1}轮训练开始-----")

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

    path = "final_train/" + args.dataset + f"/seed{seed}"

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    model_path = path + '/' + str(args.dataset) + f'_fold{fold}_seed{seed}' + '_model.pkl'

    # 加载最优权重后以完整模型形式保存（保持与下游 torch.load 兼容）
    model.load_state_dict(best_state)
    torch.save(model, model_path)

    check_model = torch.load(model_path, weights_only=False)
    val_loss, val_accuracy, val_f1 = validate(check_model, loss_func, val_loader, device, len(dataset_val))
    print('-' * 89)
    print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Validation Loss: {val_loss:.3f}')
    print(
        f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Validation Acc: {val_accuracy:.3f}, '
        f'Validation F1: {val_f1:.3f}, Validation Score: {val_accuracy:.3f}')

    print('-' * 89)
    test_loss, test_accuracy = test(check_model, loss_func, test_loader, device, len(dataset_test))
    print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test Loss: {test_loss:.3f}')
    print(
        f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test average Accuracy:{test_accuracy:.3f}')

    print('-' * 89)
    print('Training finished.')


def method_name(start_time, end_time, fold, seed):
    total_training_time_seconds = end_time - start_time

    total_training_time_hours = total_training_time_seconds / 3600

    time_file_path = f'train_time/{eventlog}/third_training_time_{fold}_seed{seed}.txt'
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
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':


    seed_list = [133, 188, 456, 789, 1666]
    # seed_list = [789, 1666]

    list_eventlog = [
        'bpi13_closed_problems',
        'p2p',
        'bpi13_problems',
        'bpi13_incidents',
        'bpi12w_complete',
        'bpi12_all_complete',
        'BPI2020_Prepaid',
    ]

    for eventlog in list_eventlog:
        for seed in seed_list:
            set_seed(seed)
            print(f"seed:  {seed}--------------开始-记录时间------------")

            start_total = time.perf_counter()

            parser = argparse.ArgumentParser(description='BPIC')

            parser.add_argument("-d", "--dataset", type=str, default=eventlog, help="dataset to use")

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

            args = parser.parse_args([])

            train_val(args, seed)

            end_total = time.perf_counter()
            method_name(start_total, end_total,0, seed)
            print(f"--------------结束-记录时间------------")
