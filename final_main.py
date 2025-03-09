import argparse
import copy
import os
import pickle
import random
import time

import numpy as np
import torch

from dgl.dataloading import GraphDataLoader
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, optim

import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

from MyDataset_final import MyDataset
from model.model import HeteroSAGE

warnings.filterwarnings("ignore", category=UserWarning)


def get_device(gpu):
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def train(model, train_loader, loss_func, optimizer, device, data_length):
    model.train()

    total_loss = 0

    total_accuracy = 0
    total_step = 0

    for batch in train_loader:
        batch = [item.to(device) for item in batch]

        Positive_graph, labels = batch

        logits = model(Positive_graph)

        logits = logits.to(device)

        loss = loss_func(logits, labels)

        optimizer.zero_grad()
        loss.backward()
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

    with torch.no_grad():
        for batch in val_loader:
            batch = [item.to(device) for item in batch]

            Positive_graph, labels = batch

            logits = model(Positive_graph)

            logits = logits.to(device)

            loss = loss_func(logits, labels)

            total_loss += loss.item()
            total_accuracy += (logits.argmax(1) == labels).sum().item()
            total_test_step += 1

    return total_loss / total_test_step, total_accuracy / data_length


def test(model, loss_func, val_loader, device, data_length):
    model.eval()

    total_loss = 0
    total_accuracy = 0

    total_step = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = [item.to(device) for item in batch]

            Positive_graph, labels = batch

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


def train_val(args):
    print("start training...")
    print("Training with the following arguments:")
    print(f"dataset: {args.dataset}")
    print(f"hidden_dim: {args.hidden_dim}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"lr: {args.lr}")
    print(f"batch_size: {args.batch_size}")
    print(f"dropout: {args.dropout}")
    print(f"num_layers: {args.num_layers}")
    print(f"gpu: {args.gpu}")

    for fold in range(3):

        print(
            f"--------------------------------------第{fold}折开始 -------------------------------------------")

        dataset_train = MyDataset(name=args.dataset + "_" + str(fold), type="train")
        dataset_val = MyDataset(name=args.dataset + "_" + str(fold), type="val")
        dataset_test = MyDataset(name=args.dataset + "_" + str(fold), type="test")

        device = get_device(args.gpu)

        model = HeteroSAGE(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            dataname=args.dataset,
            fold=fold,
            num_layers=args.num_layers
        )

        train_loader = GraphDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        val_loader = GraphDataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
        test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

        model.to(device)

        loss_func = nn.CrossEntropyLoss()

        optimizer = optim.NAdam(model.parameters(), lr=args.lr)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        patience = 10
        no_improvement_count = 0
        best_epoch = 0
        best_model = None
        best_val_acc = 0

        for epoch in range(args.num_epochs):
            print(f"------第{epoch + 1}轮训练开始-----")

            train_loss, train_accuracy = train(model, train_loader, loss_func, optimizer, device,
                                               len(dataset_train))
            val_loss, val_accuracy = validate(model, loss_func, val_loader, device, len(dataset_val))
            print(
                f'Epoch [{epoch + 1}/{args.num_epochs}], '
                f'Training Loss: {train_loss:.3f},'
                f'Train_accuracy: {train_accuracy:.3f},'
                f' Validation Accuracy: {val_accuracy:.3f}')

            if val_accuracy >= best_val_acc:
                best_val_acc = val_accuracy
                best_epoch = epoch + 1
                no_improvement_count = 0
                best_model = copy.deepcopy(model)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping: Validation accuracy has not improved for {} epochs.".format(patience))
                    break

            scheduler.step(val_loss)

        path = "final_train/" + args.dataset

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        model_path = path + '/' + str(args.dataset) + f'_fold{fold}' + '_model.pkl'

        torch.save(best_model, model_path)

        check_model = torch.load(model_path)
        val_loss, val_accuracy = validate(check_model, loss_func, val_loader, device, len(dataset_val))
        print('-' * 89)
        print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Validation Loss: {val_loss:.3f}')

        print('-' * 89)
        test_loss, test_accuracy = test(check_model, loss_func, test_loader, device, len(dataset_test))
        print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test Loss: {test_loss:.3f}')
        print(
            f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test average Accuracy:{test_accuracy:.3f}')

        print('-' * 89)
        print('Training finished.')


def method_name(start_time, end_time, fold):
    total_training_time_seconds = end_time - start_time

    total_training_time_hours = total_training_time_seconds / 3600

    time_file_path = f'train_time/{eventlog}/third_training_time_{fold}.txt'
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

        parser = argparse.ArgumentParser(description='BPIC')

        parser.add_argument("-d", "--dataset", type=str, default=eventlog, help="dataset to use")

        parser.add_argument("--hidden-dim", type=int, default=128, help="dim of hidden")

        parser.add_argument("--num-epochs", type=int, default=50, help="number of epoch")

        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

        parser.add_argument("--batch-size", type=int, default=64, help="batch size")

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")

        parser.add_argument("--num_layers", type=int, default=2, help="num_layers")

        parser.add_argument("--gpu", type=int, default=0, help="gpu")

        args = parser.parse_args()

        train_val(args)

        end_total = time.perf_counter()
        method_name(start_total, end_total, 123)
        print(f"--------------结束-记录时间------------")
