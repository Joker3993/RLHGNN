import argparse
import copy
import os
import pickle
import random

import numpy as np
import torch
import torchvision
from dgl.dataloading import GraphDataLoader
from torch import nn, optim
import matplotlib.pyplot as plt

import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau

from MyDataset import MyDataset

from model.model import HeteroSAGE

warnings.filterwarnings("ignore", category=UserWarning)


class Tran:
    def __init__(self, eventlog, choice):
        self._evenlog = eventlog
        self._fold = 0
        self._choice = choice

    def get_device(self, gpu):

        if torch.cuda.is_available() and gpu < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu}')
        else:
            return torch.device('cpu')

    def train(self, model, train_loader, loss_func, optimizer, device, data_length):

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

    def validate(self, model, loss_func, val_loader, device, data_length):

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

    def test(self, model, loss_func, val_loader, device, data_length):

        model.eval()

        total_loss = 0
        total_accuracy = 0

        total_step = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = [item.to(device) for item in batch]

                Positive_graph, labels = batch

                logits = model(Positive_graph)

                loss = loss_func(logits, labels)

                total_loss += loss.item()
                total_accuracy += (logits.argmax(1) == labels).sum().item()

                total_step += 1

        average_loss = total_loss / total_step
        average_accuracy = total_accuracy / data_length

        return average_loss, average_accuracy

    def train_val(self, args):
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

            self._fold = fold

            print(
                f"--------------------------------------第{self._fold}折开始 ， 方案 : {self._choice}-------------------------------------------")

            dataset_train = MyDataset(name=args.dataset + "_" + str(self._fold), type="train", choice=self._choice)
            dataset_val = MyDataset(name=args.dataset + "_" + str(self._fold), type="val", choice=self._choice)
            dataset_test = MyDataset(name=args.dataset + "_" + str(self._fold), type="test", choice=self._choice)

            device = self.get_device(args.gpu)

            model = HeteroSAGE(
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                dataname=self._evenlog,
                fold=self._fold,
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

                train_loss, train_accuracy = self.train(model, train_loader, loss_func, optimizer, device,
                                                        len(dataset_train))
                val_loss, val_accuracy = self.validate(model, loss_func, val_loader, device, len(dataset_val))
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

            path = f"Pretrain/action_{self._choice}/" + args.dataset
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            model_path = path + '/' + str(args.dataset) + f'_fold{fold}' + '_model.pkl'

            torch.save(best_model, model_path)

            check_model = torch.load(model_path)
            val_loss, val_accuracy = self.validate(check_model, loss_func, val_loader, device, len(dataset_val))
            print('-' * 89)
            print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Validation Loss: {val_loss:.3f}')

            print('-' * 89)
            test_loss, test_accuracy = self.test(check_model, loss_func, test_loader, device, len(dataset_test))
            print(f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test Loss: {test_loss:.3f}')
            print(
                f'Best_Epoch [{best_epoch:d}/{args.num_epochs}].In best model: Test average Accuracy:{test_accuracy:.3f}')

            print('-' * 89)
            print('Training finished.')

    def tran_main(self):

        parser = argparse.ArgumentParser(description='BPIC')

        parser.add_argument("-d", "--dataset", type=str, default=self._evenlog, help="dataset to use")

        parser.add_argument("--hidden-dim", type=int, default=128, help="dim of hidden")

        parser.add_argument("--num-epochs", type=int, default=50, help="number of epoch")

        parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

        parser.add_argument("--batch-size", type=int, default=64, help="batch size")

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")

        parser.add_argument("--num_layers", type=int, default=2, help="num_layers")

        parser.add_argument("--gpu", type=int, default=0, help="gpu")

        args = parser.parse_args()

        self.train_val(args)
