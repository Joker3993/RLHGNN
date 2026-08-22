import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, \
    precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from tqdm import tqdm

from MyDataset_final import MyDataset


def get_device(gpu):
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


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


def Final_test(args, seed):
    print("start testing...")
    print(f"dataset: {args.dataset}")
    print(f"seed: {seed}")
    print(f"batch_size: {args.batch_size}")
    print(f"gpu: {args.gpu}")

    fold = 0

    result_path = f"result_final_train/{args.dataset}/seed{seed}"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    outfile2 = open(result_path + "/" + args.dataset + f"_seed{seed}.txt", 'a')

    path = f"final_train/{args.dataset}/seed{seed}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    dataset_test = MyDataset(name=args.dataset + "_" + str(fold) + f"_seed{seed}", type="test")

    test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    device = get_device(args.gpu)

    model_path = path + '/' + str(args.dataset) + f'_fold{fold}_seed{seed}' + '_model.pkl'

    model = torch.load(model_path)
    model.to(device)

    model.eval()
    loss_func = nn.CrossEntropyLoss()

    total_loss = 0
    total_accuracy = 0
    total_step = 0
    Y_labels = []
    Y_preds = []

    with torch.no_grad():

        for batch in test_loader:
            batch = [item.to(device) for item in batch]

            Positive_graph, labels = batch

            logits = model(Positive_graph)

            loss = loss_func(logits, labels)

            total_loss += loss.item()
            total_accuracy += (logits.argmax(1) == labels).sum().item()

            y_pred = logits.argmax(1)

            Y_labels.append(labels)
            Y_preds.append(y_pred)

    Y_test_int = torch.cat(Y_labels, 0).to('cpu')

    preds_a = torch.cat(Y_preds, 0).to('cpu')

    accuracy = total_accuracy / len(dataset_test)
    precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                   pos_label=None)

    auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
    prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")

    g_mean = geometric_mean_score(Y_test_int, preds_a, average="macro")

    result_summary = (
        f"Result -> accuracy={accuracy:.4f} macro_f1={fscore:.4f} "
        f"AUC={auc_score_macro:.4f} PRAUC={prauc_score_macro:.4f} Gmean={g_mean:.4f}"
    )
    print(result_summary)

    outfile2.write(result_summary)
    outfile2.write('\n')
    outfile2.write('\n')
    outfile2.flush()
    outfile2.close()

    return {
        "accuracy": accuracy,
        "macro_f1": fscore,
        "auc_macro": auc_score_macro,
        "prauc_macro": prauc_score_macro,
        "g_mean": g_mean,
    }


def method_name(start_time, end_time, fold, seed):
    total_training_time_seconds = end_time - start_time

    time_file_path = f'pred_time/{eventlog}/predict_time_{fold}_seed{seed}.txt'
    os.makedirs(f'pred_time/{eventlog}', exist_ok=True)

    with open(time_file_path, 'w') as time_file:
        time_file.write(f"predict_time: {total_training_time_seconds:.3f} s\n")

    print("-" * 90)
    print("\n")

    print(f"{fold} fold---Total predict_time: {total_training_time_seconds:.3f} s")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_list = [133, 188, 456, 789, 1666]
    list_eventlog = [
        'p2p',
        'bpi13_closed_problems',
        'bpi13_problems',
        'bpi13_incidents',
        'bpi12w_complete',
        'bpi12_all_complete',
        'BPI2020_Prepaid',
    ]

    fold=0
    for eventlog in tqdm(list_eventlog):
        accuracies = []
        macro_f1s = []

        for seed in seed_list:
            set_seed(seed)
            start_total = time.perf_counter()

            parser = argparse.ArgumentParser(description='BPIC')

            parser.add_argument("-d", "--dataset", type=str, default=eventlog, help="dataset to use")

            parser.add_argument("--batch_size", type=int, default=64, help="batch_size")

            parser.add_argument("--gpu", type=int, default=0, help="gpu")

            args = parser.parse_args([])

            metrics = Final_test(args, seed)
            accuracies.append(metrics["accuracy"])
            macro_f1s.append(metrics["macro_f1"])

            end_total = time.perf_counter()
            method_name(start_total, end_total, fold, seed)

        summary_dir = os.path.join("result_final_train", eventlog)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f"{eventlog}_summary.txt")

        with open(summary_path, 'w') as summary_file:
            summary_file.write(f"Dataset: {eventlog}\n")
            for seed, (acc, f1) in zip(seed_list, zip(accuracies, macro_f1s)):
                summary_file.write(f"seed={seed} accuracy={acc:.4f} macro_f1={f1:.4f}\n")

            acc_mean = float(np.mean(accuracies))
            acc_std = float(np.std(accuracies, ddof=1))
            f1_mean = float(np.mean(macro_f1s))
            f1_std = float(np.std(macro_f1s, ddof=1))

            summary_file.write(f"accuracy_mean={acc_mean:.4f}\n")
            summary_file.write(f"accuracy_std={acc_std:.4f}\n")
            summary_file.write(f"macro_f1_mean={f1_mean:.4f}\n")
            summary_file.write(f"macro_f1_std={f1_std:.4f}\n")

        print(f"Summary saved to: {summary_path}")
        print(f"Accuracy mean/std: {acc_mean:.4f} / {acc_std:.4f}")
        print(f"Macro-F1 mean/std: {f1_mean:.4f} / {f1_std:.4f}")
