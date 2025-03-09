import argparse
import os
import pickle
import time

import torch
from dgl.dataloading import GraphDataLoader
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


def Final_test(args):
    print("start testing...")
    print(f"dataset: {args.dataset}")
    print(f"batch_size: {args.batch_size}")
    print(f"gpu: {args.gpu}")

    for fold in range(3):

        result_path = "result_final_train/" + args.dataset

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        outfile2 = open(result_path + "/" + args.dataset + ".txt", 'a')

        path = "final_train/" + args.dataset
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        dataset_test = MyDataset(name=args.dataset + "_" + str(fold), type="test")

        test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

        device = get_device(args.gpu)

        model_path = path + '/' + str(args.dataset) + f'_fold{fold}' + '_model.pkl'

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

        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                       pos_label=None)

        auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")

        print(classification_report(Y_test_int, preds_a, digits=3))
        print(f"AUC:{auc_score_macro}")
        print(f"PRAUC:{prauc_score_macro}")

        outfile2.write(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: ' + str(prauc_score_macro))
        outfile2.write('\n')
        outfile2.write('\n')
        outfile2.flush()
        outfile2.close()


def method_name(start_time, end_time, fold):
    total_training_time_seconds = end_time - start_time

    time_file_path = f'pred_time/{eventlog}/predict_time_{fold}.txt'
    os.makedirs(f'pred_time/{eventlog}', exist_ok=True)

    with open(time_file_path, 'w') as time_file:
        time_file.write(f"predict_time: {total_training_time_seconds:.3f} s\n")

    print("-" * 90)
    print("\n")

    print(f"{fold} fold---Total predict_time: {total_training_time_seconds:.3f} s")


if __name__ == '__main__':

    list_eventlog = [
        'bpi13_closed_problems',
        'bpi13_problems',
        'bpi13_incidents',
        'bpi12w_complete',
        'BPI2020_Prepaid',
        'bpi12_all_complete',
    ]

    for eventlog in tqdm(list_eventlog):
        start_total = time.perf_counter()

        parser = argparse.ArgumentParser(description='BPIC')

        parser.add_argument("-d", "--dataset", type=str, default=eventlog, help="dataset to use")

        parser.add_argument("--batch_size", type=int, default=64, help="batch_size")

        parser.add_argument("--gpu", type=int, default=0, help="gpu")

        args = parser.parse_args()

        Final_test(args)

        end_total = time.perf_counter()
        method_name(start_total, end_total, 123)
