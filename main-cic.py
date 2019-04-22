import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

import random as rd
import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from config.config import *

from detection.detection import Detection
from detection.preprocessing.preprocessing import process_labels

MIN_SIZE = 500
MAX_SIZE = max(MIN_SIZE+1, 2000)

np.set_printoptions(linewidth=150)

def outlier_report(df_y, outlier_indices):
    """

    :param df_y:
    :param outlier_indices:

    :return:
    """
    detected_attack = 0
    detected_normal = 0
    false_negtive = 0
    false_positive = 0
    count = 0
    max_count = len(outlier_indices)

    for i in range(df_y.shape[0]):
        label_num = df_y[i]

        if count < max_count and i == outlier_indices[count]:
            if label_num != 0:
                detected_attack += 1
            else:
                false_positive += 1
            count += 1
        else:
            if label_num != 0:
                false_negtive += 1
            else:
                detected_normal += 1
    total = detected_attack + detected_normal + false_positive + false_negtive
    print("")
    print("=========================================")
    print("             OUTLIER ACCURACY            ")
    print("=========================================")
    print("          |   Det Attack       Det Normal  ")
    print("----------|--------------------------------")
    print("Is Attack |   ", detected_attack, "          ", false_negtive)
    print("Is Normal |   ", false_positive, "           ", detected_normal)
    print("")
    print("          |  Det Attack        Det Normal ")
    print("-----------------------------------------")
    print("Is Attack |  ", round(detected_attack / total * 100, 4), "         ", round(false_negtive / total * 100, 4))
    print("Is Normal |  ", round(false_positive / total * 100, 4), "          ",
          round(detected_normal / total * 100, 4))
    print("")
    precision = detected_attack/(detected_attack + false_positive)
    recall = detected_attack/(detected_attack + false_negtive)
    f1 = (2*precision*recall)/(precision+recall)
    print("Precision: ", round(precision, 3))
    print("Recall: ", round(recall, 3))
    print("F1-Score: ", round(f1, 3))
    return [precision, recall, f1]

def classification_result(y, y_pred):
    """

    :param y:
    :param y_pred:

    :return:
    """
    assert len(y) == len(y_pred)
    correct = []
    wrong = []
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            correct.append(i)
        else:
            wrong.append(i)
    return correct, wrong

def read_file(name):
    """

    :param name:

    :return:
    """
    df = pd.read_csv(name, na_values="Infinity", dtype=dtypes)
    df = df.replace("Infinity", sys.maxsize)
    df = df.fillna(sys.maxsize)

    df_X = df.iloc[:, 0:-1].copy()
    df_X = df_X.apply(pd.to_numeric)

    df_y = df.iloc[:, -1].copy()
    df_y = process_labels(df_y, labels, classes)

    return df_X, df_y

if __name__ == '__main__':

    data_directoy = "../MachineLearningCVE/"
    filemon = "Monday-WorkingHours.pcap_ISCX.csv"
    filetue = "Tuesday-WorkingHours.pcap_ISCX.csv"
    filewed = "Wednesday-workingHours.pcap_ISCX.csv"
    filethr1 = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    filethr2 = "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
    filefri1 = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    filefri2 = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    filefri3 = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

    file_list = [filemon, filetue, filewed, filethr1, filethr2, filefri1, filefri2, filefri3]
    file_list = [data_directoy + x for x in file_list]

    print("Loading data...")
    df_X, df_y = read_file(file_list[0])
    print("Done")

    det = Detection(classes=classes , threshold_percentile=threshold_percentile, truth_size=truth_size,
                    truth_update_frac=truth_update_frac, truth_save_folder=truth_save_folder,
                    outlier_save_folder=outlier_save_folder, classifier_save_folder=classifier_save_folder)

    print("Initializing detectors...")
    i = 0
    size = df_X.shape[0]
    new_features = df_X.columns

    while i < size:
        print(str(i) + "/" + str(size))
        j = rd.randint(i + MIN_SIZE, i + MAX_SIZE)
        j = min(j, size)

        X = pd.DataFrame(df_X[i:j].values, columns=new_features, dtype=np.float64)
        y= np.ndarray.flatten(pd.DataFrame(df_y[i:j].values, columns=['class'], dtype=np.int64).values)
        if i == 0:
            det.initialize(X, X, y)
            i = j
            continue
        det.update_outlier(X)
        #det.update_classifier(X, y)
        i = j
    print("Training Complete")

    cls_reports = []
    out_reports = []
    breakpoints = {x : [] for x in file_list[1:]}
    for file in file_list[1:]:
        print("Loading data...")
        df_X, df_y = read_file(file)
        df_X = pd.DataFrame(df_X, columns=new_features, dtype=np.float64)
        print("Done")

        print("Analyzing file " + file)
        out_idx = []
        final_pred = []
        y_pred_list = []
        i = 0
        c = 0
        size = df_X.shape[0]
        while i < size:
            breakpoints[file].append(i)
            print(str(i) + "/" + str(size))
            j = rd.randint(i + MIN_SIZE, i + MAX_SIZE)
            j = min(j, size)
            X = pd.DataFrame(df_X[i:j].values, columns=new_features, dtype=np.float64)
            y = pd.DataFrame(df_y[i:j].values, columns=['class'], dtype=np.int64)

            outlier_indices = det.detect_outliers(X)
            normal_indices = X.index.difference(outlier_indices)
            det.update_outlier(X.iloc[normal_indices])

            outlier_X = X.iloc[outlier_indices]
            outlier_y = y.iloc[outlier_indices]

            y_pred = det.classfiy(outlier_X)
            det.update_classifier(outlier_X, np.ndarray.flatten(outlier_y.values))

            normal_X = X.iloc[outlier_y.loc[outlier_y['class'] == 0].index]
            det.update_outlier(normal_X)

            out_idx.extend([x + breakpoints[file][c] for x in outlier_indices])
            c += 1
            y_pred_list.extend(y_pred)

            if c % 10 == 0:
                _ = outlier_report(np.ndarray.flatten(y.values), outlier_indices)
                print(classification_report(np.ndarray.flatten(outlier_y.values), y_pred, labels=classes))
                print(confusion_matrix(np.ndarray.flatten(outlier_y.values), y_pred, labels=classes))

            outcount = 0
            for k in range(X.shape[0]):
                if outlier_indices[min(outcount, len(outlier_indices) - 1)] == k:
                    final_pred.append(y_pred[outcount])
                    outcount += 1
                else:
                    final_pred.append(0)  # Class Normal
            i = j

        cls_reports.append(classification_report(df_y.values, final_pred, labels=classes, output_dict=True))

        print("FINAL REPORT OF FILE " + file)
        print("======================================================================================")
        out_reports.append(outlier_report(np.ndarray.flatten(df_y.values), out_idx))
        print("======================================================================================")
        print(classification_report(df_y.iloc[out_idx].values, y_pred_list, labels=classes))
        print(confusion_matrix(df_y.iloc[out_idx].values, y_pred_list, labels=classes))
        print("======================================================================================")
        print(classification_report(df_y.values, final_pred, labels=classes))
        print(confusion_matrix(df_y.values, final_pred, labels=classes))
        print("======================================================================================")

    micro_avg = []
    macro_avg = []
    weighted = []
    support = []
    for i in range(len(cls_reports)):
        micro_avg.append(cls_reports[i]['micro avg'])
        macro_avg.append(cls_reports[i]['macro avg'])
        weighted.append(cls_reports[i]['weighted avg'])
        support.append(cls_reports[i]['weighted avg']['support'])

    total = sum(support)
    weights = [x/total for x in support]
    print(weights)
    mi = [0,0,0]
    ma = [0,0,0]
    w = [0,0,0]
    o = [0,0,0]
    word = ['precision', 'recall', 'f1-score']
    for i in range(len(cls_reports)):
        for j in range(3):
            mi[j] += weights[i] * micro_avg[i][word[j]]
            ma[j] += weights[i] * macro_avg[i][word[j]]
            w[j] += weights[i] * weighted[i][word[j]]
            o[j] += weights[i] * out_reports[i][j]
    print(o)
    print(mi)
    print(ma)
    print(w)