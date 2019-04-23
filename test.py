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

MIN_SIZE = 1
MAX_SIZE = max(MIN_SIZE+1, 20)

np.set_printoptions(linewidth=150, precision=2)

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
    if detected_attack > 0:
        precision = detected_attack/(detected_attack + false_positive)
        recall = detected_attack/(detected_attack + false_negtive)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision = 0
        recall = 0
        f1 = 0

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

    #df_X = df_X.sample(frac=1)
    #df_y = df_y.iloc[df_X.index]

    return df_X, df_y

if __name__ == '__main__':

    data_directoy = "../TestData/"
    filemon = "normal.csv"
    filetue = "dos.csv"
    filewed = "portscan.csv"
    filethr1 = "ddos.csv"

    file_list = [filemon, filetue, filewed, filethr1]
    file_list = [data_directoy + x for x in file_list]

    print("Loading data...")
    df_X, df_y = read_file(file_list[0])
    new_features = df_X.columns

    det = Detection(classes=classes , threshold_percentile=threshold_percentile, truth_size=truth_size,
                    truth_update_frac=truth_update_frac, truth_save_folder=truth_save_folder,
                    outlier_save_folder=outlier_save_folder, classifier_save_folder=classifier_save_folder)

    print("Initializing detection system...")
    det.initialize(df_X, df_X, np.ndarray.flatten(df_y.values))
    print("Training Complete")

    name = ["DoS", "PortScan", "DDoS"]

    for i in range(1,len(file_list)):
        print(name[i-1] + " Test...")
        df_X, df_y = read_file(file_list[i])
        y_pred = []
        outlier_indices = []
        size = df_X.shape[0]
        c = 0
        index = df_X.index.tolist()
        for k in range(len(index)):
            j = index[k]
            if c % 50 == 0 and c > 0:
                print("\rStep " + str(c) +" of " + str(size), sep=' ', end='', flush=True)
                #print("Step " + str(c) +" of " + str(size))
                """
                y_now = df_y.iloc[index[:k]]
                cls_rep = classification_report(y_now.values, y_pred, labels=classes, output_dict=True)
                print("F1 Scores => ", end='')
                print("Normal: ", np.round(cls_rep['0']['f1-score'],2), end='')
                print(" | Dos: ", np.round(cls_rep['1']['f1-score'],2), end='')
                print(" | PortScan: ", np.round(cls_rep['2']['f1-score'],2), end='')
                print(" | DDoS: ", np.round(cls_rep['3']['f1-score'],2), end='')
                print(" | Weighted: ", np.round(cls_rep['weighted avg']['f1-score'],2))
                """
            c += 1

            X = df_X.iloc[[j]]
            y = df_y.iloc[[j]]
            outlier = det.detect_outliers(X)
            if outlier:
                outlier_indices.append(j)
                pred = det.classfiy(X)[0]
                if pred == 0:
                    det.update_outlier(X)
                det.update_classifier(X, np.ndarray.flatten(y.values))
                y_pred.append(pred)
            else:
                det.update_outlier(X)
                y_pred.append(0)
        print("")
        print(classification_report(df_y.values, y_pred, labels=classes))
