import pandas as pd
import numpy as np
import random as rd
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from config.config import *

from detection.detection import Detection
from detection.preprocessing.preprocessing import process_features
from detection.preprocessing.preprocessing import process_labels

MIN_SIZE = 50
MAX_SIZE = max(MIN_SIZE+1, 200)


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

if __name__ == '__main__':
    data_directoy = "../NSL-KDD/"
    train_file = data_directoy + "KDDTrain+.csv"
    train_file_20_percent = data_directoy + "KDDTrain+_20Percent.csv"
    test_file = data_directoy + "KDDTest+.csv"

    normal_file = data_directoy + "20/normal.csv"
    data_file = data_directoy + "20/data.csv"
    labels_file = data_directoy + "20/labels.csv"

    det = Detection(classes=classes , threshold_percentile=threshold_percentile, truth_size=truth_size,
                    truth_update_frac=truth_update_frac, truth_save_folder=truth_save_folder,
                    outlier_save_folder=outlier_save_folder, classifier_save_folder=classifier_save_folder)

    df = pd.read_csv(data_directoy + train_file, header=None, names=headers)
    df_X = df[features].copy()
    df_y = df[target].copy()

    df_X = process_features(df_X, categorical_feature_list, categories_list)
    df_X = df_X.astype(np.float64)
    df_y = process_labels(df_y, labels, classes)

    normal_X = df_X.iloc[df_y.loc[df_y['class'] == 0].index]
    truth_size = min(truth_size, normal_X.shape[0])
    # Choose "truth_size" randomly sampled rows from all normal records
    truth = normal_X.sample(n=truth_size, random_state=None)

    df_X = df_X.drop(truth.index)
    df_y = df_y.drop(truth.index)
    y_cls = df_y['class']
    det.initialize(truth, df_X, y_cls)

    df_t = pd.read_csv(data_directoy + test_file, header=None, names=headers)
    Xt = df_t[features].copy()
    yt = df_t[target].copy()

    Xt = process_features(Xt, categorical_feature_list, categories_list)
    Xt = Xt.astype(np.float64)
    yt = process_labels(yt, labels, classes)
    yt_cls = yt['class']

    i = 0
    s = Xt.shape[0]
    list_Xt = []
    list_yt = []
    new_features = Xt.columns.values

    breakpoints = []

    while i < s:
        breakpoints.append(i)
        j = rd.randint(i + MIN_SIZE, i + MAX_SIZE)
        j = min(j, s)
        Xt_slice = pd.DataFrame(Xt[i:j].values, columns=new_features, dtype=np.float64)
        yt_slice = pd.DataFrame(yt_cls[i:j].values, columns=['class'], dtype=np.int64)
        list_Xt.append(Xt_slice)
        list_yt.append(yt_slice)
        i = j

    y_pred_list = []
    out_idx = []
    final_pred = []

    for i in range(len(list_Xt)):
        print("Step " + str(i))
        X = list_Xt[i]
        y = list_yt[i]
        y_cls = y

        outlier_indices = det.detect_outliers(X)
        normal_indices = X.index.difference(outlier_indices)
        det.update_outlier(X.iloc[normal_indices])

        outlier_X = X.iloc[outlier_indices]
        outlier_y = y_cls.iloc[outlier_indices]
        normal_X = X.iloc[outlier_y.loc[outlier_y['class'] == 0].index]

        y_pred = det.classfiy(outlier_X)
        det.update_classifier(outlier_X, np.ndarray.flatten(outlier_y.values))
        det.update_outlier(normal_X)

        out_idx.extend([x + breakpoints[i] for x in outlier_indices])

        outcount = 0
        for j in range(list_Xt[i].shape[0]):
            if outlier_indices[min(outcount, len(outlier_indices) - 1)] == j:
                final_pred.append(y_pred[outcount])
                outcount += 1
            else:
                final_pred.append(0) # Class Normal
        y_pred_list.extend(y_pred)

    print("==================================================================================")
    print("FINAL REPORT")

    _ = outlier_report(yt_cls, out_idx)
    print(classification_report(yt_cls.iloc[out_idx], y_pred_list, labels=classes))
    print(classification_report(yt_cls, final_pred, labels=classes))


