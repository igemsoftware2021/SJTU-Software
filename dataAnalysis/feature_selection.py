# 从差异表达分析筛选出的指标中筛选出4个指标，匹配出最好的svm模型

from RDA import *
import pandas as pd
from sklearn.model_selection import train_test_split
from svm_model import *


def getCombine(allnames):
    ids = []
    names = []
    for i in range(len(allnames)):
        for j in range(i + 1, len(allnames)):
            ids.append([i, j])
            names.append([allnames[i], allnames[j]])
    return ids, names


def featSelect(data, y):
    miRNA_names, miRNA_names_up, miRNA_names_down = RDA_new(data)
    # 正负两种指标都进行两两组合
    up_ids, up_names = getCombine(miRNA_names_up)
    down_ids, down_names = getCombine(miRNA_names_down)

    data_ = pd.concat([data.bh, data.us], axis=1)

    max_accuracy = 0
    max_f1 = 0
    max_mix_score = 0
    best_report = 0
    best_params = 0
    best_w = 0
    best_b = 0
    for i, j in up_names:
        for m, n in down_names:
            X = data_.loc[[i, j, m, n], :]
            X = np.array(X)
            X = np.transpose(X)

            # 划分训练集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

            # 进行训练
            accuracy, f1, report, params, w, b = SVC(X_train, y_train, X_test, y_test)
            mix_score = accuracy + f1
            if mix_score > max_mix_score:
                max_mix_score = mix_score
                max_accuracy = accuracy
                max_f1 = f1
                best_report = report
                best_params = params
                best_w = w
                best_b = b
                combine = [i, j, m, n]

    return max_mix_score, max_accuracy, max_f1, best_report, best_params, best_w, best_b, combine
