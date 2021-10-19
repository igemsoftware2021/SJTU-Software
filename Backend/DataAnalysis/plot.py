from RDA import *
import pandas as pd
from sklearn.model_selection import train_test_split
from svm_model import *
from feature_selection import *
from dataset import load_data, Data_miRna

import warnings

warnings.filterwarnings("ignore")

# 载入数据，数据已经进行了log操作（normalization），数值可以直接进行利用
file1 = "./data/lung/lung_bh_1.txt"
file2 = "./data/lung/lung_us_1.txt"


def select(file1, file2):
    """file1:正常人数据 file2:病人数据"""
    data_new = load_data(file1, file2)

    y_new = Data_miRna.getY(data_new)

    max_mix_score, max_accuracy, max_f1, best_report, \
    best_params, best_w, best_b, combine, best_X_test, best_y_test = featSelect_plot(data_new, y_new)

    return max_mix_score, max_accuracy, max_f1, best_report, best_params, best_w, best_b, combine, best_X_test, best_y_test


max_mix_score, max_accuracy, max_f1, best_report, \
best_params, best_w, best_b, combine, best_X_test, best_y_test = select(file1, file2)

print()
