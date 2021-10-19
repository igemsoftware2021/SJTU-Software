# 把患病的人作为群体对待，然后找患病群体和正常群体表达值差异比较大的miRNA。
# 导入包
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# t-test函数，返回p-value
def t_test(a, b):
    a = np.array(a)
    b = np.array(b)
    t1, p1 = stats.levene(a, b)
    if p1 > 0.05:
        t2, p2 = stats.ttest_ind(a, b, equal_var=False)
    else:
        t2, p2 = stats.ttest_ind(a, b)
    return p2


def RDA(data):
    # 每个基因（行）BH样本（正常人员）的表达平均值和方差
    bh = data.loc[:, 'TCGA_CV_6933_11': 'TCGA_WA_A7GZ_11']
    bh_mean = data.loc[:, 'TCGA_CV_6933_11': 'TCGA_WA_A7GZ_11'].mean(axis=1)
    bh_var = data.loc[:, 'TCGA_CV_6933_11': 'TCGA_WA_A7GZ_11'].var(axis=1)

    # 每个基因（行）US样本（病人）的表达平均值和方差
    us = data.loc[:, 'TCGA_4P_AA8J_01':'TCGA_WA_A7H4_01']
    us_mean = data.loc[:, 'TCGA_4P_AA8J_01':'TCGA_WA_A7H4_01'].mean(axis=1)
    us_var = data.loc[:, 'TCGA_4P_AA8J_01':'TCGA_WA_A7H4_01'].var(axis=1)

    cout = []  # 得到的差异表达值fold change

    miRNA_list = []  # 筛选出的miRNA序号
    miRNA_list_up = []  # 筛选出的miRNA_up组序号
    miRNA_list_down = []  # 筛选出的miRNA_down组序号
    miRNA_cout = []  # 筛选出的miRNA对应的差异表达值
    miRNA_cout_up = []  # 筛选出的miRNA_up对应的差异表达值
    miRNA_cout_down = []  # 筛选出的miRNA_down对应的差异表达值

    # 设置阈值进行筛选（同时也进行了统计检验）
    for i in range(data.shape[0]):
        if bh_mean[i] != 0:
            x = us_mean[i] / bh_mean[i]
        else:
            x = 0
        cout.append(x)

    for i in range(data.shape[0]):
        # p = t_test(us.iloc[i, :], bh.iloc[i, :])
        if cout[i] > 2:
            miRNA_list_up.append(i)
            miRNA_cout_up.append(cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(cout[i])
        if cout[i] < 0.5 and cout[i] > 0:
            miRNA_list_down.append(i)
            miRNA_cout_down.append(-1 / cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(-1 / cout[i])

    print(miRNA_list_up)
    print(len(miRNA_list_up))
    print(miRNA_list_down)
    print(len(miRNA_list_down))

    miRNA_names = []  # 筛选出来的miRNA名称
    miRNA_names_up = []
    miRNA_names_down = []

    for i in range(len(miRNA_list)):
        miRNA_names.append(data.index[miRNA_list[i]])
    for i in range(len(miRNA_list_up)):
        miRNA_names_up.append(data.index[miRNA_list_up[i]])
    for i in range(len(miRNA_list_down)):
        miRNA_names_down.append(data.index[miRNA_list_down[i]])

    print(miRNA_names)

    # 将得到的分数值进行绘图
    plt.figure()
    plt.bar(miRNA_names, miRNA_cout)
    plt.xticks(rotation=60)
    plt.xlabel('miRNA', fontsize=2)

    plt.savefig('bar1.png', dpi=1600)
    plt.show()

    # 根据差异分析筛选出特征后的X
    X = data.loc[miRNA_names, :]
    X = np.array(X)
    X = np.transpose(X)
    # X=np.log10(X)

    y = pd.read_table('miR_data_y.txt')
    y = np.array(y).ravel()
    return X, y


def RDA_new(data):
    # 每个基因（行）BH样本（正常人员）的表达平均值和方差
    bh = data.bh
    bh_mean = data.bh_mean
    bh_var = data.bh_var

    # 每个基因（行）US样本（病人）的表达平均值和方差
    us = data.us
    us_mean = data.us_mean
    us_var = data.us_var

    cout = []  # 得到的差异表达值fold change

    miRNA_list = []  # 筛选出的miRNA序号
    miRNA_list_up = []  # 筛选出的miRNA_up组序号
    miRNA_list_down = []  # 筛选出的miRNA_down组序号
    miRNA_cout = []  # 筛选出的miRNA对应的差异表达值
    miRNA_cout_up = []  # 筛选出的miRNA_up对应的差异表达值
    miRNA_cout_down = []  # 筛选出的miRNA_down对应的差异表达值

    # 设置阈值进行筛选（同时也进行了统计检验）
    for i in range(bh.shape[0]):
        if bh_mean.iloc[i] != 0:
            x = us_mean.iloc[i] / bh_mean.iloc[i]
        else:
            x = 0
        cout.append(x)

    for i in range(bh.shape[0]):
        # p = t_test(us.iloc[i, :], bh.iloc[i, :])
        if cout[i] > 2:
            miRNA_list_up.append(i)
            miRNA_cout_up.append(cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(cout[i])
        if cout[i] < 0.5 and cout[i] > 0:
            miRNA_list_down.append(i)
            miRNA_cout_down.append(-1 / cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(-1 / cout[i])

    miRNA_names = []  # 筛选出来的miRNA名称
    miRNA_names_up = []
    miRNA_names_down = []

    for i in range(len(miRNA_list)):
        miRNA_names.append(bh.index[miRNA_list[i]])
    for i in range(len(miRNA_list_up)):
        miRNA_names_up.append(bh.index[miRNA_list_up[i]])
    for i in range(len(miRNA_list_down)):
        miRNA_names_down.append(bh.index[miRNA_list_down[i]])

    return miRNA_names, miRNA_names_up, miRNA_names_down

def RDA_new_6(data):
    # 每个基因（行）BH样本（正常人员）的表达平均值和方差
    bh = data.bh
    bh_mean = data.bh_mean
    bh_var = data.bh_var

    # 每个基因（行）US样本（病人）的表达平均值和方差
    us = data.us
    us_mean = data.us_mean
    us_var = data.us_var

    cout = []  # 得到的差异表达值fold change

    miRNA_list = []  # 筛选出的miRNA序号
    miRNA_list_up = []  # 筛选出的miRNA_up组序号
    miRNA_list_down = []  # 筛选出的miRNA_down组序号
    miRNA_cout = []  # 筛选出的miRNA对应的差异表达值
    miRNA_cout_up = []  # 筛选出的miRNA_up对应的差异表达值
    miRNA_cout_down = []  # 筛选出的miRNA_down对应的差异表达值

    # 设置阈值进行筛选（同时也进行了统计检验）
    for i in range(bh.shape[0]):
        if bh_mean.iloc[i] != 0:
            x = us_mean.iloc[i] / bh_mean.iloc[i]
        else:
            x = 0
        cout.append(x)

    for i in range(bh.shape[0]):
        # p = t_test(us.iloc[i, :], bh.iloc[i, :])
        if cout[i] > 6:
            miRNA_list_up.append(i)
            miRNA_cout_up.append(cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(cout[i])
        if cout[i] < 1/6 and cout[i] > 0:
            miRNA_list_down.append(i)
            miRNA_cout_down.append(-1 / cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(-1 / cout[i])

    miRNA_names = []  # 筛选出来的miRNA名称
    miRNA_names_up = []
    miRNA_names_down = []

    for i in range(len(miRNA_list)):
        miRNA_names.append(bh.index[miRNA_list[i]])
    for i in range(len(miRNA_list_up)):
        miRNA_names_up.append(bh.index[miRNA_list_up[i]])
    for i in range(len(miRNA_list_down)):
        miRNA_names_down.append(bh.index[miRNA_list_down[i]])

    return miRNA_names, miRNA_names_up, miRNA_names_down

def RDA_new_4(data):
    # 每个基因（行）BH样本（正常人员）的表达平均值和方差
    bh = data.bh
    bh_mean = data.bh_mean
    bh_var = data.bh_var

    # 每个基因（行）US样本（病人）的表达平均值和方差
    us = data.us
    us_mean = data.us_mean
    us_var = data.us_var

    cout = []  # 得到的差异表达值fold change

    miRNA_list = []  # 筛选出的miRNA序号
    miRNA_list_up = []  # 筛选出的miRNA_up组序号
    miRNA_list_down = []  # 筛选出的miRNA_down组序号
    miRNA_cout = []  # 筛选出的miRNA对应的差异表达值
    miRNA_cout_up = []  # 筛选出的miRNA_up对应的差异表达值
    miRNA_cout_down = []  # 筛选出的miRNA_down对应的差异表达值

    # 设置阈值进行筛选（同时也进行了统计检验）
    for i in range(bh.shape[0]):
        if bh_mean.iloc[i] != 0:
            x = us_mean.iloc[i] / bh_mean.iloc[i]
        else:
            x = 0
        cout.append(x)

    for i in range(bh.shape[0]):
        # p = t_test(us.iloc[i, :], bh.iloc[i, :])
        if cout[i] > 4:
            miRNA_list_up.append(i)
            miRNA_cout_up.append(cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(cout[i])
        if cout[i] < 0.25 and cout[i] > 0:
            miRNA_list_down.append(i)
            miRNA_cout_down.append(-1 / cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(-1 / cout[i])

    miRNA_names = []  # 筛选出来的miRNA名称
    miRNA_names_up = []
    miRNA_names_down = []

    for i in range(len(miRNA_list)):
        miRNA_names.append(bh.index[miRNA_list[i]])
    for i in range(len(miRNA_list_up)):
        miRNA_names_up.append(bh.index[miRNA_list_up[i]])
    for i in range(len(miRNA_list_down)):
        miRNA_names_down.append(bh.index[miRNA_list_down[i]])

    return miRNA_names, miRNA_names_up, miRNA_names_down

def RDA_new_plot(data):
    # 每个基因（行）BH样本（正常人员）的表达平均值和方差
    bh = data.bh
    bh_mean = data.bh_mean
    bh_var = data.bh_var

    # 每个基因（行）US样本（病人）的表达平均值和方差
    us = data.us
    us_mean = data.us_mean
    us_var = data.us_var

    cout = []  # 得到的差异表达值fold change

    miRNA_list = []  # 筛选出的miRNA序号
    miRNA_list_up = []  # 筛选出的miRNA_up组序号
    miRNA_list_down = []  # 筛选出的miRNA_down组序号
    miRNA_cout = []  # 筛选出的miRNA对应的差异表达值
    miRNA_cout_up = []  # 筛选出的miRNA_up对应的差异表达值
    miRNA_cout_down = []  # 筛选出的miRNA_down对应的差异表达值

    # 设置阈值进行筛选（同时也进行了统计检验）
    for i in range(bh.shape[0]):
        if bh_mean.iloc[i] != 0:
            x = us_mean.iloc[i] / bh_mean.iloc[i]
        else:
            x = 0
        cout.append(x)

    for i in range(bh.shape[0]):
        # p = t_test(us.iloc[i, :], bh.iloc[i, :])
        if cout[i] > 2:
            miRNA_list_up.append(i)
            miRNA_cout_up.append(cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(cout[i])
        if cout[i] < 0.5 and cout[i] > 0:
            miRNA_list_down.append(i)
            miRNA_cout_down.append(-1 / cout[i])
            miRNA_list.append(i)
            miRNA_cout.append(-1 / cout[i])

    miRNA_names = []  # 筛选出来的miRNA名称
    miRNA_names_up = []
    miRNA_names_down = []

    for i in range(len(miRNA_list)):
        miRNA_names.append(bh.index[miRNA_list[i]])
    for i in range(len(miRNA_list_up)):
        miRNA_names_up.append(bh.index[miRNA_list_up[i]])
    for i in range(len(miRNA_list_down)):
        miRNA_names_down.append(bh.index[miRNA_list_down[i]])

    # 将得到的分数值进行绘图
    plt.figure()
    plt.bar(miRNA_names, miRNA_cout)
    plt.xticks(rotation=60)
    plt.xlabel('miRNA', fontsize=2)

    plt.savefig('bar.png', dpi=1600)
    plt.show()

    return miRNA_names, miRNA_names_up, miRNA_names_down