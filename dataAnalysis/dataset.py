import pandas as pd

def getData(file_bh, file_us):
    """bh(正常人员)us(病人)"""
    bh_data = pd.read_table(file_bh, header=0, index_col=0)
    us_data = pd.read_table(file_us, header=0, index_col=0)
    return bh_data, us_data

class Data_miRna():
    def __init__(self, bh, us):
        self.bh = bh
        self.us = us
        self.bh_mean = bh.mean(axis=1)
        self.bh_var = bh.var(axis=1)
        self.us_mean = us.mean(axis=1)
        self.us_var = us.var(axis=1)

    def getY(self):
        y = []
        for i in range(self.bh.shape[1]):
            y.append(1)
        for j in range(self.us.shape[1]):
            y.append(0)
        return y

def load_data(file1, file2):
    if "csv" in file1:
        data1 = pd.read_csv(file1, header=0, index_col=0)
    else:
        data1 = pd.read_table(file1, header=0, index_col=0)
    if "csv" in file2:
        data2 = pd.read_csv(file2, header=0, index_col=0)
    else:
        data2 = pd.read_table(file2, header=0, index_col=0)
    data = Data_miRna(data1, data2)
    return data


