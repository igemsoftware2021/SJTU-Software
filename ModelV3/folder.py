#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在两个文件夹下，比例改0.7这个值即可
import os
import random
import shutil
from shutil import copy2

trainfiles = os.listdir('/\\dataset\\dbnFiles')
num_train = len(trainfiles)
print( "num_train: " + str(num_train) )
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)
num = 0

trainDir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA\\TR0'
validDir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA\\TS0'

for i in index_list:
    fileName = os.path.join('/\\dataset\\dbnFiles', trainfiles[i])
    if num < num_train*0.9:
        print(str(fileName))
        copy2(fileName, trainDir)
    else:
        copy2(fileName, validDir)
    num += 1