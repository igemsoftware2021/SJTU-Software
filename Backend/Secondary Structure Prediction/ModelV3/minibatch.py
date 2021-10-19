import os
import re
from shutil import copy2
from matplotlib import pyplot as plt

trainDir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA\\TR0'
testDir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA\\TS0'


train100Dir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA100\\TR0'
test100Dir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA100\\TS0'

train100ADJDir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA100_ADJ\\TR0'
test100ADJDir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA100_ADJ\\TS0'

train200Dir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA200\\TR0'
test200Dir = 'C:\\Users\\18113\\Desktop\\iGem 2021\\ModelV3\\dataset\\bpRNA200\\TS0'

trainList = os.listdir(train100Dir)
testList = os.listdir(test100Dir)
trainAdjList = os.listdir(train100ADJDir)
testAdjList = os.listdir(test100ADJDir)

train_remove = []
test_remove = []

for j in trainAdjList:
    if trainList.__contains__(j):
        continue
    else:
        print('Repeated in Adj: '+ j)
        train_remove.append(j)

for j in testAdjList:
    if testList.__contains__(j):
        continue
    else:
        print('Repeated in Adj: ' + j)
        test_remove.append(j)

for i in train_remove:
    name = os.path.join(train100ADJDir, i)
    if os.path.isfile(name):
        print('Remove: ' + name)
        os.remove(name)
    else:
        continue

for i in test_remove:
    name = os.path.join(test100ADJDir, i)
    if os.path.isfile(name):
        print('Remove: ' + name)
        os.remove(name)
    else:
        continue

# lengths = []
# len100 = []
# len200 = []
# problem_bp = []
# vocab = ['R', 'Y', 'M', 'K', 'S', 'W', 'B', 'V', 'D', 'N', 'I', 'P', 'X']
# for i in trainList:
#     bp_path = os.path.join(train100Dir, i)
#     with open(bp_path) as f:
#         lines = f.readlines()
#         length = re.findall("\d+", lines[1])
#         lengths.append(int(length[0]))
#
#         for i in range(len(lines)):
#             if lines[i].find('#') != -1:
#                 continue
#             else:
#                 start_line = i
#                 break
#
#         seq = lines[start_line]
#         seq = seq.rstrip('\n')
#         seq = seq.upper()
#         result = lines[start_line + 1]
#         result = result.rstrip('\n')
#
#         if seq.find('A') == -1 or seq.find('U') == -1 or seq.find('C') == -1 or seq.find('G') == -1:
#             print(seq)
#             print(bp_path)
#             problem_bp.append(bp_path)
#
#         if seq.find('_') != -1 or seq.find('.') != -1:
#             print(seq)
#             print(bp_path)
#             problem_bp.append(bp_path)
#
#         for j in vocab:
#             if seq.find(j) != -1:
#                 print(seq)
#                 print(bp_path)
#                 problem_bp.append(bp_path)
#
#
# for i in testList:
#     bp_path = os.path.join(test100Dir, i)
#     with open(bp_path) as f:
#         lines = f.readlines()
#         length = re.findall("\d+", lines[1])
#         lengths.append(int(length[0]))
#
#         for i in range(len(lines)):
#             if lines[i].find('#') != -1:
#                 continue
#             else:
#                 start_line = i
#                 break
#
#         seq = lines[start_line]
#         seq = seq.rstrip('\n')
#         seq = seq.upper()
#         result = lines[start_line + 1]
#         result = result.rstrip('\n')
#
#         if seq.find('A') == -1 or seq.find('U') == -1 or seq.find('C') == -1 or seq.find('G') == -1:
#             print(seq)
#             print(bp_path)
#             problem_bp.append(bp_path)
#
#         if seq.find('_') != -1 or seq.find('.') != -1:
#             print(seq)
#             print(bp_path)
#             problem_bp.append(bp_path)
#
#         for j in vocab:
#             if seq.find(j) != -1:
#                 print(seq)
#                 print(bp_path)
#                 problem_bp.append(bp_path)



# for i in problem_bp:
#     if os.path.isfile(i):
#         print(i)
#         os.remove(i)
#     else:
#         continue

bins_interval=100

# margin 设定的左边和右边空留的大小

def probability_distribution(data, bins_interval=1, margin=1):

    bins = range(min(data), max(data) + bins_interval - 1, bins_interval)

    print(len(bins))

    for i in range(0, len(bins)):

        print(bins[i])

    plt.xlim(min(data) - margin, max(data) + margin)

    plt.title("probability-distribution")

    plt.xlabel('Interval')

    plt.ylabel('Probability')

    plt.hist(x=data, bins=bins, histtype='bar', color=['r'])

    plt.show()

# probability_distribution(lengths, bins_interval=bins_interval)