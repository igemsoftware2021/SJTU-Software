import os

# path = '../data_new_v2/TR0'
# path2 = '../data_test/TR0'
#
# fasta_file = 'TR0.fasta'
#
# files = os.listdir(path)
#
# fp1 = open(fasta_file,'w')
#
# sum = 0
# excess = False
# for file in files:
#     pos = path + "/" + file
#     with open(pos,'r') as fp:
#         for i in range(3):
#             fp.readline()
#         seq = fp.readline()
#         if(len(seq)<=100):
#             fp1.write(seq)

str = 'UUUGGCGAUAAUAGCUUGUAGGAACCACCUGAUCCCAUUCCGAACUCAGAAGUGAAACUACAAUAGCGCCGAUGAUAGUCUGGCAAUGCCCAGUAA'
main = 'echo ' + str + ' | ./linearpartition -V -M'
a = os.popen(main).read()
print(a.split()[1])

