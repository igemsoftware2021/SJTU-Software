import pandas as pd
data = pd.read_table("bladder_1.txt", header=0, index_col=0)
data_bh = data.loc[:, 'GSM3107239': 'GSM3107347']
data_us = data.loc[:, 'GSM3106847': 'GSM3107238']
data_bh.to_csv("bladder_bh_1.txt", sep='\t')
data_us.to_csv("bladder_us_1.txt", sep='\t')
print(data_bh)
print(data_us)