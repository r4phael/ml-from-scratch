import pandas as pd
#import numpy as np
import scipy.stats as sci
import random

dev46 = pd.read_csv('/home/r4ph/PycharmProjects/ml-from-scratch/csv/LM/LM - 46.csv')

df_false = (dev46.loc[dev46['Smell'] == False])
df_false = df_false.drop(columns = 'Smell')

df_true = (dev46.loc[dev46['Smell'] == True])
df_true = df_true.drop(columns = 'Smell')

l_true = []
for columns in df_true:
    l_true += (df_true[columns].tolist())

l_false = []
for columns in df_false:
    l_false += (df_false[columns].tolist())

random.shuffle(l_true)

# While my list is not empty
for i in range((len(l_true) - len(l_false))):
    l_true.pop(i)

l_false = sci.zscore(l_false)
l_true = sci.zscore(l_true)

wil = sci.wilcoxon(l_true, l_false, zero_method='wilcox')

print(wil)
