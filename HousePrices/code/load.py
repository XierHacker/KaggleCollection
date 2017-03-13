'''
this file is used to load data
'''
from __future__ import print_function,division
import numpy as np
import pandas as pd

#read data from .csv file
train_df=pd.read_csv(filepath_or_buffer="../data/train.csv",index_col=0)
#test_df=pd.read_csv(filepath_or_buffer="../data/test.csv",index_col=0)

#show 5 rows
print (train_df.head())


