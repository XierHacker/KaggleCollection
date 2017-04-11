import numpy as np
import pandas as pd

train_frame=pd.read_csv("../data/train_small.csv",index_col=0)
test_frame=pd.read_csv("../data/train_small.csv",index_col=0)

print(train_frame.shape)
print(train_frame.head())
print(train_frame.describe())