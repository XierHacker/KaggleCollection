import pandas as pd
import numpy as np

train_frame=pd.read_csv("../data/train.csv")
print(train_frame.shape)
print(train_frame.head)
test_frame=pd.read_csv("../data/test.csv")
print(test_frame.shape)
