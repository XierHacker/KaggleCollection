import numpy as np
import pandas as pd

train_frame=pd.read_csv("../data/train.csv")
print(train_frame.shape)

test_frame=pd.read_csv("../data/test.csv")
print(test_frame.shape)

macro_frame=pd.read_csv("../data/macro.csv")
print(macro_frame.shape)