import numpy as np
import pandas as pd

#load DataFrame from .csv files
def loadFrame():
    train_frame=pd.read_csv("../data/train.csv")
    test_frame = pd.read_csv("../data/test.csv")
    macro_frame = pd.read_csv("../data/macro.csv")
    return train_frame,test_frame,macro_frame

