import pandas as pd
import numpy as np

def loadFrame():
    train_frame=pd.read_csv("../data/train.csv")
    test_frame=pd.read_csv("../data/test.csv")
    return train_frame,test_frame


