import numpy as np
import pandas as pd

#load DataFrame from .csv files
def loadFrame():
    train_frame=pd.read_csv("../data/train.csv")
    test_frame = pd.read_csv("../data/test.csv")
    macro_frame = pd.read_csv("../data/macro.csv",index_col="timestamp")

    #in macro_frame trans index str type to datetime type
    new_index = pd.to_datetime(macro_frame.index)
    macro_frame = macro_frame.reindex(new_index)

    return train_frame,test_frame,macro_frame

