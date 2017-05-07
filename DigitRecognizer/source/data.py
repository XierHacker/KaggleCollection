'''
this file is used to load data and do some basic transform opration.
if you want see more about the data.you can open the dataExplore.ipynb file in this folder.
'''

import pandas as pd
import numpy as np

def loadFrame():
    train_frame=pd.read_csv("../data/train.csv")
    test_frame=pd.read_csv("../data/test.csv")
    #pop the labels and one-hot coding
    train_labels_frame=train_frame.pop("label")
    train_labels_frame=pd.get_dummies(data=train_labels_frame)

    return train_frame,train_labels_frame,test_frame


