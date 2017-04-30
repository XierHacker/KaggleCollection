import pandas as pd
import numpy as np

#load Frame from .csv files
def loadFrame():
    train_frame=pd.read_csv("../data/train.csv",index_col="id")
    test_frame=pd.read_csv("../data/test.csv")
    train_labels_frame=train_frame.pop("species")
    return train_frame,train_labels_frame,test_frame

'''

train_frame=pd.read_csv("../data/train.csv")
print(train_frame.head())
print(train_frame.shape)

labels=train_frame.pop("species")
print(labels)
print(labels.shape)
print(train_frame.shape)


#load clean DataSet from Frame
def loadCleanDataSet():
    train_frame,test_frame=loadFrame()
    train_labels_frame=train_frame.pop("species")
'''


