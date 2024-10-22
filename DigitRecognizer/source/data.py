'''
this file is used to load data and do some basic transform opration.
if you want see more about the data.you can open the dataExplore.ipynb file in this folder.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def loadFrame():
    train_frame=pd.read_csv("../data/train.csv")
    test_frame=pd.read_csv("../data/test.csv")
    #pop the labels and one-hot coding
    train_labels_frame=train_frame.pop("label")
    train_labels_frame=pd.get_dummies(data=train_labels_frame)

    return train_frame,train_labels_frame,test_frame


def vecToPic(dataSet):
    dataSet=np.reshape(dataSet,newshape=(-1,28,28,1))
    return dataSet

#show
def showpic(dataSet,i,labels=None):
    img=dataSet[i][:,:,0]
    plt.imshow(img)
    if(labels!=None):
        print("label:",labels[i])

    plt.show()

