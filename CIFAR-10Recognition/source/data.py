import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os

fileNameList=os.listdir("../data/train/")
size=0
for fileName in fileNameList:
    if(os.path.isfile(path=os.path.join("../data/train/",fileName))):
        size+=1
print(size)


#print(len(fileList))
#print(fileList)

'''
def pics_to_csv(path,isTrain=False):
    pass

def pics_to_TFRecord(path,isTrain=False):
    pass
'''



