import numpy as np
import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


train_clean_frame,y=data.loadTrainData()

#print(train_clean_frame.shape)
#print(y.shape)
#print(train_clean_frame)
#print(y)