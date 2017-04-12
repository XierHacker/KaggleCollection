from __future__ import print_function,division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#load data
train_data=pd.read_csv("../data/train.csv").values
train_data_size=len(train_data)

print (train_data_size)
print ("element type of train_data:",train_data.dtype)
print ("shape of train_data:",train_data.shape)
print ("train_data[0]\n",train_data[1])
#test_samples=pd.read_csv("../data/test.csv").values
'''
#shuffle and transformat
def shuffle():
    np.random.shuffle(train_data)
    train_samples = train_data[:, 1:].astype(np.float32)
    train_samples=np.reshape(train_samples,newshape=(-1,28,28,1)) #change shape to [batch, in_height, in_width, in_channels]

    temp_train_labels = train_data[:, 0]  # train_labels,use 0'th column
    # data transformat(vectorize labels)
    # onehot encoding
    train_labels = np.zeros(shape=(temp_train_labels.shape[0], 10)).astype(np.float32)
    for row in range(train_labels.shape[0]):
        train_labels[row, temp_train_labels[row]] = 1.0

    return train_samples,train_labels
'''



'''
#test

print ("element type of train_samples:",train_samples.dtype)
print ("element type of train_labels:",train_labels.dtype)
print ("the shape of test_samples:",test_samples.shape)

print ("shape of train_labels:",train_labels.shape)
print ("shape of train_samples:",train_samples.shape)
'''

#show i'th pic
def showpic(samples,labels,i):
    print ("the i'th pic is:",labels[i])
    plt.imshow(samples[i])
    plt.show()

#showpic(train_samples,train_labels,1)
