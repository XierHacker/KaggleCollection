# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:16:37 2016

@author: xierhacker
"""

from __future__ import print_function,division
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt


'''
#print the infomation of the data
#you can remove the comment to run this code

#load data from .mat file,return a dict
train_dict=loadmat("../data/train_32x32")
print("type of result:",type(train_dict))
print("keys:",train_dict.keys())

#information of X
print("shape of X:",train_dict['X'].shape)
print("type of X:",type(train_dict['X']))
print("type of X's element:",train_dict['X'].dtype)

#information of y
print("shape of y:",train_dict['y'].shape)
print("type of y:",type(train_dict['y']))
print("type of y's element:",train_dict['y'].dtype)
'''

#load data and return samples and labels
def load(filename):
    temp_dict=loadmat(filename)
    return temp_dict['X'],temp_dict['y']


#change the shape and content of data to easy to use
def transformat(samples,labels):
    #samples:(32, 32, 3, 73257)->(73257,32, 32, 3)
    temp_samples=np.transpose(samples,(3,0,1,2))
    #labels:[1]->[0,1,0,0,0,0,0,0,0,0]
    temp_labels=np.zeros((labels.shape[0],10))
    for row in range(labels.shape[0]):
        temp_labels[row,(labels[row]%10)]=1.0
    return temp_samples.astype(np.float32),temp_labels.astype(np.float32)
    

#normolize
def normalize(samples):
    #gray((R+G+B)/3)
    samples=np.add.reduce(samples,axis=3,keepdims=True)/3.0
    
    #0~255->-1.0~1.0
    return samples/128-1.0
    

def distribution(labels,name):
    pass
#show the i'th picture
def showpic(samples,labels,i):
    print(i,"'th label:\n",labels[i])
    plt.imshow(samples[i])
    plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    