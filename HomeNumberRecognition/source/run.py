#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:56:36 2016

@author: xierhacker
"""

from __future__ import print_function,division
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
import loadData
#import dp

#load data
train_samples,train_labels=loadData.load("../data/train_32x32")
test_samples,test_labels=loadData.load("../data/test_32x32")

'''
#print the infomation of the data
#you can remove the comment to run this code
print("the shape of train_set:",train_samples.shape)
print("the shape of train_labels:",train_labels.shape[0])
print("type of samples element:",train_samples.dtype)
print("type of labels element:",train_labels.dtype)
print("1-20:\n",train_labels[1:20])
'''

#data transformat
train_samples,train_labels=loadData.transformat(train_samples,train_labels)
test_samples,test_labels=loadData.transformat(test_samples,test_labels)

'''
#print the infomation of the data
#you can remove the comment to run this code
print("the shape of train_set:",train_samples.shape)
print("the shape of train_labels:",train_labels.shape)
print("type of samples element:",trai n_samples.dtype)
print("type of labels element:",train_labels.dtype)
print("1-20:\n",train_labels[1:20])
'''

'''
#show i'th pic
loadData.showpic(train_samples,train_labels,1)
loadData.showpic(train_samples,train_labels,2)
loadData.showpic(train_samples,train_labels,3)
loadData.showpic(train_samples,train_labels,4)
'''

#normalized 
train_samples=loadData.normalize(train_samples)
test_samples=loadData.normalize(test_samples)

print("the shape of train_set:",train_samples.shape)
print("the shape of train_labels:",train_labels.shape)
print(train_samples[1:3])


'''
#show i'th pic
loadData.showpic(train_samples,train_labels,1)
loadData.showpic(train_samples,train_labels,2)
loadData.showpic(train_samples,train_labels,3)
loadData.showpic(train_samples,train_labels,4)
'''

#net work
#net=dp.Network(neurons_in_hidden=128,batch_size=100)
#net.define_graph()
#net.run()






















