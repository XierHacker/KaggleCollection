'''
use this file to do some test
'''
from __future__ import print_function,division
import data
import numpy as np
import tensorflow as tf
samples,labels=data.shuffle()
print ("shape of samples:",samples.shape)
print ("shape of labels:",labels.shape)

data.showpic(samples,labels,10)







