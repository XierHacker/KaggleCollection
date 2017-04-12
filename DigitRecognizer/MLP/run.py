from __future__ import print_function,division
import numpy as np
#import tensorflow as tf
import data
import mlp


print ("TEST:")
net=mlp.Network(120,120)
net.train(10,300,0.000001)

