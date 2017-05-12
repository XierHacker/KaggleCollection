from __future__ import print_function,division
import numpy as np
#import tensorflow as tf
import data
import convnet


print ("TEST:")
net=convnet.Network()
net.train(30,300,0.001)

