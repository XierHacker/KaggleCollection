import numpy as np
import pandas as pd
import RMSLE
import data
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt

'''

a=np.array([1,2,3,4])
b=a*a
print(a)
print(b)
print(a.sum())

error=RMSLE.loss(a,b)
print(error)
'''
train_frame,test_frame,macro_frame=data.loadFrame()
print(macro_frame.shape)
print(macro_frame.index)

print(type(macro_frame.index[0]))

#convert str index to DetetimeIndex
new_index=pd.to_datetime(macro_frame.index)
print(new_index)

macro_frame=macro_frame.reindex(new_index)
print(macro_frame.index)

print(type(macro_frame.index[0]))

print(type(macro_frame))

macro_frame["apartment_build"].plot()

plt.show()




