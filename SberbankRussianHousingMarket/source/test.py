import numpy as np
import RMSLE
a=np.array([1,2,3,4])
b=a*a
print(a)
print(b)
print(a.sum())

error=RMSLE.loss(a,b)
print(error)