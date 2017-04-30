import numpy as np
a=np.array([1,2,3,4])
a=np.reshape(a,newshape=(4,1))
print(a.shape)

b=np.zeros(shape=(4,4))
print(b)

result=np.concatenate([a,b],axis=1)
print(result)