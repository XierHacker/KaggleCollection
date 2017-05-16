import numpy as np
import tensorflow as tf
import pandas as pd
from lenet5 import LeNet5


train_frame=pd.read_csv("../../data/train.csv")
test_frame=pd.read_csv("../../data/test.csv")

#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")
#trans format
train_frame=train_frame.astype(np.float32)
test_frame=test_frame.astype(np.float32)


#get values from DataFrame
trainSet=train_frame.values
testSet=test_frame.values
trainSet=np.reshape(a=trainSet,newshape=(-1,28,28,1))
testSet=np.reshape(a=testSet,newshape=(-1,28,28,1))
y=train_labels_frame.values




#load model
lenet=LeNet5()
lenet.fit(X=trainSet,y=y,epochs=25)

#predict
pred=lenet.predict(X=testSet)




#write to .csv file
data={"ImageId":range(1,test_frame.shape[0]+1),"Label":pred}
result=pd.DataFrame(data=data)
result.to_csv(path_or_buf="submission.csv",index=False)