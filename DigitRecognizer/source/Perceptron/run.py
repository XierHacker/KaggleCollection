import numpy as np
import tensorflow as tf
import pandas as pd
from perceptron import Perceptron


train_frame=pd.read_csv("../../data/train.csv")
test_frame=pd.read_csv("../../data/test.csv")

#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")
#trans format
train_frame=train_frame.astype(np.float32)
test_frame=test_frame.astype(np.float32)

#load model
percept=Perceptron()
percept.fit(X=train_frame.values,y=train_labels_frame.values)

#predict
pred=percept.predict(X=test_frame.values)


#write to .csv file
data={"ImageId":range(1,test_frame.shape[0]+1),"Label":pred}
result=pd.DataFrame(data=data)
result.to_csv(path_or_buf="submission.csv",index=False)
