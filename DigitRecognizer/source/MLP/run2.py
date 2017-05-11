'''
    this code is copy from my project.
     GitHhub:https://github.com/XierHacker/DeepModels/tree/master/MLP
'''
import numpy as np
import tensorflow as tf
import pandas as pd
from mlp2 import MLP


train_frame=pd.read_csv("../../data/train.csv")
test_frame=pd.read_csv("../../data/test.csv")

#pop the labels and one-hot coding
train_labels_frame=train_frame.pop("label")
#trans format
train_frame=train_frame.astype(np.float32)
test_frame=test_frame.astype(np.float32)

#load model
mlp_model=MLP(300,100)
mlp_model.fit(X=train_frame.values,y=train_labels_frame.values)

#predict
pred=mlp_model.predict(X=test_frame.values)


#write to .csv file
data={"ImageId":range(1,test_frame.shape[0]+1),"Label":pred}
result=pd.DataFrame(data=data)
result.to_csv(path_or_buf="submission.csv",index=False)