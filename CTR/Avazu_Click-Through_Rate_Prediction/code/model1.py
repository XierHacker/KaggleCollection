import numpy as np
import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss


train_clean_frame,y_frame=data.loadTrainData()
dataSet=train_clean_frame.values
labels=y_frame.values

print(dataSet.shape)
print(labels.shape)

#example1
#split dataSet
train_set,test_set,train_labels,test_labels=train_test_split(dataSet,labels,test_size=0.2)
print(train_set.shape)
print(test_set.shape)
print(train_labels.shape)
print(test_labels.shape)

#model train
LR=LogisticRegression(n_jobs=-1)
LR.fit(X=train_set,y=train_labels)

#model predict
result=LR.predict(X=test_set)
prob=LR.predict_proba(X=test_set)

#print("result:\n",result)
#print("ground truth:\n",test_labels)

acc=accuracy_score(y_true=test_labels,y_pred=result)
print(acc)

#logloss
LogLoss=log_loss(y_true=test_labels,y_pred=prob)
print(LogLoss)