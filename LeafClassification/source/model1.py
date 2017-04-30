import data
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

#load data
train_frame,train_labels_frame,test_frame=data.loadFrame()

train_dataSet=train_frame.values
train_labels=train_labels_frame.values
test_dataSet=test_frame.values

print(train_dataSet.shape)
print(train_labels.shape)
print(test_dataSet.shape)

#model
RF=RandomForestClassifier(n_estimators=20)
RF.fit(X=train_dataSet,y=train_labels)

#predict
result=RF.predict(X=test_dataSet)
prob=RF.predict_proba(X=test_dataSet)
print(result)
print(result.shape)
print(prob.shape)

#classes
print(RF.classes_.shape)

#losloss
loss=log_loss(y_true=result,y_pred=prob)
print(loss)

#