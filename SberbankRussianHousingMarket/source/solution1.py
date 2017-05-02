'''
this is a simple solution of this problem,we don't take the economy status
into account.
so, we just care about train.csv and test.csv give to us

code is based on the solution1.ipynb
    
'''

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
import data
import RMSLE

train_frame,train_labels_frame,test_frame,macro_frame=data.loadFrame()

#merge the train_frame and test_frame
all_frame=pd.concat((train_frame,test_frame),axis=0)

#timestamp is useless in this solution
all_frame.pop("timestamp")

#one-hot encoding
all_frame_dummy=pd.get_dummies(data=all_frame)


#deal with missing value
mean_cols=all_frame_dummy.mean()
filled_all_frame_dummy=all_frame_dummy.fillna(value=mean_cols)

#recover the trainSet and testSet
train_frame_dummy=filled_all_frame_dummy.loc[train_frame.index]
test_frame_dummy=filled_all_frame_dummy.loc[test_frame.index]


#extract data from DataFrame
trainData=train_frame_dummy.values
trainLabels=train_labels_frame.values
testData=test_frame_dummy.values

'''
#split trainSet
trainSet,validationSet,train_y,validation_y=train_test_split(trainData,trainLabels,test_size=0.2)
print(trainSet.shape,validationSet.shape,train_y.shape,validation_y.shape)

#build and fit model
RF=RandomForestRegressor(n_estimators=20)
BR=BaggingRegressor(base_estimator=RF,n_estimators=10)
BR.fit(X=trainSet,y=train_y)

#predict
pred=BR.predict(X=validationSet)
print(pred.shape)
print(pred)
print(validation_y)

print(RMSLE.loss(validation_y,pred))
'''

#build and fit model
RF=RandomForestRegressor(n_estimators=20)
BR=BaggingRegressor(base_estimator=RF,n_estimators=10)
BR.fit(X=trainData,y=trainLabels)

#predict
pred=BR.predict(X=testData)

result={"id":test_frame.index,"price_doc":pred}
submit_frame=pd.DataFrame(data=result)

print(submit_frame)
submit_frame.to_csv(path_or_buf="answer1.csv",index=False)

