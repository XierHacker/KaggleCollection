import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#get the ndarray value
X_train=data.train_frame_dummy.values
y_train=data.y_train
X_test=data.test_frame_dummy.values


#print(X_train.shape)
#print(X_test.shape)

#variable selection(cross validation)
#RF regression
max_features=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
scores=[]
for max_feature in max_features:
    RF=RandomForestRegressor(n_estimators=200,max_features=max_feature)
    score=np.sqrt(-1*cross_val_score(estimator=RF,X=X_train,y=y_train,scoring="neg_mean_squared_error",cv=10))
    scores.append(np.mean(score))

print(scores)
#we can use max_features=0.3
plt.plot(max_features,scores)
plt.show()