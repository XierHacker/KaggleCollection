import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

#get the ndarray value
X_train=data.train_frame_dummy.values
y_train=data.y_train
X_test=data.test_frame_dummy.values


#based on model.py
ridge=Ridge(alpha=15)
scores=[]
nums=[1,10,15,20,25,30,35,40]
for num in nums:
    bagging_regressor=BaggingRegressor(base_estimator=ridge,n_estimators=num)
    score=np.sqrt(-1*cross_val_score(estimator=bagging_regressor,X=X_train,y=y_train,scoring="neg_mean_squared_error",cv=10))
    scores.append(np.mean(score))

#we get 10 is the best choise
plt.plot(nums,scores)
plt.show()