import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

#get the ndarray value
X_train=data.train_frame_dummy.values
y_train=data.y_train
X_test=data.test_frame_dummy.values


#based on model.py
ridge=Ridge(alpha=15)
scores=[]
depths=[1,2,3,4,5,6,7,8]
for depth in depths:
    xgb=XGBRegressor(max_depth=depth)
    score=np.sqrt(-1*cross_val_score(estimator=xgb,X=X_train,y=y_train,scoring="neg_mean_squared_error",cv=10))
    scores.append(np.mean(score))

#we get depth=5 is the best choise
plt.plot(depths,scores)
plt.show()