import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


X=load_boston().data
#print(X.shape)

y=load_boston().target
#print(y.shape)

#ridge regression
alphas=np.logspace(start=-3,stop=2,num=50)
score_Ridge=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    test_score=np.sqrt(-1*cross_val_score(estimator=ridge,X=X,y=y,scoring='neg_mean_squared_error',cv=10))
    score_Ridge.append(np.mean(test_score))


#random forest
max_features=[0.1,0.3,0.5,0.7,0.9,0.99]
score_RF=[]
for max_feature in max_features:
    RF=RandomForestRegressor(n_estimators=200,max_features=max_feature)
    test_score=np.sqrt(-1*cross_val_score(estimator=RF,X=X,y=y,scoring='neg_mean_squared_error',cv=10))
    score_RF.append(np.mean(test_score))

ax1=plt.subplot(2,1,1)
ax2=plt.subplot(2,1,2)

ax1.plot(alphas,score_Ridge,label="Alpha vs CV Error")
ax1.set_xlabel("Alpha")
ax1.set_ylabel("CV Error")

#ax1.title("")


ax2.plot(max_features,score_RF,label="Max_Features vs CV Error")
ax2.set_xlabel("Max_Features")
ax2.set_ylabel("CV Error")

plt.show()