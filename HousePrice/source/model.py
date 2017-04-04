import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

#get the ndarray value
X_train=data.train_frame_dummy.values
y_train=data.y_train
X_test=data.test_frame_dummy.values


#print(X_train.shape)
#print(X_test.shape)

#variable selection(cross validation)
#Ridge regression
alphas=np.logspace(start=-3,stop=2,num=50)
scores=[]
for alpha in alphas:
    ridge=Ridge(alpha=alpha)
    score=np.sqrt(-1*cross_val_score(estimator=ridge,X=X_train,y=y_train,scoring="neg_mean_squared_error",cv=10))
    scores.append(np.mean(score))

print(scores)
#we can use alpha=10
plt.plot(alphas,scores)
plt.show()

