import numpy as np
import pandas as pd
import data
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


#get the ndarray value
X_train=data.train_frame_dummy.values
y_train=data.y_train
X_test=data.test_frame_dummy.values




ridge=Ridge(alpha=10)
RF=RandomForestRegressor(n_estimators=200,max_features=0.3)

#fit the model
ridge.fit(X=X_train,y=y_train)
RF.fit(X=X_train,y=y_train)

#predict
ridge_predict=np.expm1(ridge.predict(X_test))
RF_predict=np.expm1(RF.predict(X_test))

#print(ridge_predict.shape)
#print(RF_predict.shape)

#final result
result=(ridge_predict+RF_predict)/2

#submit
content={"Id":data.test_frame.index,"SalePrice":result}
submission_df=pd.DataFrame(data=content)
print(submission_df)
submission_df.to_csv(path_or_buf="submission.csv",index=False)

