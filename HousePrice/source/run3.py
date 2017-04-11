import numpy as np
import pandas as pd
import data
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


#get the ndarray value
X_train=data.train_frame_dummy.values
y_train=data.y_train
X_test=data.test_frame_dummy.values



#based on the results of model.py and model3.py
ridge=Ridge(alpha=15)
xgb_regressor=XGBRegressor(max_depth=5)


#fit the model
xgb_regressor.fit(X=X_train,y=y_train)


#predict
result=np.expm1(xgb_regressor.predict(X_test))

#submit
content={"Id":data.test_frame.index,"SalePrice":result}
submission_df=pd.DataFrame(data=content)
print(submission_df)
submission_df.to_csv(path_or_buf="submission3.csv",index=False)