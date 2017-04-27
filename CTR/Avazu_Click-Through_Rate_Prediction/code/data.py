import numpy as np
import pandas as pd

train_frame=pd.read_csv("../data/train_small.csv",index_col=0)
#test_frame=pd.read_csv("../data/train_small.csv",index_col=0)

#print(train_frame.shape)
#print(train_frame.head())
#print(train_frame.describe())


#DataFiled
'''
id: ad identifier
click: 0/1 for non-click/click
hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
C1 -- anonymized categorical variable
banner_pos
site_id
site_domain
site_category
app_id
app_domain
app_category
device_id
device_ip
device_model
device_type
device_conn_type
C14-C21 -- anonymized categorical variables
'''

#some freatures are useless and we drop it
def loadTrainData():
    new_df=train_frame.drop(labels=["site_id","app_id","device_id","device_ip",
                                    "site_domain","site_category","app_domain","app_category",
                                    "device_model"],axis=1)
    new_dummy_df=pd.get_dummies(data=new_df)
    clicked=new_df.pop("click")
    return new_dummy_df,clicked


