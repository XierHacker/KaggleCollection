import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_frame=pd.read_csv("../data/train.csv",index_col=0)
test_frame=pd.read_csv("../data/test.csv",index_col=0)

def processing(trainSet,testSet):
    return train_frame_processed,test_frame_processed




'''
#观察数据(前5行)
print(train_frame.head())
#索引一行的方法(因为在前面读取的部分使用的index_col=0)
#print(train_frame.ix[1])

#观察price的分布,其中log1p把price平滑化
#log1p过得最后记得用exp1m回复
price=pd.DataFrame(data={"price":train_frame["SalePrice"],
                         "log(price+1)":np.log1p(train_frame["SalePrice"])} )
price.hist()
plt.show()
'''

#弹出训练集的价格,并且平滑化
y_train=np.log1p(train_frame.pop("SalePrice"))
#print(y_train)

#查看形状
#print(train_frame)
#print(test_frame)

#合并测试集和训练集
all_frame=pd.concat((train_frame,test_frame),axis=0)
#print(all_frame)


#因为MSSubClass这个属性没有数学上的大小关系等含义,吧其中的元素变为字符串类型作为类别
#当然,其他是整形但是没有大小关系只表示类别的也都要转换为str类型.
#print(all_frame["MSSubClass"].dtype)
all_frame["MSSubClass"]=all_frame["MSSubClass"].astype(str)
print(all_frame["MSSubClass"].dtype)
#print(all_frame["MSSubClass"].value_counts())


#类别特征通过one-hot encoding转换为数值形式.
all_frame_dummy=pd.get_dummies(data=all_frame)
#print(all_frame_dummy.head())

#处理缺失值.
#查看缺失值及缺失值数量
#print(all_frame_dummy.isnull().sum().sort_values(ascending=False).head())

#采用平均数填充的方式来填充缺失值.
mean_cols=all_frame_dummy.mean()
#print(mean_cols.head())

#这个函数会把我传进去的Series各个平均值元素按照列名赋值给对应应的缺失值.
filled_all_frame_dummy=all_frame_dummy.fillna(value=mean_cols)
#print(filled_all_frame_dummy.isnull().sum().sum())

#标准化之前数字型的数据
#找到numerical的数据columns(列)索引
numeric_cols=all_frame.columns[all_frame.dtypes!="object"]
#print(numeric_cols)
#print(filled_all_frame_dummy.loc[:,numeric_cols])
numeric_cols_means=filled_all_frame_dummy.loc[:,numeric_cols].mean()
#print(numeric_cols_means)
numeric_cols_std=filled_all_frame_dummy.loc[:,numeric_cols].std()
#print(numeric_cols_std)
filled_all_frame_dummy.loc[:,numeric_cols]=(filled_all_frame_dummy.loc[:,numeric_cols]-numeric_cols_means)/numeric_cols_std
#print(filled_all_frame_dummy)

#数据集分回 训练/测试集
train_frame_dummy=filled_all_frame_dummy.loc[train_frame.index]
test_frame_dummy=filled_all_frame_dummy.loc[test_frame.index]

#print(train_frame_dummy.shape)
#print(test_frame_dummy.shape)