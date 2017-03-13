# -*- coding: utf-8 -*-
from __future__ import print_function,division
#from pandas import Series,DataFrame
import pandas as pd

#传入data却不传入索引列表，那么自动创建0～N-1的索引
S=pd.Series(data=[1,2,3,4])
print ("S:\n",S)

#传入了data和索引列表
print ()
S2=pd.Series(data=[4,3,2,1],index=["a","b","c","d"])
print ("S2:\n",S2)
print (S2.index)
#通过索引的方式来访问一个或者一列值（很像字典的访问）
print (S2['c'])
print (S2[['a','b','c']])

#通过字典创建（上面还说了很像一个字典）
print ()
dict={"leo":24,"kate":23,"mat":11}
S3=pd.Series(data=dict)
print ("S3:\n",S3)

#即使是传入一个字典，还是可以传入一个索引的，
# 要是索引和字典中的相同，那么就会并进去
# 要是不相同，那么找不到值，相应的value就会被设为NaN
print ()
idx=["leo","kate","pig","cat"]
S4=pd.Series(data=dict,index=idx)
print ("S4:\n",S4)