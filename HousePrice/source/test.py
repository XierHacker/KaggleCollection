import numpy as np
import pandas as pd

df=pd.DataFrame(data=[[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],
                index=["a","b","c","d"],
                columns=["one","two"])
print(df.columns)
print("df:")
print(df)

#直接使用sum()方法,返回一个列求和的Series,自动跳过NaN值
print("df.mean()")
print(df.mean())

#当轴为1.就会按行求和
print("df.mean(axis=1)")
print(df.mean(axis=1))

#选择skipna=False可以禁用跳过Nan值
print("df.mean(axis=1,skipna=False):")
print(df.mean(axis=1,skipna=False))


'''
#等长列表组成的字典
data={
        "name":["leo","tom","kate","pig"],
        "age":[10,20,30,40],
        "weight":[50,50,40,200]
}

frame=pd.DataFrame(data=data)
print("frame:")
print(frame)

#指定列顺序columns
frame2=pd.DataFrame(data=data,columns=["name","weight","age"])
print("frame2:")
print(frame2)

#指定index
frame3=pd.DataFrame(data=data,columns=["name","weight","age","height"],index=["one","two","three","four"])
print("frame3:")
print(frame3)


#索引一列
print("name:\n",frame3["name"])
print("weight:\n",frame3.weight)

#改变一列的值
frame3["height"]=100
print("frame3")
print(frame3)

print(frame3.index)

#对于一个Series来说,行数保持不变,列数变为不同类的个数
#但是每一行还是以编码的形式表示原来的类别
#这个函数返回是一个DataFrame,其中列名为各种类别
s = pd.Series(list('abca'))
print("original:")
print(s)
print("get dummy:")
s_dummy=pd.get_dummies(data=s)
print(s_dummy)
print("type of s_dummy:",type(s_dummy))


df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
                   'C': [1, 2, 3]})
print("original:")
print(df)

#其中只要是类别相关的,都会被hot-encoding
#每一个特征(原始形式的列名)下面有几种不同的类别,就会生成几列(比如A下面只有a和b两种形式,就会生成A_a和A_b两列)
#原始为数字的那些特征,保持不变
#prefix表示你对于新生成的那些列想要的前缀,你可以自己命名
df_dummy=pd.get_dummies(data=df,prefix=["A","B"])
print("get dummy:")
print(df_dummy)




s=pd.Series(data=["tom","jack","kate",np.nan])
print(s)

print(s.isnull())
print(type(s.isnull()))

df = pd.DataFrame({'A': ['a', 'b', np.nan], 'B': ['b', 'a', 'c'],
                   'C': [1, 2, np.nan]})
print("original:")
print(df)
print(df.isnull().sum())

'''


