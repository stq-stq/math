#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 数据导入
data = pd.read_excel('D:\desktop\data2.xlsx')
df=data.iloc[:,1:9]
# 处理数据
df.iloc[:,5]=df.iloc[:,5]*-1

# 数据标准化
def MaxMinNormalization(x):
    x=(x-np.min(x))/(np.max(x)-np.min(x))
    return x
df=MaxMinNormalization(df)
print(df)

# 协方差矩阵
covX = np.around(np.corrcoef(df.T),decimals=3)
# 替换nan
def replace_nan_with_zero(arr):
    arr = np.nan_to_num(arr)
    return arr
covX = replace_nan_with_zero(covX)
print(covX)

featValue, featVec=  np.linalg.eig(covX.T)  #求解系数相关矩阵的特征值和特征向量
print(featValue, featVec)

featValue = sorted(featValue)[::-1]
print(featValue)

# 同样的数据绘制散点图和折线图
plt.scatter(range(1, df.shape[1] + 1), featValue)
plt.plot(range(1, df.shape[1] + 1), featValue)

plt.title("Scree Plot")  
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")

plt.grid()  # 显示网格
plt.show()  # 显示图形

# 求特征值的贡献度
gx = featValue/np.sum(featValue)
print(gx)

# 求特征值的累计贡献度
lg = np.cumsum(gx)
print(lg)

#选出主成分
k=[i for i in range(len(lg)) if lg[i]<0.85]
k = list(k)
print(k)

# 选出前k个主成分的特征向量
selectVec = np.matrix(featVec.T[k]).T
selectVe=selectVec*(-1)
print(selectVec)

# 得分
finalData = np.dot(df,selectVec)
finalData = pd.DataFrame(finalData)
finalData.to_csv('finalData.csv',index=False)
print(finalData)

y1=pd.read_csv('finalData.csv')
y1=MaxMinNormalization(y1)
# 绘制主成分得分图
plt.plot(data.iloc[:,10],y1)

# 添加标题和坐标轴标签
plt.title('Scatter Plot of Final Data')
plt.xlabel('PCA')
plt.ylabel('date')

# 显示图形
plt.show()

# 绘制折线图
a=pd.read_csv('finalData.csv')
y1=0.7*a.iloc[:,0]+0.3*a.iloc[:,1]
y1=MaxMinNormalization(y1)
data0=pd.read_excel('D:\desktop\data0.xlsx')
y=data0.iloc[:,5]
x=data0.iloc[:,1]
y=MaxMinNormalization(y)
# 绘制第一个折线图
plt.plot(x,y1, label='ISI')

# 绘制第二个折线图
plt.plot(x, y, label='SHSZ Composite Index')

# 添加标题和坐标轴标签
plt.title('Two Lines on One Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 显示图例
plt.legend()

# 显示图形
plt.show()