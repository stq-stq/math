#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# In[1]:

# 数据清洗
data = pd.read_excel('D:\desktop\data5.xlsx')
df=data.iloc[:,1:5]
# df.iloc[:,3]=df.iloc[:,3]*-1
df

# In[1]:
# In[1]:
# 数据标准化
def MaxMinNormalization(x):
    x=(x-np.min(x))/(np.max(x)-np.min(x))
    return x
df=MaxMinNormalization(df)
df
# In[1]:
# 协方差矩阵
covX = np.around(np.corrcoef(df.T),decimals=3)
# 替换nan
def replace_nan_with_zero(arr):
    arr = np.nan_to_num(arr)
    return arr
covX = replace_nan_with_zero(covX)
covX
# In[1]:
featValue, featVec=  np.linalg.eig(covX.T)  #求解系数相关矩阵的特征值和特征向量
featValue, featVec
# In[1]:
featValue = sorted(featValue)[::-1]
featValue
# In[1]:
# 同样的数据绘制散点图和折线图
plt.scatter(range(1, df.shape[1] + 1), featValue)
plt.plot(range(1, df.shape[1] + 1), featValue)

plt.title("Scree Plot")  
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")

plt.grid()  # 显示网格
plt.show()  # 显示图形
# In[1]:
# 求特征值的贡献度
gx = featValue/np.sum(featValue)
gx
# In[1]:
# 求特征值的累计贡献度
lg = np.cumsum(gx)
lg
# In[1]:
#选出主成分
k=[i for i in range(len(lg)) if lg[i]<0.85]
k = list(k)
print(k)
# In[1]:
# 选出前k个主成分的特征向量
selectVec = np.matrix(featVec.T[k]).T
selectVe=selectVec*(-1)
selectVec
# In[1]:
# 得分
finalData = np.dot(df,selectVec)
finalData = pd.DataFrame(finalData)
finalData.to_csv('finalData5.csv',index=False)
finalData
# In[1]:
y1=pd.read_csv('finalData5.csv')
y1=MaxMinNormalization(y1)
# print(y1)
# 绘制散点图
plt.plot(data.iloc[:,8],y1)

# 添加标题和坐标轴标签
plt.title('Scatter Plot of Final Data')
plt.xlabel('PC1')
plt.ylabel('date')

# 显示图形
plt.show()
# In[1]:
y1=pd.read_csv('finalData5.csv')
y1 = 0.7*y1.iloc[:,0]+0.3*y1.iloc[:,1]
y1.to_excel('score5.xlsx',index=False)
y1=MaxMinNormalization(y1)
data0=pd.read_excel('D:\desktop\data0.xlsx')
y=data0.iloc[:,5]
x=data0.iloc[:,1]
y=MaxMinNormalization(y)
# 绘制第一个折线图
plt.plot(x,y1, label='Line 1')

# 绘制第二个折线图
plt.plot(x, y, label='Line 2')

# 添加标题和坐标轴标签
plt.title('Two Lines on One Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 显示图例
plt.legend()

# 显示图形
plt.show()
# In[1]:
y1=pd.read_csv('finalData5.csv')
y1=MaxMinNormalization(y1)
data0=pd.read_excel('D:\desktop\data0.xlsx')
y=data0.iloc[:,5]
x=data0.iloc[:,1]
y=MaxMinNormalization(y)
# 绘制第一个折线图
plt.plot(x,y1, label='Line 1')

# 绘制第二个折线图
plt.plot(x, y, label='Line 2')

# 添加标题和坐标轴标签
plt.title('Two Lines on One Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 显示图例
plt.legend()

# 显示图形
plt.show()

# In[1]:
# 绘图
 
plt.figure(figsize = (14,14))
ax = sns.heatmap(selectVec, annot=True, cmap="BuPu")
 
# 设置y轴字体大小
ax.yaxis.set_tick_params(labelsize=15)
plt.title("Factor Analysis", fontsize="xx-large")
 
# 设置y轴标签
plt.ylabel("Sepal Width", fontsize="xx-large")
# 显示图片
plt.show()
 
# 保存图片
# plt.savefig("factorAnalysis", dpi=500)
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]: