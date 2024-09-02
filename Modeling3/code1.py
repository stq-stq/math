
# #!/usr/bin/env python
# coding: utf-8
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from math import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_excel('D:\desktop\Modeling\data2.xlsx')
# 画出第二列到第四列两两之间的散点图
# 中文显示问题，需要设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
sns.pairplot(df1.iloc[:,4:7], diag_kind='kde', plot_kws={'s': 5})
plt.show()
# 求出相关系数矩阵
corr_matrix = df1.iloc[:,4:7].corr()
print(corr_matrix)


# 建立最后一列分别和第一列第二列的拟合模型,并画出拟合曲线
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df1=pd.read_excel('D:\desktop\Modeling\data2.xlsx')
df1=df1.iloc[:,4:7]
# 划分训练集和测试集
train_size = int(len(df1) * 0.8)
train_data, test_data = df1.iloc[:train_size, 0:1], df1.iloc[train_size:, 0:1]
train_target, test_target = df1.iloc[:train_size, 1:2], df1.iloc[train_size:, 1:2]

# 建立线性回归模型
model = LinearRegression()
model.fit(train_data, train_target)

# 预测测试集数据
test_predict = model.predict(test_data)

# 计算均方误差和决定系数
mse = mean_squared_error(test_target, test_predict)
r2 = r2_score(test_target, test_predict)
# 绘制标准化残差图
residual = test_target - test_predict
sns.residplot(test_predict, residual, lowess=True, color='g')
plt.xlabel('Predicted Value')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.show()
plt.figure()
# 画出拟合曲线
plt.scatter(train_data, train_target, label='Train Data')
plt.scatter(test_data, test_target, label='Test Data')
plt.plot(test_data, test_predict, label='Fitted Line')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 定义要拟合的函数
def func(t, a, b):
    return a + b*np.log(t)
t_data = df1.iloc[:,1]
y_data = df1.iloc[:,2]
y_data = np.log(y_data)
# 进行拟合
popt, pcov = curve_fit(func, t_data, y_data)

# 打印拟合得到的参数
print("拟合得到的参数：")
print("a =", popt[0])
print("b =", popt[1])
# print("c =", popt[2])

# 计算拟合的R-squared
y_fit = func(t_data, *popt)
ssr = np.sum((y_fit - y_data) ** 2)
sst = np.sum((y_data - np.mean(y_data)) ** 2)
r2 = 1 - ssr / sst
print("拟合的R-squared值为：", r2)

# 绘制拟合曲线
plt.scatter(t_data, y_data, label='Data', color='g')
plt.plot(t_data, func(t_data, *popt), label='Fitted Line')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

#计算每年的总水流量和总排沙量
data=pd.read_excel('D:\desktop\Modeling\data2.xlsx')
col1_name = data.columns[5]
col2_name = data.columns[6] 
yearly_data=data.groupby(data.iloc[:,0]).agg({col1_name:'sum',col2_name:'sum'})
yearly_data.to_excel('yearly_data.xlsx',index=True)