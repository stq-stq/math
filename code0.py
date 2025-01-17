# In[1]:
# #!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import norm
import seaborn as sns

# In[1]:
# 读取数据
data = pd.ExcelFile('D:\desktop\Modeling\附件1.xlsx')
data = pd.concat([data.parse('2016'), data.parse('2017'), data.parse('2018'), data.parse('2019'), data.parse('2020'), data.parse('2021')])
print(data)

# 查找缺失值
for col in range(data.shape[1]):
    columnData = data.iloc[:, col]
    missingIndices = columnData[columnData.isnull()].index

if len(missingIndices)!= 0:
    missing = '有'
    miss = 1
    print('第', col, '列缺失值的索引：', missingIndices)
else:
    missing = '无'

# data.to_excel('D:\desktop\Modeling\data.xlsx', index=False)

# In[1]:
# 处理时间格式
data=pd.read_excel('D:\desktop\Modeling\data.xlsx')
time_list=[]
for i in range(len(data)):
    m,d,h=str(int(data.iloc[i,1])),str(int(data.iloc[i,2])),str(data.iloc[i,3])
    if(int(data.iloc[i,1])<10):
        m="0"+str(int(data.iloc[i,1]))
    if(int(data.iloc[i,2])<10):
        d="0"+str(int(data.iloc[i,2]))
# print(m,d)
    time=str(int(data.iloc[i,0]))+"-"+m+"-"+d+" "+h
# print(time)
    time_list.append(time)
temp=pd.DataFrame(time_list,columns=["时刻"])
temp["时刻"]=pd.to_datetime(temp["时刻"], format="%Y-%m-%d %H:%M", errors='coerce')
# temp.to_excel('data1.xlsx',index=False)
temp
# In[1]:
data=pd.read_excel('D:\desktop\Modeling\data.xlsx')
data=data.iloc[:,4:8]

x=data.iloc[:,0]
y=data.iloc[:,1]
# In[1]:
def Zscore_outlier(df):
    m = np.mean(df)
    sd = np.std(df)
    out=[]
    row_indices = []
    for i, value in enumerate(df): 
        z = (value-m)/sd
        if np.abs(z) > 3: 
            out.append(value)
            row_indices.append(i)
    print("异常值:",out,'\n',"索引为：",row_indices)

# 检测异常值
print(Zscore_outlier(x))
print(Zscore_outlier(y))
# In[1]:
# 进行正态分布检验
statistic, p_value = normaltest(data.iloc[1:,1])

# 绘制数据的直方图
plt.hist(data.iloc[1:,1], bins=30, density=True, alpha=0.6, color='g')

# 绘制正态分布的概率密度函数
x = np.linspace(np.min(data.iloc[1:,1]), np.max(data.iloc[1:,1]), 100)
pdf = norm.pdf(x, loc=np.mean(data.iloc[1:,1]), scale=np.std(data.iloc[1:,1]))
plt.plot(x, pdf, 'r', label='Normal Distribution')

# 添加标题和标签
plt.title('Histogram of Data with Normal Distribution Curve')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# 显示图形
plt.show()
# 打印正态分布检验的结果
if p_value > 0.05:
    print("数据服从正态分布")
else:
    print("数据不服从正态分布")
# In[1]:
#将时间序列转换为数值型特征
df1=data.copy()
df1.iloc[:,0]=df1.iloc[:,0].apply(lambda x:x.timestamp())
df1.to_excel('D:\desktop\Modeling\data1.xlsx',index=False)
# In[1]:
# 对最后一列进行分段线性插补
df1.iloc[:,3]=df1.iloc[:,3].interpolate(method='polynomial', order=2)
df1.iloc[:,3]
df1.to_excel('D:\desktop\Modeling\data1.xlsx',index=False)
# In[1]:
# 绘制时间序列图
df1=pd.read_excel('D:\desktop\Modeling\data1.xlsx')
plt.plot(df1.iloc[:,0],df1.iloc[:,1],label='Water Level')
plt.xlabel('Time')
plt.ylabel('Value')
plt.figure()
plt.plot(df1.iloc[:,0],df1.iloc[:,2],label='Flow Rate')
plt.xlabel('Time')
plt.ylabel('Value')
plt.figure()
plt.plot(df1.iloc[:,0],df1.iloc[:,3],label='Sediment Concentration')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
# In[1]:
df1=pd.read_excel('D:\desktop\Modeling\data2.xlsx')
# 画出第二列到第四列两两之间的散点图
# 中文显示问题，需要设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
sns.pairplot(df1.iloc[:,4:7], diag_kind='kde', plot_kws={'s': 5})
plt.show()
# 求出相关系数矩阵
corr_matrix = df1.iloc[:,4:7].corr()
corr_matrix

# In[1]:
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

# In[1]:
from scipy.optimize import minimize
import numpy as np
from math import exp
from scipy.optimize import curve_fit
import numpy as np

# 定义要拟合的函数
def func(t, a, b):
    return a + b*np.log(t)

# 假设您有一些示例数据
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
# In[1]: