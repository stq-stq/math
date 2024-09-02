
# #!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import norm
import seaborn as sns


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
print(temp)

data=pd.read_excel('D:\desktop\Modeling\data.xlsx')
data=data.iloc[:,4:8]

x=data.iloc[:,0]
y=data.iloc[:,1]

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

#将时间序列转换为数值型特征
df1=data.copy()
df1.iloc[:,0]=df1.iloc[:,0].apply(lambda x:x.timestamp())
df1.to_excel('D:\desktop\Modeling\data1.xlsx',index=False)

# 对最后一列进行分段线性插补
df1.iloc[:,3]=df1.iloc[:,3].interpolate(method='polynomial', order=2)
df1.iloc[:,3]
df1.to_excel('D:\desktop\Modeling\data1.xlsx',index=False)

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