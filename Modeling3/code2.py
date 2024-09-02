# #!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import norm
import seaborn as sns
from pywt import wavedec
import pywt

# 读取数据
data=pd.read_excel('D:\desktop\Modeling\data2.xlsx')
print(data)

col1_name=data.columns[7]
col2_name=data.columns[8]
mounths_data=data.groupby([data.iloc[:,0],data.iloc[:,1]]).agg({col1_name:'sum',col2_name:'sum'})
# mounths_data.to_excel('D:\desktop\Modeling\mounths_data.xlsx')
time_list=[]
for i in range(len(mounths_data)):
    m=str(int(data.iloc[i,1]))
    if(int(data.iloc[i,1])<10):
        m="0"+str(int(data.iloc[i,1]))
# print(m,d)
    time=str(int(data.iloc[i,0]))+"-"+m
# print(time)
    time_list.append(time)
temp=pd.DataFrame(time_list,columns=["时刻"])
# temp["时刻"]=pd.to_datetime(temp["时刻"], format="%Y-%m", errors='coerce')
temp.to_excel('data1.xlsx',index=False)
print(temp)

data=pd.read_excel('D:\desktop\Modeling\mounths_data.xlsx')
print(data)

# 绘制月度水沙通量图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12, 6))
plt.plot(data.iloc[:,2], data[col1_name], label='Sales')
plt.title('Monthly values')
plt.xlabel('Date')
plt.ylabel(col1_name)
plt.figure(figsize=(12, 6))
plt.plot(data.iloc[:,2], data[col2_name], label='Orders')
plt.title('Monthly values')
plt.xlabel('Date')
plt.ylabel(col2_name)
plt.legend()
plt.show()


# 季节性分解
# 绘制月度水通量图
seasonal0 = pd.read_excel('data1.xlsx')
seasonal = seasonal0.iloc[:,1]

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data.index, data.iloc[:,4])
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel(col1_name)

plt.subplot(2, 1, 2)
plt.plot(seasonal.index, seasonal)
plt.title('Seasonal Component')
plt.xlabel('Date')
plt.ylabel('Seasonal Factor')

plt.tight_layout()
plt.show()
seasonal = seasonal0.iloc[:,2]

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data.index, data.iloc[:,4])
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel(col2_name)

plt.subplot(2, 1, 2)
plt.plot(seasonal.index, seasonal)
plt.title('Seasonal Component')
plt.xlabel('Date')
plt.ylabel('Seasonal Factor')

plt.tight_layout()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(data.iloc[:,3], lags=24, ax=plt.gca(), title=col1_name+'ACF')

plt.subplot(1, 2, 2)
plot_pacf(data.iloc[:,3], lags=24, ax=plt.gca(), title=col1_name+'PACF')

plt.show()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(data.iloc[:,4], lags=24, ax=plt.gca(), title=col2_name+'ACF')

plt.subplot(1, 2, 2)
plot_pacf(data.iloc[:,4], lags=24, ax=plt.gca(), title=col2_name+'PACF')

plt.show()


from statsmodels.tsa.stattools import adfuller
# 标准化数据
data.iloc[:,4] = (data.iloc[:,4] - data.iloc[:,4].mean()) / data.iloc[:,3].std()

seasonal_diff = data.iloc[:,4] - data.iloc[:,4].shift(12)
seasonal_diff.dropna(inplace=True)

result = adfuller(seasonal_diff)
print(col1_name+'ADF Statistic:', result[0])
print(col1_name+'p-value:', result[1])

# 小波分析
from pywt import waverec
from pywt import wavedec

# 计算小波系数
coef, freq = wavedec(data.iloc[:,7], 'db4', level=4)

# 重构信号
data_rec = waverec(coef, 'db4')

# 绘制原始信号和重构信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data.index, data.iloc[:,7])
plt.title('Original Time Series')
plt.xlabel('Date')
plt.ylabel(col1_name)

plt.subplot(2, 1, 2)
plt.plot(data.index, data_rec)
plt.title('Reconstructed Time Series')
plt.xlabel('Date')
plt.ylabel(col1_name)


plt.tight_layout()
plt.show()


from matplotlib.font_manager import FontProperties
#从Exce1文件读取小波系数实部数据
file_path=r'D:\desktop\Modeling\mounths_data.xlsx' #使用原始字符串
try:
    df = pd.read_excel(file_path,header=0)
    df=df.iloc[:,3:6]
except FileNotFoundError:
    print(f"File not found: {file_path}")
    raise

#提取时间轴和小波系数实部数据
time_axis = df.iloc[:,[0,2]].columns.values
scales = df.iloc[:,0].values
coefficients_real = df.iloc[:,[0,2]].values
plt.rcParams['font.sans-serif'] = ['SimHei']
#绘制小波系数实部的等值线图
plt.figure(figsize=(8,20),dpi=300)
contourf = plt.contourf(time_axis, scales, coefficients_real, cmap='jet')
#Colorbar设置
cbar = plt.colorbar(contourf,label='Real Value',ax=plt.gca())#使用ax参数将Colorbar绑定到当前的Axes
# 调整colorbar的标签字体大小和字体
cbar.ax.tick_params(axis='y',labelsize=24) # 适当调整字体大小
for label in cbar.ax.yaxis.get_ticklabels():
    label.set_size(24)#设置刻度字体大小
#绘制所有等值线，使用黑色的细线
contour_lines = plt.contour(time_axis, scales, coefficients_real, colors='black', linewidths=0.5)
plt.show()

x = df.iloc[:,0].values
y = df.iloc[:,2].values

# 小波变换
w = pywt.Wavelet('db4')
maxlev = pywt.dwt_max_level(len(y), w.dec_len)
print("maximum level is " + str(maxlev))# 最大层数
threshold = 0.04 # 设定阈值
coeffs = pywt.wavedec(y, 'db4', level=maxlev) # 小波分解
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
y2 = pywt.waverec(coeffs, 'db4')

# 计算方差
N = len(y)
variance = []
for i in range(1, N):
    variance.append(np.var(y2[:i] - y[:i]))

# 绘制小波方差图
plt.plot(variance)
plt.xlabel('time')
plt.ylabel('variance')
plt.title('Wavelet variance')
plt.show()