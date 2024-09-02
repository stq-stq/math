# In[1]:
# #!/usr/bin/env python
# coding: utf-8
#准备工作
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import itertools
from sklearn.metrics import r2_score as rs
import warnings

warnings.filterwarnings("ignore")#忽略输出警告
plt.rcParams["font.sans-serif"]=["SimHei"]#用来正常显示中文标签
plt.rcParams["axes.unicode_minus"]=False#用来显示负号


# In[1]:
df=pd.read_excel("D:\desktop\Modeling\mounths_data.xlsx")
df
# In[1]:
#分解时序
#STL（Seasonal and Trend decomposition using Loess）是一个非常通用和稳健强硬的分解时间序列的方法
import statsmodels.api as sm
data1=df.iloc[:,4]
data1.index=df.iloc[:,2]
# # 数据标准化
# data1=(data1-data1.mean())/data1.std()
decompostion=sm.tsa.STL(data1, period=12).fit()#statsmodels.tsa.api:时间序列模型和方法
decompostion.plot()
#趋势效益
trend=decompostion.trend
#季节效应
seasonal=decompostion.seasonal
#随机效应
residual=decompostion.resid

# In[1]:
#平稳性检验
#自定义函数用于ADF检查平稳性
from statsmodels.tsa.stattools import adfuller as ADF
def test_stationarity(timeseries,alpha):#alpha为检验选取的显著性水平
    adf=ADF(timeseries)
    p=adf[1]#p值
    critical_value=adf[4]["5%"]#在95%置信区间下的临界的ADF检验值
    test_statistic=adf[0]#ADF统计量
    if p<alpha and test_statistic<critical_value:
        print("ADF平稳性检验结果：在显著性水平%s下，数据经检验平稳"%alpha)
        return True
    else:
        print("ADF平稳性检验结果：在显著性水平%s下，数据经检验不平稳"%alpha)
        return False
#原始数据平稳性检验
test_stationarity(data1,1e-3)
# water_seasonal=data1

# In[1]:
#将数据化为平稳数据
#一阶差分
water_diff1=data1.diff(1)
#十三步差分
water_seasonal=water_diff1.diff(13)#非平稳序列经过d阶常差分和D阶季节差分变为平稳时间序列
print(water_seasonal)
#十三步季节差分平稳性检验结果
test_stationarity(water_seasonal.dropna(),1e-3)#使用dropna()去除NaN值

# In[1]:
#LB白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
re = lb_test(water_seasonal.dropna(), lags=1)
print(re)
# In[1]:
# #搜索法定阶
# def SARIMA_search(data):
#     p=q=range(0,3)
#     s=[12]#周期为12
#     d=[1]#做了一次季节性差分
#     PDQs=list(itertools.product(p,d,q,s))#itertools.product()得到的是可迭代对象的笛卡儿积
#     pdq=list(itertools.product(p,d,q))#list是python中是序列数据结构，序列中的每个元素都分配一个数字定位位置
#     params=[]
#     seasonal_params=[]
#     results=[]
#     grid=pd.DataFrame()
#     for param in pdq:
#         for seasonal_param in PDQs:
#             #建立模型
#             mod= sm.tsa.SARIMAX(data,order=param,seasonal_order=seasonal_param,\
#                             enforce_stationarity=False, enforce_invertibility=False)
#             #实现数据在模型中训练
#             result=mod.fit()
#             print("ARIMA{}x{}-AIC:{}".format(param,seasonal_param,result.aic))
#             #format表示python格式化输出，使用{}代替%
#             params.append(param)
#             seasonal_params.append(seasonal_param)
#             results.append(result.aic)
#     grid["pdq"]=params
#     grid["PDQs"]=seasonal_params
#     grid["aic"]=results
#     print(grid[grid["aic"]==grid["aic"].min()])
    
# SARIMA_search(water_seasonal.dropna())

# In[1]:
#建立模型
model=sm.tsa.SARIMAX(data1,order=(3,1,3),seasonal_order=(2,1,1,12))
SARIMA_m=model.fit()
print(SARIMA_m.summary())
# In[1]:
#模型检验
# test_white_noise(SARIMA_m.resid,0.05)#SARIMA_m.resid提取模型残差，并检验是否为白噪声
fig=SARIMA_m.plot_diagnostics(figsize=(15,12))#plot_diagnostics对象允许我们快速生成模型诊断并调查任何异常行为

# In[1]:
#模型预测
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
#获取预测结果，自定义预测误差
def PredictionAnalysis(data,model,start,dynamic=False):
    pred=model.get_prediction(start=start,dynamic=dynamic,full_results=True)
    pci=pred.conf_int()#置信区间
    pm=pred.predicted_mean#预测值
    truth=data[start:]#真实值
    pc=pd.concat([truth,pm,pci],axis=1)#按列拼接
    pc.columns=['true','pred','up','low']#定义列索引
    print("1、MSE:{}".format(mse(truth,pm)))
    print("2、RMSE:{}".format(np.sqrt(mse(truth,pm))))
    print("3、MAE:{}".format(mae(truth,pm)))
    return pc
#绘制预测结果
def PredictonPlot(pc):
    plt.figure(figsize=(10,8))
    plt.fill_between(pc.index,pc['up'],pc['low'],color='grey',\
                     alpha=0.15,label='confidence interval')#画出置信区间
    plt.plot(pc['true'],label='base data')
    plt.plot(pc['pred'],label='prediction curve')
    plt.legend()
    plt.show
    return True
pred=PredictionAnalysis(data1,SARIMA_m,2,dynamic=False)
PredictonPlot(pred)

# In[1]:
#预测未来
forecast=SARIMA_m.get_forecast(steps=24)
#预测整体可视化
fig,ax=plt.subplots(figsize=(20,16))
data1.plot(ax=ax,label="base data")
forecast.predicted_mean.plot(ax=ax,label="forecast data")
#ax.fill_between(forecast.conf_int().index(),forecast.conf_int().iloc[:,0],\
#               forecast.conf_int().iloc[:,1],color='grey',alpha=0.15,label='confidence interval')
ax.legend(loc="best",fontsize=20)
ax.set_xlabel("时间（月）",fontsize=20)
ax.set_ylabel("水通量",fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show
# 将预测结果保存到excel文件中
forecast_data=pd.DataFrame(forecast.predicted_mean)
forecast_data.to_excel("D:\desktop\Modeling\Forecast_data.xlsx")
# In[1]:
df1=pd.read_excel("D:\desktop\Modeling\data20160608.xlsx")
df2=pd.read_excel("D:\desktop\Modeling\data20161020.xlsx")
df3=pd.read_excel("D:\desktop\Modeling\data20170511.xlsx")
df4=pd.read_excel("D:\desktop\Modeling\data20170905.xlsx")
df4=pd.read_excel("D:\desktop\Modeling\data20180913.xlsx")
df5=pd.read_excel("D:\desktop\Modeling\data20190413.xlsx")
df6=pd.read_excel("D:\desktop\Modeling\data20191015.xlsx")
df7=pd.read_excel("D:\desktop\Modeling\data20200319.xlsx")
df8=pd.read_excel("D:\desktop\Modeling\data20210314.xlsx")
labels = ['Data 1', 'Data 2', 'Data 3', 'Data 4', 'Data 5', 'Data 6', 'Data 7', 'Data 8']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

plt.figure(figsize=(10, 6))

for i, data in enumerate([df1,df2,df3,df4,df5,df6,df7,df8]):
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], label=labels[i], color=colors[i])

plt.title('Multiple Data on One Chart')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]: