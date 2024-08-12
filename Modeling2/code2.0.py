#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
import statsmodels.api as sm
import statsmodels.stats.diagnostic

# In[1]:
data = pd.read_excel('D:\desktop\python\math1\score.xlsx')
# data.iloc[:,1]=data.iloc[:,1]*-1
data

# In[1]:
# 绘图
plt.rcParams['font.sans-serif'] = ['SimHei']
fig, axes = plt.subplots(nrows=3, ncols=1, dpi=120, figsize=(15,15))
for i, ax in enumerate(axes.flatten()):
    df = data[data.columns[i]]
    ax.plot(df, color='red', linewidth=1)
    # Decorations
    ax.set_title(data.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# In[1]:
# 正态分布检验
fig = plt.figure(figsize = (10,6))
datas=data.iloc[:,0]
ax2 = fig.add_subplot(1,1,1)
datas.hist(bins=50,ax = ax2)
datas.plot(kind = 'kde', secondary_y=True,ax = ax2)
plt.grid()

plt.show()	

# In[1]:
# 格兰杰因果检验
from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
            print(test_result)
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
data1=data.iloc[:,:2]
data2=data.iloc[:,[0,2]]
grangers_causation_matrix(data1, variables = data1.columns)   
grangers_causation_matrix(data2, variables = data2.columns)   

# In[1]:
# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data1=data_scaled.iloc[:,:2]
data2=data_scaled.iloc[:,[0,2]]

# 协整测试
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(data1)
cointegration_test(data2)

# In[1]:
# 训练集和测试集划分
nobs = 4
df_train1, df_test1 = data1[0:-nobs], data1[-nobs:]
df_train2, df_test2 = data2[0:-nobs], data2[-nobs:]

# Check size
print(df_train1.shape,df_train2.shape)
print(df_test1.shape,df_test2.shape)  
# In[1]:
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    
for name, column in df_train1.items():
    adfuller_test(column, name=column.name)
    print('\n')
for name, column in df_train2.items():
    adfuller_test(column, name=column.name)
    print('\n')

# In[1]:
# 做一阶差分
# 1st difference
df_differenced1 = df_train1.diff().dropna()
df_differenced2 = df_train2.diff().dropna()
for name, column in df_differenced1.items():
    adfuller_test(column, name=column.name)
    print('\n')
for name, column in df_differenced2.items():
    adfuller_test(column, name=column.name)
    print('\n')

# In[1]:
# 选择VAR模型的阶数P
model1 = VAR(df_differenced1)
model2 = VAR(df_differenced2)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model1.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
for i in [1,2,3,4,5,6,7,8,9]:
    result = model2.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

# In[1]:
# 选取最优的VAR模型
x = model1.select_order(maxlags=12)
x.summary()
x = model2.select_order(maxlags=12)
x.summary()
# In[1]:
# 训练VAR模型
model_fitted = model1.fit(9)
model_fitted.summary()
# In[1]:
# DW检验
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(data1.columns, out):
    print(col, ':', round(val, 2))

# In[1]:
# 用VAR模型来预测时序数据
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_differenced1.values[-lag_order:]
forecast_input

# In[1]:
# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=data1.index[-nobs:], columns=data1.columns + str(0.1))
df_forecast

# In[1]:
# 逆向
def invert_transformation(df_train, df_forecast, second_diff=True):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[col+str(0.1)] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[col+str(0.1)].cumsum()
        # Roll back 1st Diff
        df_fc[col+str(0.2)] = df_train[col].iloc[-1] + df_fc[col+str(0.1)].cumsum()
    return df_fc
df_results = invert_transformation(df_train1, df_forecast, second_diff=True)        
df_results

# In[1]:
# 绘图
fig, axes = plt.subplots(nrows=int(len(data1.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(data1.columns, axes.flatten())):
    df_results[col+str(0.2)].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test1[col][-nobs:].plot(legend=True, ax=ax)
    ax.set_title(col)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# In[1]:
# 计算整体准确度
from statsmodels.tsa.stattools import acf
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                                actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                                actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

df1 = np.array(df_test1.iloc[:,0])
df2 = np.array(df_results.iloc[:,2])
df3 = np.array(df_test1.iloc[:,1])
df4 = np.array(df_results.iloc[:,3])
print('Forecast Accuracy of: rgnp')
accuracy_prod = forecast_accuracy(df2, df1)
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))

print('\nForecast Accuracy of: pgnp')
accuracy_prod = forecast_accuracy(df4, df3)
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))

from sklearn import metrics
r2 = metrics.r2_score(df1,df2)    
print(r2)
# In[1]:
# In[1]:
# In[1]: