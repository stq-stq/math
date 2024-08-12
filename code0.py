# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import norm
import seaborn as sns

# In[1]:
# 读取数据
data = pd.ExcelFile('D:\desktop\Modeling\附件1.xlsx')
data

for col in data.columns:
    # 检测缺失值
    missing_values = data[col].isnull()
    print(missing_values)
    # 输出缺失值的索引
    missing_value_indices = data[missing_values].index
    print("缺失值的索引：", missing_value_indices)

# data.to_excel('D:\desktop\Modeling\data.xlsx', index=False)

# In[1]:
## 根据时间变量变化的数据散点可视化
## 水位的变化情况
plt.figure(figsize=(12,3))
x = data.loc[:,1]
y = data.loc[:,5]
plt.scatter(x,y)
plt.xlabel("时间")
plt.ylabel("水位(m)")
plt.title("")
plt.show()

## 流量的变化情况
# plt.figure(figsize=(12,3))
# # dfq1 = data.loc[:,['年','流量(m3/s)']]
# # p = sns.lineplot(data=dfq1, x="年", y="流量(m3/s)",lw = 2)
# x = data.loc[:,0]
# y = data.loc[:,5]
# # p = sns.scatterplot(x,y)
# p = sns.lineplot(x,y,lw = 2)
# plt.xlabel("时间")
# plt.ylabel("流量("+"$m^3$"+"/s)")
# plt.title("")
# plt.savefig('figs/流量的变化情况.png', dpi=300, bbox_inches='tight')
# plt.show()

# ## 含沙量的变化情况
# plt.figure(figsize=(12,3))
# # dfq1 = data.loc[:,['年','含沙量(kg/m3)']]
# # p = sns.lineplot(data=dfq1, x="年", y="含沙量(kg/m3)",lw = 2)
# x = data.loc[:,0]
# y = data.loc[:,6]
# # p = sns.scatterplot(x,y)
# p = sns.lineplot(x,y,lw = 2)
# plt.xlabel("时间")
# plt.ylabel("含沙量(kg/"+"$m^3$"+")")
# plt.title("")
# plt.savefig('figs/含沙量的变化情况.png',dpi=300,bbox_inches='tight')
# plt.show()


# In[1]:
# 补填缺失值
def fill_missing_values(df):
    for i in range(len(df)):
        if df[i]==999:
            for j in range(i,len(df)):
                if df[j]!=999:
                    t= (df[i-1]+df[j])/2
                    df[i]=t
                    break
    return df
data = data.fillna(999)
data = fill_missing_values(data)
data
# In[1]:
# 绘制各列散点图
x = data.iloc[:, 4]
row_numbers = range(len(x))
# row_numbers = data.iloc[:,3]
y = data.iloc[:, 5]
plt.scatter(row_numbers, x)
plt.figure()
plt.scatter(row_numbers, y)
plt.title('2D Scatter Plot')
plt.xlabel('Row Number')
plt.ylabel('Data Value')
plt.show()
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
statistic, p_value = normaltest(data.iloc[:,5])

# 绘制数据的直方图
plt.hist(data.iloc[:,5], bins=30, density=True, alpha=0.6, color='g')

# 绘制正态分布的概率密度函数
x = np.linspace(np.min(data.iloc[:,5]), np.max(data.iloc[:,5]), 100)
pdf = norm.pdf(x, loc=np.mean(data.iloc[:,5]), scale=np.std(data.iloc[:,5]))
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
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]:
# In[1]: