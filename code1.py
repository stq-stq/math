# In[1]:
# #!/usr/bin/env python
# coding: utf-8
import pandas as pd
# In[1]:
#计算每年的总水流量和总排沙量
data=pd.read_excel('D:\desktop\Modeling\data2.xlsx')
col1_name = data.columns[5]
col2_name = data.columns[6] 
yearly_data=data.groupby(data.iloc[:,0]).agg({col1_name:'sum',col2_name:'sum'})
yearly_data.to_excel('yearly_data.xlsx',index=True)
# In[3]: