
import pandas as pd
import numpy as np

# 读取数据
data = pd.read_excel('D:\desktop\Modeling\data.xls')

column_types = data.dtypes
str_columns = column_types[column_types == 'object'].index
for col in data.columns:
    if col not in str_columns:
        # 检测缺失值
        missing_values = data[col].isnull()
        print(missing_values)
        # 输出缺失值的索引
        missing_value_indices = data[missing_values].index
        print("缺失值的索引：", missing_value_indices)

# 填充缺失值
data=data.fillna(axis=0,method='ffill')

for col in data.columns:
    if col not in str_columns:
        # 检测缺失值
        missing_values = data[col].isnull()
        print(missing_values)
        # 输出缺失值的索引
        missing_value_indices = data[missing_values].index
        print("缺失值的索引：", missing_value_indices)

# 保存数据
data.to_csv('D:\desktop\Modeling\data_filled.csv',index=False)