#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 所有文件缺失值检测 

parentFolder = r'D:\desktop\模拟论文1'  # 父文件夹路径

# 第一级
subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder, f))]

for i, subfolderName in enumerate(subfolders):
    subfolderPath = os.path.join(parentFolder, subfolderName)

    # 第二级
    innerSubfolders = [f for f in os.listdir(subfolderPath) if os.path.isdir(os.path.join(subfolderPath, f))]

    for j, innerSubfolderName in enumerate(innerSubfolders):
        innerSubfolderPath = os.path.join(subfolderPath, innerSubfolderName)

        # 获取 xlsx 文件
        files = [f for f in os.listdir(innerSubfolderPath) if f.endswith('.xlsx')]

        # 遍历 xlsx 文件
        miss = 0
        for k, fileName in enumerate(files):
            filePath = os.path.join(innerSubfolderPath, fileName)
            if fileName.startswith('~$'):
                continue

            data = pd.read_excel(filePath)

            # 查找缺失值
            for col in range(data.shape[1]):
                columnData = data.iloc[:, col]
                missingIndices = columnData[columnData.isnull()].index
            
            if len(missingIndices)!= 0:
                missing = '有'
                miss = 1
                print(f'{subfolderName} 文件下 {innerSubfolderName} 下 {fileName} 表格：{missing}缺失值')
                print('第', col, '列缺失值的索引：', missingIndices)
            else:
                missing = '无'

            #异常值
            # 处理后文件要保存的文件夹路径
            save_folder_path = 'D:\\desktop\\data\\'+ subfolderName +'\\' + innerSubfolderName +''

            # 数据异常值检测
            x = data.iloc[1:, 0]
            y = data.iloc[1:, 1]
            z = data.iloc[1:, 2]
            x1 = data.iloc[1:, 3]
            y1 = data.iloc[1:, 4]
            z1 = data.iloc[1:, 5]
            def Zscore_outlier(df):
                m = np.mean(df)
                sd = np.std(df)
                # out=[]
                # row_indices = []
                for i, value in enumerate(df): 
                    z = (value-m)/sd
                    if np.abs(z) > 3: 
                        # out.append(value)
                        # row_indices.append(i)
                        return False
                # print("异常值:",out,'\n',"索引为：",row_indices)
                return True

            def Winsorization_outliers(df):
                if Zscore_outlier(df):
                    return []
                q1 = np.percentile(df , 0.5)
                q3 = np.percentile(df , 99.5)
                out=[]
                row_indices = []
                for i, value in enumerate(df):
                    if value > q3 or value < q1:
                        out.append(value)
                        row_indices.append(i)
                # print("异常值:",out,'\n',"索引为：",row_indices)
                return row_indices
            arr=Winsorization_outliers(x)+Winsorization_outliers(y)+Winsorization_outliers(z)+Winsorization_outliers(x1)+Winsorization_outliers(y1)+Winsorization_outliers(z1)
            dataNew=data.drop(arr,axis=0)
            
            filtered_file_path = os.path.join(save_folder_path, fileName)
            dataNew.to_excel(filtered_file_path, index=False)

            # 绘制各列散点图
            row_numbers = np.arange(len(data)-1)
            plt.figure(figsize=(12, 8))
            plt.scatter(row_numbers, x)
            plt.scatter(row_numbers, y)
            plt.scatter(row_numbers, z)
            plt.scatter(row_numbers, x1)
            plt.scatter(row_numbers, y1)
            plt.scatter(row_numbers, z1)
            plt.title('2D Scatter Plot')
            plt.xlabel('Row Number')
            plt.ylabel('Data Value')

        if miss == 0:
            print(f'{subfolderName} 文件下 {innerSubfolderName}无缺失值')