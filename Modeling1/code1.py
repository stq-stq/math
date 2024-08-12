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

# 定义特征提取函数
def extract_features(data):
    features = {}

    # 时间域特征
    for col in ['acc_x(g)', 'acc_y(g)', 'acc_z(g)', 'gyro_x(dps)', 'gyro_y(dps)', 'gyro_z(dps)']:
        features[f'{col}_mean'] = np.mean(data[col])
        features[f'{col}_std'] = np.std(data[col])
        features[f'{col}_max'] = np.max(data[col])
        features[f'{col}_min'] = np.min(data[col])
        features[f'{col}_median'] = np.median(data[col])# 中位数
        features[f'{col}_q1'] = np.percentile(data[col], 25)# 25%分位数
        features[f'{col}_q3'] = np.percentile(data[col], 75)# 75%分位数
        features[f'{col}_skew'] = skew(data[col])# 偏度
        features[f'{col}_kurtosis'] = kurtosis(data[col])# 峰度

    # 频率域特征
    for col in ['acc_x(g)', 'acc_y(g)', 'acc_z(g)', 'gyro_x(dps)', 'gyro_y(dps)', 'gyro_z(dps)']:
        fft_vals = fft(data[col].values)# 快速傅里叶变换
        fft_freq = np.fft.fftfreq(len(fft_vals))# 频率
        fft_power = np.abs(fft_vals) ** 2# 功率
        features[f'{col}_fft_power'] = np.sum(fft_power)# 频谱能量
        features[f'{col}_fft_peak_freq'] = fft_freq[np.argmax(fft_power)]# 频谱峰值频率
    
    return features

# person1
folder_path = 'D:/desktop/data/附件1/Person1/'
file_prefix = 'SY'
file_suffix = '.xlsx'
# 存储所有特征数据的列表
all_features = []
# 遍历所有文件，提取特征
for i in range(1, 61):  # 这里假设有60个文件
    file_name = f'{file_prefix}{i}{file_suffix}'
    file_path = os.path.join(folder_path, file_name)
    
    # 读取数据
    data = pd.read_excel(file_path)
    
    # 提取特征
    features = extract_features(data)
    all_features.append(features)

# 将所有特征数据转换为DataFrame
feature_data = pd.DataFrame(all_features)

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data)

# 使用K-means聚类分析
kmeans = KMeans(n_clusters=12, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# 将聚类结果加入原数据
feature_data['Cluster'] = clusters
feature_data.head()

plt.figure(figsize=(10, 7))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=clusters, palette='viridis')#
plt.title('Clusters of Activities')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend(title='Cluster')
plt.show()

from sklearn.decomposition import PCA
# PCA降维到2D
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# 聚类结果展示
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7)
plt.title('K-means Clusters Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# 分类后第一列为对应的数字，第二列为分组数量
result_table = feature_data.groupby('Cluster').size().reset_index(name='Counts')

# person2
folder_path = 'D:/desktop/data/附件1/Person2/'
file_prefix = 'SY'
file_suffix = '.xlsx'

# 存储所有特征数据的列表
all_features = []

# 遍历所有文件，提取特征
for i in range(1, 61):  # 这里假设有60个文件
    file_name = f'{file_prefix}{i}{file_suffix}'
    file_path = os.path.join(folder_path, file_name)
    
    # 读取数据
    data = pd.read_excel(file_path)
    
    # 提取特征
    features = extract_features(data)
    all_features.append(features)

# 将所有特征数据转换为DataFrame
feature_data1 = pd.DataFrame(all_features)

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data1)#标准化和拟合

# 使用K-means聚类分析
kmeans = KMeans(n_clusters=12, random_state=42)
clusters = kmeans.fit_predict(scaled_features)# 每个文件对应的聚类结果

# 将聚类结果加入原数据
feature_data1['Cluster'] = clusters

#  聚类结果展示
plt.figure(figsize=(10, 7))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=clusters, palette='viridis')# 不同色调体现
plt.title('Clusters of Activities')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend(title='Cluster')
plt.show()

from sklearn.decomposition import PCA
# PCA降维到2D 主成分分析
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# 聚类结果展示
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7)
plt.title('K-means Clusters Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# 分类后第一列为对应的数字，第二列为分组数量
result_table = feature_data.groupby('Cluster').size().reset_index(name='Counts')

# person3
folder_path = 'D:/desktop/data/附件1/Person3/'
file_prefix = 'SY'
file_suffix = '.xlsx'

all_features = []
for i in range(1, 61):  
    file_name = f'{file_prefix}{i}{file_suffix}'
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_excel(file_path)
    features = extract_features(data)
    all_features.append(features)

feature_data2 = pd.DataFrame(all_features)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data2)

kmeans = KMeans(n_clusters=12, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

feature_data2['Cluster'] = clusters

# 整合
clusters = feature_data['Cluster']
clusters1 = feature_data1['Cluster']
clusters2 = feature_data2['Cluster']

columns = ['分类', 'Person1', 'Person2', 'Person3']
results = pd.DataFrame(columns=columns)

# 填充分类数据
results['分类'] = [f'第{i}类' for i in range(1, 13)]
results['Person1'] = [sum(clusters == i) for i in range(12)]
results['Person2'] = [sum(clusters1 == i) for i in range(12)]
results['Person3'] = [sum(clusters2 == i) for i in range(12)]

# 获取聚类结果
clusters = feature_data['Cluster']

# 创建一个空的DataFrame用于保存最终的结果
columns = ['分类', 'Person1', 'Person2', 'Person3']
results1 = pd.DataFrame(columns=columns)

# 填充分类数据
results1['分类'] = [f'第{i}类' for i in range(1, 13)]
results1['Person1'] = [''] * 12  # 初始化为空字符串
results1['Person2'] = [''] * 12  # 假设Person2没有数据
results1['Person3'] = [''] * 12  # 假设Person3没有数据

# 根据数据顺序推导文件名
file_names = [f'SY{i}' for i in range(1, len(feature_data) + 1)]

# 填充Person1的具体实验编号
for i in range(12):
    indices = feature_data[clusters == i].index
    indices1 = feature_data[clusters1 == i].index
    indices2 = feature_data[clusters2 == i].index
    experiment_list = [file_names[idx] for idx in indices]
    experiment_list1 = [file_names[idx] for idx in indices1]
    experiment_list2 = [file_names[idx] for idx in indices2]
    results1.at[i, 'Person1'] = ', '.join(experiment_list)
    results1.at[i, 'Person2'] = ', '.join(experiment_list1)
    results1.at[i, 'Person3'] = ', '.join(experiment_list2)

results.to_csv('聚类个数.csv', index=None)
results1.to_csv('聚类实验编号.csv', index=None)

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data1)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=clusters, palette='viridis')
plt.title('Clusters of Activities')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend(title='Cluster')
plt.show()

# PCA降维到2D
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# 聚类结果展示
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7)
plt.title('K-means Clusters Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()