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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 定义特征提取函数
def extract_features(data):
    features = {}

    # 时间域特征
    for col in ['acc_x(g)', 'acc_y(g)', 'acc_z(g)', 'gyro_x(dps)', 'gyro_y(dps)', 'gyro_z(dps)']:
        features[f'{col}_mean'] = np.mean(data[col])
        features[f'{col}_std'] = np.std(data[col])
        features[f'{col}_max'] = np.max(data[col])
        features[f'{col}_min'] = np.min(data[col])
        features[f'{col}_median'] = np.median(data[col])
        features[f'{col}_q1'] = np.percentile(data[col], 25)
        features[f'{col}_q3'] = np.percentile(data[col], 75)
        features[f'{col}_skew'] = skew(data[col])
        features[f'{col}_kurtosis'] = kurtosis(data[col])

    # 频率域特征
    for col in ['acc_x(g)', 'acc_y(g)', 'acc_z(g)', 'gyro_x(dps)', 'gyro_y(dps)', 'gyro_z(dps)']:
        fft_vals = fft(data[col].values)
        fft_freq = np.fft.fftfreq(len(fft_vals))
        fft_power = np.abs(fft_vals) ** 2
        features[f'{col}_fft_power'] = np.sum(fft_power)
        features[f'{col}_fft_peak_freq'] = fft_freq[np.argmax(fft_power)]
    
    return features

# 文件夹路径和文件命名规则
base_folder_path = 'D:/desktop/模拟论文1/附件2'
file_prefix = 'a'
file_prefix1 = 't'
file_suffix = '.xlsx'

# 存储所有特征数据的列表
all_features = []

# 遍历所有实验人员
for person_id in range(4, 14):  # 这里假设有10位实验人员
    folder_path = os.path.join(base_folder_path, f'Person{person_id}')
    
    # 遍历该实验人员的所有文件，提取特征
    for i in range(1, 13):
        for j in range(1,6):
            file_name = f'{file_prefix}{i}{file_prefix1}{j}{file_suffix}'
            file_path = os.path.join(folder_path, file_name)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f'文件 {file_path} 不存在，跳过。')
                continue
            
            # 读取数据
            data = pd.read_excel(file_path)
            
            # 提取特征
            features = extract_features(data)
            
            # 添加实验人员和文件信息
            features['person_id'] = person_id
            features['file_name'] = file_name
            
            all_features.append(features)

# 将所有特征数据转换为DataFrame
feature_data = pd.DataFrame(all_features)
feature_data.to_csv('附件2_feature_data.csv', index=None)

# # 判别模型

feature_data = pd.read_csv('./附件2_feature_data.csv')
# 提取活动类别函数 根據文件名
def extract_activity_type(file_name):
    return int(file_name.split('t')[0][1:])

# 对feature_data['file_name']中的每个元素应用提取函数
feature_data['activity_type'] = feature_data['file_name'].apply(extract_activity_type)

# 标准化特征
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data.drop(columns=['file_name','person_id']))

# 准备训练数据和标签
X = scaled_features
y = feature_data['activity_type']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train) 

# 预测测试集
y_pred = clf.predict(X_test)

# 计算分类准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 构建混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

labels = range(1,13)
# 创建热力图
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

# 添加图表标题和标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# 显示图表
plt.show()

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data.drop(columns=['file_name','person_id']))

# 使用K-means聚类分析
kmeans = KMeans(n_clusters=12, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# 将聚类结果加入原数据
feature_data['Cluster'] = clusters
feature_data.head()

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

# 打印分类结果的统计信息
result_table = feature_data.groupby('Cluster').size().reset_index(name='Counts')
print(result_table)


# # 比较分类结果
from collections import Counter

# 构建二元组
feature_data['tuple'] = list(zip(feature_data['activity_type'], feature_data['Cluster']))

# 统计二元组出现的频率
tuple_counts = Counter(feature_data['tuple'])

# 找出数量最多的12类
top_12_tuples = tuple_counts.most_common(18)

from sklearn.metrics import confusion_matrix, accuracy_score
# 提取实际活动状态和预测的聚类状态
y_true = feature_data['activity_type']
y_pred = feature_data['Cluster']

# 构建混淆矩阵
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 计算整体分类准确度
accuracy = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {accuracy:.2f}")

# 计算每种活动状态的分类准确度
activity_types = sorted(set(y_true))
activity_accuracies = {}

for activity in activity_types:
    idx = (y_true == activity)
    correct_predictions = np.sum(y_true[idx] == y_pred[idx])
    total_predictions = np.sum(idx)
    activity_accuracies[activity] = correct_predictions / total_predictions

print("Activity-wise Accuracies:")
for activity, acc in activity_accuracies.items():
    print(f"Activity {activity}: {acc:.2f}")

# 优化混淆矩阵
optimized_matrix = conf_matrix.copy()
num_classes = conf_matrix.shape[0]

for i in range(num_classes):
    max_index = np.argmax(conf_matrix[i])
    if max_index != i:
        # 交换行 i 和 max_index
        optimized_matrix[[i, max_index], :] = optimized_matrix[[max_index, i], :]

print("Optimized Confusion Matrix:")
print(optimized_matrix)

# 重新计算整体分类准确度
optimized_accuracy = np.trace(optimized_matrix) / np.sum(optimized_matrix)
print(f"Optimized Overall Accuracy: {optimized_accuracy:.2f}")


# # 附件3检验
# 文件夹路径和文件命名规则
folder_path = 'D:/desktop/模拟论文1/附件3/附件3/'
file_prefix = 'SY'
file_suffix = '.xlsx'

# 存储所有特征数据的列表
all_features = []

# 遍历所有文件，提取特征
for i in range(1, 31):  # 这里假设有60个文件
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

# 创建一个字典来存储每个聚类类别的主要活动类型
cluster_to_activity = {}

for (activity, cluster), count in tuple_counts.most_common(32):
    if cluster not in cluster_to_activity:
        cluster_to_activity[cluster] = activity

# 创建一个新的列，将 Cluster 转换为主要活动类型
feature_data['Predicted_Activity'] = feature_data['Cluster'].map(cluster_to_activity)
feature_data['Predicted_Activity']=feature_data['Predicted_Activity'].fillna(0,inplace=False)
feature_data['Predicted_Activity'] = (feature_data['Predicted_Activity']+1).astype(int)
print(feature_data['Predicted_Activity'])

# 活动类别对应的活动名称
activity_labels = {
    1: '向前走',
    2: '向左走',
    3: '向右走',
    4: '步行上楼',
    5: '步行下楼',
    6: '向前跑',
    7: '跳跃',
    8: '坐下',
    9: '站立',
    10: '躺下',
    11: '乘坐电梯向上移动',
    12: '乘坐电梯向下移动'
}

# 创建数据框
df_results = pd.DataFrame()

ls = range(1,31)
file_names = []
for i in ls:
    file_name = 'SY'+ str(i)
    file_names.append(file_name)
df_results['file_name'] = file_names
df_results['Activity_Label'] = feature_data['Predicted_Activity'].map(activity_labels)

# 保存结果到Excel文件
df_results[['file_name', 'Activity_Label']].to_excel('./问题2结果.xlsx', index=False, header=['活动类型', '判别状态'])

# 查看结果数据框(
print(df_results)