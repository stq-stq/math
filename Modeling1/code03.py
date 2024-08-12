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
from statsmodels.formula.api import ols
import statsmodels.api as sm

feature_data = pd.read_csv('./附件2_feature_data.csv')
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
base_folder_path = 'D:/desktop/模拟论文1/附件1'
file_prefix = 'SY'
file_suffix = '.xlsx'

# 存储所有特征数据的列表
all_features = []

# 遍历所有实验人员
for person_id in range(1, 4):  
    folder_path = os.path.join(base_folder_path, f'Person{person_id}')
    
    # 遍历该实验人员的所有文件，提取特征
    for i in range(1, 61):
            file_name = f'{file_prefix}{i}{file_suffix}'
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
feature_data1 = pd.read_csv('./附件2_feature_data.csv')

# 将附件12合并
feature_data = pd.concat([feature_data, feature_data1])

feature_data.to_csv('./问题3_feature_data.csv',index=None)

# # 代码正式开始部分
feature_data = pd.read_csv('./问题3_feature_data.csv')
# 数据清洗和归一化
scaler = StandardScaler()
# 计算总体加速度 去掉之前的person_id, file_name
data_scaled = scaler.fit_transform(feature_data.iloc[:, :-2])
# 将标准化后的数据转换回 DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=feature_data.columns[:-2])

# 计算总体加速度
data_scaled_df['acc'] = np.sqrt(data_scaled_df['acc_x(g)_mean']**2 + data_scaled_df['acc_y(g)_mean']**2 + data_scaled_df['acc_z(g)_mean']**2)

data_scaled_df['实验人员编号'] = feature_data['person_id']

# 方差分析 acc是因变量 实验人员编号是自变量 对数据进行拟合后进行方差分析
model = ols('acc ~ C(实验人员编号)', data=data_scaled_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

person_info = pd.read_excel('D:/desktop/模拟论文1/附件4.xlsx')
print(person_info)

# 清理人员信息数据
person_info.columns = ['实验人员编号', '年龄', '体重', '身高']
person_info['体重'] = person_info['体重'].str.replace('cm', '').astype(float)
person_info['身高'] = person_info['身高'].str.replace('kg', '').astype(float)

# 合并数据 将所有体重等数据与实验人员编号对应
merged_data = pd.merge(data_scaled_df, person_info, on='实验人员编号', how='left')

# 数据标准化
scaler = StandardScaler()
person_features = ['年龄', '体重', '身高']
merged_data[person_features] = scaler.fit_transform(merged_data[person_features])

# 回归分析
X = merged_data[person_features] 
X = sm.add_constant(X)  # 添加常数项
y = merged_data['acc']
model = sm.OLS(y, X).fit()# 指定了使用普通最小二乘法基于给定的自变量 X 和因变量 y 来构建模型
print(model.summary())

import matplotlib.font_manager as fm
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数显示问题
# 相关性分析
corr = merged_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

from sklearn.decomposition import PCA
# PCA降维
pca = PCA(n_components=2)
pca_result = pca.fit_transform(merged_data.iloc[:, :-4])  # 除去最后四列（person_id, acc, age, height, weight）
merged_data['pca1'] = pca_result[:, 0]
merged_data['pca2'] = pca_result[:, 1]

# 绘制PCA结果
plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='实验人员编号', data=merged_data, palette='tab10')
plt.title('PCA of Sensor Data')
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 准备数据
X = merged_data.iloc[:, :-6]  # 除去最后四列（person_id, acc, age, height, weight）
y = merged_data['实验人员编号']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建SVM分类模型
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)

# 评价模型
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

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
base_folder_path = 'D:/desktop/模拟论文1/附件5'
file_prefix = 'a'
file_prefix1 = 't'
file_suffix = '.xlsx'

# 存储所有特征数据的列表
all_features = []

# 遍历所有实验人员
for person_id in range(1, 6):  
    folder_path = os.path.join(base_folder_path, f'unknow{person_id}')
    
    # 遍历该实验人员的所有文件，提取特征
    for i in range(1, 13):
        for j in range(1,2):
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

unknown_feature_data = pd.DataFrame(all_features)
unknown_feature_data.to_csv('unknown_activity_data.csv', index=None)

unknown_feature_data = pd.read_csv('unknown_activity_data.csv')  # 假设未知数据存放在此文件中
# 数据清洗和归一化
scaler = StandardScaler()
# 计算总体加速度
data_scaled = scaler.fit_transform(unknown_feature_data.iloc[:, :-2])
# 将标准化后的数据转换回 DataFrame
unknown_data_scaled_df = pd.DataFrame(data_scaled, columns=feature_data.columns[:-2])

# 计算总体加速度
unknown_data_scaled_df['acc'] = np.sqrt(unknown_data_scaled_df['acc_x(g)_mean']**2 + unknown_data_scaled_df['acc_y(g)_mean']**2 + unknown_data_scaled_df['acc_z(g)_mean']**2)

# 对未知数据进行预测
predicted_person = svm_model.predict(unknown_data_scaled_df)
for index, i in enumerate(predicted_person):
    if i <4:# 限定是实验2的10个实验人员
        predicted_person[index] =4
print(predicted_person)

# 统计每12次实验中出现次数最多的实验人员
predicted_person_chunks = [predicted_person[i:i + 12] for i in range(0, len(predicted_person), 12)]
most_common_persons = [Counter(chunk).most_common(1)[0][0] for chunk in predicted_person_chunks]

# 创建包含活动类型的数据框
activity_types = ['Unkonw1', 'Unknow2', 'Unknow3', 'Unknow4', 'Unknow5']
results_df = pd.DataFrame({'活动类型': activity_types, '判别结果': most_common_persons})
print(results_df)

# 保存结果为excel文件
results_df.to_excel('question3_results.xlsx', index=False)