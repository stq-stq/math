import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
#所有文件缺失值检测 

# parentFolder = r'D:\desktop\模拟论文1'  # 父文件夹路径

# # 第一级
# subfolders = [f for f in os.listdir(parentFolder) if os.path.isdir(os.path.join(parentFolder, f))]

# for i, subfolderName in enumerate(subfolders):
#     subfolderPath = os.path.join(parentFolder, subfolderName)

#     # 第二级
#     innerSubfolders = [f for f in os.listdir(subfolderPath) if os.path.isdir(os.path.join(subfolderPath, f))]

#     for j, innerSubfolderName in enumerate(innerSubfolders):
#         innerSubfolderPath = os.path.join(subfolderPath, innerSubfolderName)

#         # 获取 xlsx 文件
#         files = [f for f in os.listdir(innerSubfolderPath) if f.endswith('.xlsx')]

#         # 遍历 xlsx 文件
#         miss = 0
#         for k, fileName in enumerate(files):
#             filePath = os.path.join(innerSubfolderPath, fileName)

#             data = pd.read_excel(filePath)

#             # 查找缺失值
#             for col in range(data.shape[1]):
#                 columnData = data.iloc[:, col]
#                 missingIndices = columnData[columnData.isnull()].index
            
#             if len(missingIndices)!= 0:
#                 missing = '有'
#                 miss = 1
#                 print(f'{subfolderName} 文件下 {innerSubfolderName} 下 {fileName} 表格：{missing}缺失值')
#                 print('第', col, '列缺失值的索引：', missingIndices)
#             else:
#                 missing = '无'

#             #异常值
#             # 处理后文件要保存的文件夹路径
#             save_folder_path = 'D:\\desktop\\data\\'+ subfolderName +'\\' + innerSubfolderName +''

#             # 数据异常值检测
#             x = data.iloc[1:, 0]
#             y = data.iloc[1:, 1]
#             z = data.iloc[1:, 2]
#             x1 = data.iloc[1:, 3]
#             y1 = data.iloc[1:, 4]
#             z1 = data.iloc[1:, 5]
#             def Zscore_outlier(df):
#                 m = np.mean(df)
#                 sd = np.std(df)
#                 # out=[]
#                 # row_indices = []
#                 for i, value in enumerate(df): 
#                     z = (value-m)/sd
#                     if np.abs(z) > 3: 
#                         # out.append(value)
#                         # row_indices.append(i)
#                         return False
#                 # print("异常值:",out,'\n',"索引为：",row_indices)
#                 return True

#             def Winsorization_outliers(df):
#                 if Zscore_outlier(df):
#                     return []
#                 q1 = np.percentile(df , 0.5)
#                 q3 = np.percentile(df , 99.5)
#                 out=[]
#                 row_indices = []
#                 for i, value in enumerate(df):
#                     if value > q3 or value < q1:
#                         out.append(value)
#                         row_indices.append(i)
#                 # print("异常值:",out,'\n',"索引为：",row_indices)
#                 return row_indices
#             arr=Winsorization_outliers(x)+Winsorization_outliers(y)+Winsorization_outliers(z)+Winsorization_outliers(x1)+Winsorization_outliers(y1)+Winsorization_outliers(z1)
#             dataNew=data.drop(arr,axis=0)

#             # 标准化数据
#             from sklearn.preprocessing import StandardScaler
#             scaler = StandardScaler()
#             dataNew = pd.DataFrame(scaler.fit_transform(dataNew), columns=dataNew.columns)
            
#             filtered_file_path = os.path.join(save_folder_path, 'filtered_' + fileName)
#             dataNew.to_excel(filtered_file_path, index=False)

#             # # 绘制箱线图
#             # plt.figure()
#             # plt.boxplot(Data)
#             # plt.xlabel('Data Group')
#             # plt.ylabel('Value')
#             # plt.title('Box Plot of Six Datasets')
#             # plt.show()

#         if miss == 0:
#             print(f'{subfolderName} 文件下 {innerSubfolderName}无缺失值')

# # 绘制三维散点图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# 箱线图
# filePath = r'D:\desktop\模拟论文1\附件1\person1\SY1.xlsx'
# Data = pd.read_excel(filePath)
# data1 = Data.iloc[1:, 0]
# data2 = Data.iloc[1:, 1]
# data3 = Data.iloc[1:, 2]
# data4 = Data.iloc[1:, 3]
# data5 = Data.iloc[1:, 4]
# data6 = Data.iloc[1:, 5]
# data = [data1, data2, data3, data4, data5, data6]
# plt.boxplot(data)
# plt.xlabel('Data Group')
# plt.ylabel('Value')
# plt.title('Box Plot of Six Datasets')
# plt.show()

# 柱状图
# filePath = r'D:\desktop\模拟论文1\附件1\person1\SY60.xlsx'
# data = pd.read_excel(filePath)
# data1 = data.iloc[1:, 0]
# data2 = data.iloc[1:, 1]
# data3 = data.iloc[1:, 2]
# data4 = data.iloc[1:, 3]
# data5 = data.iloc[1:, 4]
# data6 = data.iloc[1:, 5]
# Data = [data1, data2, data3, data4, data5, data6]
# for i in range(6):
#     min_value = Data[i].min()
#     max_value = Data[i].max()
#     interval = 0.5  # 区间大小
#     counts, bins = np.histogram(Data[i], bins=np.arange(min_value, max_value + interval, interval))
#     plt.figure()
#     plt.bar(bins[:-1], counts, width=interval, align='edge')
#     plt.xlabel('区间')
#     plt.ylabel('数量')
#     plt.title('数据在不同区间的数量分布')
#     plt.show()

# # 文件夹内异常值处理
# folder_path = r'D:\desktop\模拟论文1\附件1\person1'
# # 处理后文件要保存的文件夹路径
# save_folder_path = 'D:\desktop\附件1\person1'

#  for file_name in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file_name)
#     data = pd.read_excel(file_path)

    # # 标准化数据
    # mean = np.mean(data, axis=0)
    # std = np.std(data, axis=0)
    # standardized_data = (data - mean) / std
    # # 将处理后的数据保存回 Excel 文件
    # standardized_data.to_excel(file_path, index=False)

    # 数据异常值检测
    # x = data.iloc[1:, 0]
    # y = data.iloc[1:, 1]
    # z = data.iloc[1:, 2]
    # x1 = data.iloc[1:, 3]
    # y1 = data.iloc[1:, 4]
    # z1 = data.iloc[1:, 5]
    # def Zscore_outlier(df):
    #     m = np.mean(df)
    #     sd = np.std(df)
    #     out=[]
    #     row_indices = []
    #     for i, value in enumerate(df): 
    #         z = (value-m)/sd
    #         if np.abs(z) > 3: 
    #             out.append(value)
    #             row_indices.append(i)
    #     # print("异常值:",out,'\n',"索引为：",row_indices)
    #     return row_indices

    # def Winsorization_outliers(df):
    #     q1 = np.percentile(df , 0.5)
    #     q3 = np.percentile(df , 99.5)
    #     out=[]
    #     row_indices = []
    #     for i, value in enumerate(df):
    #         if value > q3 or value < q1:
    #             out.append(value)
    #             row_indices.append(i)
    #     # print("异常值:",out,'\n',"索引为：",row_indices)
    #     return row_indices
    # dataNew=data.drop(Winsorization_outliers(x)+Winsorization_outliers(y)+Winsorization_outliers(z)+Winsorization_outliers(x1)+Winsorization_outliers(y1)+Winsorization_outliers(z1),axis=0)
    # # 构建处理后文件在新路径下的保存路径（可以根据需要修改保存文件名）
    # filtered_file_path = os.path.join(save_folder_path, 'filtered_' + file_name)
    # dataNew.to_excel(filtered_file_path, index=False)

# # 主成分分析
folder_path = r'D:\desktop\data\附件1\person1'
# 处理后文件要保存的文件夹路径
save_folder_path = 'D:\desktop\模拟论文1\data'

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_excel(file_path)
    colo = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    x = 50 * np.random.rand(100, 3)
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig,  elev=30, azim=20)
    # 这段文本是关于在 Python 编程语言中使用 matplotlib 库创建图形和三维坐标轴的相关代码描述。其中，“fig = plt.figure(figsize=(12, 8))”表示创建一个图形对象，并指定其大小为宽 12 高 8 。“ax = Axes3D(fig,  elev=30, azim=20)”则是在已创建的图形对象上创建一个三维坐标轴对象，同时设置了其仰角为 30 度，方位角为 20 度。 

    shape = x.shape
    sse = []
    score = []
    K = 12 # 分为K类
    for k in [K]:
        clf = KMeans(n_clusters=k)
        clf.fit(x)
        sse.append(clf.inertia_)#惯性值
        lab = clf.fit_predict(x)#预测
        score.append(silhouette_score(x, clf.labels_, metric='euclidean'))#轮廓系数
        for i in range(shape[0]):
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('k=' + str(k))
            ax.scatter(x[i, 1],x[i, 2], x[i, 3], c=colo[lab[i]])
        plt.show()
