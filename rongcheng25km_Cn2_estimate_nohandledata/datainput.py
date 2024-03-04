import pandas as pd
import os

# 导入测试数据
# 定义存放CSV文件的文件夹路径
folder_path = 'D:\data\荣成\荣成25km\Test'

# 获取文件夹中所有CSV文件的路径
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 初始化一个空列表来存储数据
dataframes_Test = []

# 遍历CSV文件路径，读取数据，并添加到列表中
for file in csv_files:
    file_path = os.path.join(folder_path, file)  # 构造完整的文件路径
    df = pd.read_csv(file_path, encoding='gbk')  # 读取CSV文件
    dataframes_Test.append(df)  # 将DataFrame添加到列表中

Test_vertical = pd.concat(dataframes_Test, axis=0)


X_test = Test_vertical.iloc[:, 1:7]
y_test = Test_vertical.iloc[:, 7]

print(X_test)
print(y_test)