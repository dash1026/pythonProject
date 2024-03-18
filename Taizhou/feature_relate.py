import seaborn as sns
import matplotlib.pyplot as plt

# 假设df是一个Pandas DataFrame，其中包含你的数据
# 下面我将创建一个示例DataFrame，你应该用你自己的数据替代它
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# 定义文件路径
# 定义文件路径
file_path_Xtrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTrain.csv'
file_path_ytrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTrain.csv'
file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest.csv'

file_path_X_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\XTest_Value_day002.csv'
file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day002.csv'

# 使用pandas的read_csv函数读取数据
X_train = pd.read_csv(file_path_Xtrain)
y_train = pd.read_csv(file_path_ytrain)
X_test = pd.read_csv(file_path_Xtest)
y_test = pd.read_csv(file_path_ytest)

X_test_day1 = pd.read_csv(file_path_X_test_day1)
y_test_day1 = pd.read_csv(file_path_y_test_day1)
X_test_day2 = pd.read_csv(file_path_X_test_day2)
y_test_day2 = pd.read_csv(file_path_y_test_day2)


y_train = y_train.values.ravel()

# 对数据进行预处理，将Nan的使用该列的下一个非nan代替
X_train = X_train.fillna(method='bfill')
X_test = X_test.fillna(method='bfill')


# 初始化MinMaxScaler
scaler = MinMaxScaler()

# 使用X_train拟合scaler，然后转换X_train
X_train_scaled = scaler.fit_transform(X_train)

# 使用相同的scaler转换所有测试集
X_test_scaled = scaler.transform(X_test)
X_test_day1_scaled = scaler.transform(X_test_day1)
X_test_day2_scaled = scaler.transform(X_test_day2)


#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 假定这里已经完成了数据的读取和缩放

# 定义你想要保留的特征列表
selected_features_names = ['P', 'T', 'H', 'WS', 'WD', 'RH', 'WShear', 'TShear']

# 从原始特征DataFrame中选择这些特征
X_train_selected = X_train[selected_features_names]
X_test_selected = X_test[selected_features_names]

# 使用MinMaxScaler
scaler = MinMaxScaler()

# 对选定的特征进行缩放
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# 将缩放后的数组转换回DataFrame以便绘图
X_train_selected_df = pd.DataFrame(X_train_selected_scaled, columns=selected_features_names)
X_test_selected_df = pd.DataFrame(X_test_selected_scaled, columns=selected_features_names)

# 合并训练集和测试集以获得完整的数据集
data_selected = pd.concat([X_train_selected_df, X_test_selected_df], axis=0, ignore_index=True)

# 计算相关系数矩阵
corr_matrix_selected = data_selected.corr()

# 绘制热图
plt.figure(figsize=(10, 8))
ax = sns.heatmap(corr_matrix_selected,
                 cmap='coolwarm',
                 annot=True,
                 xticklabels=selected_features_names,
                 yticklabels=selected_features_names)

# 设置坐标轴标签的字体大小
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize='large')
ax.set_yticklabels(ax.get_yticklabels(), fontsize='large')

# 设置坐标轴的名称
plt.xlabel('Feature', fontsize='large')
plt.ylabel('Feature', fontsize='large')

plt.show()
