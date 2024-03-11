import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# [在此处插入数据加载、预处理的代码]
# 定义文件路径
file_path_Xtrain = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTrain.csv'
file_path_ytrain = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTrain.csv'
file_path_Xtest = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest.csv'
file_path_ytest = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest.csv'



file_path_X_test_day1 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day001.csv'
file_path_y_test_day1  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day002.csv'
file_path_y_test_day2  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day002.csv'
file_path_X_test_day3 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day003.csv'
file_path_y_test_day3  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day003.csv'
file_path_X_test_day4 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day004.csv'
file_path_y_test_day4  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day004.csv'

# 使用pandas的read_csv函数读取数据
X_train = pd.read_csv(file_path_Xtrain)
y_train = pd.read_csv(file_path_ytrain)
X_test = pd.read_csv(file_path_Xtest)
y_test = pd.read_csv(file_path_ytest)

X_test_day1 = pd.read_csv(file_path_X_test_day1)
y_test_day1 = pd.read_csv(file_path_y_test_day1)
X_test_day2 = pd.read_csv(file_path_X_test_day2)
y_test_day2 = pd.read_csv(file_path_y_test_day2)
X_test_day3 = pd.read_csv(file_path_X_test_day3)
y_test_day3 = pd.read_csv(file_path_y_test_day3)
X_test_day4 = pd.read_csv(file_path_X_test_day4)
y_test_day4 = pd.read_csv(file_path_y_test_day4)

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
X_test_day3_scaled = scaler.transform(X_test_day3)
X_test_day4_scaled = scaler.transform(X_test_day4)

# 现在，所有的数据集都已经按照X_train的尺度进行了归一化

# 初始化MinMaxScaler和随机森林回归器
scaler = MinMaxScaler()
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# 对数据进行缩放
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 应用序列特征选择
sfs = SFS(rf_regressor,
          k_features=7,  # 你想要的特征数量
          forward=True,  # 向前选择
          floating=False,  # 不使用浮动选择
          scoring='neg_mean_squared_error',  # 用于评价的指标
          cv=0)  # 不使用交叉验证

# 使用训练数据拟合SFS
sfs = sfs.fit(X_train_scaled, y_train)

# 打印选定的特征：
selected_features_names = list(sfs.k_feature_names_)
print('Selected features:', selected_features_names)

# 使用选择的特征来更新训练和测试数据集
X_train_optimized = X_train_scaled[:, sfs.k_feature_idx_]
X_test_optimized = X_test_scaled[:, sfs.k_feature_idx_]

# 重新训练随机森林回归器
rf_regressor.fit(X_train_optimized, y_train)

# 预测并计算MSE
y_pred = rf_regressor.predict(X_test_optimized)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f'R Mean Squared Error with selected features: {rmse}')
