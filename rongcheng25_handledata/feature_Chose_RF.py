from math import sqrt

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

from sklearn.ensemble import RandomForestRegressor

# 初始化随机森林回归器
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train_scaled, y_train)
y_pred_temp = rf.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred_temp)
rmse = sqrt(mse)
print(f'all feature R Mean Squared Error: {rmse}')

# 获取特征重要性
feature_importances = rf.feature_importances_

# 将特征重要性转换为Pandas的DataFrame以便于观察
features = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(features)

# 假设我们决定只保留重要性前80%的特征
important_features = features[features['Importance'] > features['Importance'].quantile(0.2)]

# 更新训练和测试数据集
X_train_optimized = X_train[important_features['Feature']]
X_test_optimized = X_test[important_features['Feature']]

# 可以选择再次使用随机森林或其他模型进行训练，以评估特征优化的效果
rf_optimized = RandomForestRegressor(n_estimators=100, random_state=42)
rf_optimized.fit(X_train_optimized, y_train)

from sklearn.metrics import mean_squared_error

# 预测并计算MSE
y_pred = rf_optimized.predict(X_test_optimized)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f'R Mean Squared Error: {rmse}')
