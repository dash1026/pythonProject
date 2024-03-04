from math import sqrt

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# 定义文件路径
file_path_feature = 'D:\\data\\Python_data\\rongcheng_six_all.csv'
file_path_label = 'D:\\data\\Python_data\\rongcheng_one_all.csv'

# 使用pandas的read_csv函数读取数据
data_feature = pd.read_csv(file_path_feature)
data_label = pd.read_csv(file_path_label)

# # 查看前几行数据以确认正确导入
# print(data_feature.head())
# print(data_label.head())

X_train = data_feature.iloc[:4606]
y_train = data_label.iloc[:4606]
X_test = data_feature.iloc[4606:]
y_test = data_label.iloc[4606:]
X_test_day1 = data_feature.iloc[4606:4911]
y_test_day1 = data_label.iloc[4606:4911]
X_test_day2 = data_feature.iloc[4911:5197]
y_test_day2 = data_label.iloc[4911:5197]
X_test_day3 = data_feature.iloc[5197:5501]
y_test_day3 = data_label.iloc[5197:5501]
X_test_day4 = data_feature.iloc[5501:]
y_test_day4 = data_label.iloc[5501:]

y_train = y_train.values.ravel()
# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正则化参数
    'gamma': ['scale', 'auto'],  # 核函数参数
    'epsilon': [0.01, 0.1, 0.2, 0.5],  # SVR中的epsilon参数
}

# 创建SVR实例
svr = SVR(kernel='rbf')

# 使用网格搜索进行参数优化
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# 查看最优参数和最优分数
print("Best Parameters:", grid_search.best_params_)
print("Best Score (MSE):", -grid_search.best_score_)

# 使用最优参数的模型进行预测
best_svr = grid_search.best_estimator_
y_pred = best_svr.predict(X_test)

# 计算和打印MSE和RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse}, RMSE: {rmse}")

# 使用最佳参数的模型进行预测
best_grid = grid_search.best_estimator_
predictions = best_grid.predict(X_test)

# 评估预测性能
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"测试集上的RMSE: {rmse}")

# 在测试集上进行预测
y_pred = best_grid.predict(X_test)
y_pred_day1 = best_grid.predict(X_test_day1)
y_pred_day2 = best_grid.predict(X_test_day2)
y_pred_day3 = best_grid.predict(X_test_day3)
y_pred_day4 = best_grid.predict(X_test_day4)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mse_day1 = mean_squared_error(y_test_day1, y_pred_day1)
mse_day2 = mean_squared_error(y_test_day2, y_pred_day2)
mse_day3 = mean_squared_error(y_test_day3, y_pred_day3)
mse_day4 = mean_squared_error(y_test_day4, y_pred_day4)



# 将y_pred转换为DataFrame
df_pred_day1 = pd.DataFrame(y_pred_day1, columns=['Predicted Values'])
df_pred_day2 = pd.DataFrame(y_pred_day2, columns=['Predicted Values'])
df_pred_day3 = pd.DataFrame(y_pred_day3, columns=['Predicted Values'])
df_pred_day4 = pd.DataFrame(y_pred_day4, columns=['Predicted Values'])
# 将DataFrame保存为CSV文件
df_pred_day1.to_csv('predicted_values1.csv', index=False)
df_pred_day2.to_csv('predicted_values2.csv', index=False)
df_pred_day3.to_csv('predicted_values3.csv', index=False)
df_pred_day4.to_csv('predicted_values4.csv', index=False)

print("预测结果已经保存到predicted_values.csv")

rmse = sqrt(mse)
rmse_day1 = sqrt(mse_day1)
rmse_day2 = sqrt(mse_day2)
rmse_day3 = sqrt(mse_day3)
rmse_day4 = sqrt(mse_day4)



# # 使用numpy的corrcoef函数计算相关系数矩阵
# corr_matrix_day1 = np.corrcoef(y_test_day1, y_pred_day1)
# corr_matrix_day2 = np.corrcoef(y_test_day2, y_pred_day2)
# corr_matrix_day3 = np.corrcoef(y_test_day3, y_pred_day3)
# corr_matrix_day4 = np.corrcoef(y_test_day4, y_pred_day4)
#
# # 相关系数矩阵中的[0, 1]或[1, 0]元素是y_test和y_pred之间的相关系数
# correlation1 = corr_matrix_day1[0, 1]
# correlation2 = corr_matrix_day2[0, 1]
# correlation3 = corr_matrix_day3[0, 1]
# correlation4 = corr_matrix_day4[0, 1]
#
# print(f"y_test和y_pred之间的相关性为: {correlation1}")
# print(f"y_test和y_pred之间的相关性为: {correlation2}")
# print(f"y_test和y_pred之间的相关性为: {correlation3}")
# print(f"y_test和y_pred之间的相关性为: {correlation4}")


print(f"Test RMSE: {rmse}")
print(f"Test RMSE: {rmse_day1}")
print(f"Test RMSE: {rmse_day2}")
print(f"Test RMSE: {rmse_day3}")
print(f"Test RMSE: {rmse_day4}")