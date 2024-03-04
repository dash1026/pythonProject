from math import sqrt

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

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

# 设置参数网格
param_grid = {
    'n_estimators': [100, 200, 300], # 树的数量
    'max_depth': [None, 10, 20, 30], # 最大深度
    'min_samples_split': [2, 5, 10], # 分裂内部节点所需的最小样本数
    'min_samples_leaf': [1, 2, 4], # 在叶节点处需要的最小样本数
    'bootstrap': [True, False] # 是否使用bootstrap样本
}

# 创建随机森林模型
rf = RandomForestRegressor()

# 使用网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# 训练模型
grid_search.fit(X_train, y_train)

# 打印最佳参数
print("最佳参数:", grid_search.best_params_)

# 使用最佳参数的模型进行预测
best_grid = grid_search.best_estimator_
predictions = best_grid.predict(X_test)

# 评估预测性能，这里可以使用适合回归问题的评估指标，如RMSE
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"测试集上的RMSE: {rmse}")

# 在测试集上进行预测

y_pred_day1 = best_grid.predict(X_test_day1)
y_pred_day2 = best_grid.predict(X_test_day2)
y_pred_day3 = best_grid.predict(X_test_day3)
y_pred_day4 = best_grid.predict(X_test_day4)

# 评估模型性能
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

rmse_day1 = sqrt(mse_day1)
rmse_day2 = sqrt(mse_day2)
rmse_day3 = sqrt(mse_day3)
rmse_day4 = sqrt(mse_day4)


print(f"Test RMSEday1: {rmse_day1}")
print(f"Test RMSEday1: {rmse_day2}")
print(f"Test RMSEday1: {rmse_day3}")
print(f"Test RMSEday1: {rmse_day4}")

