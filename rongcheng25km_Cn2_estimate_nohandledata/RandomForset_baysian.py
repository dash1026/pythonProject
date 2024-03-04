from math import sqrt

import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV, cross_val_score
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

# 定义随机森林的性能评估函数
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    estimator = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        max_features=min(max_features, 0.999), # 因为max_features是比例
        max_depth=int(max_depth),
        random_state=2
    )
    cval = cross_val_score(estimator, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return cval.mean()

# 定义贝叶斯优化的参数空间
params = {
    'n_estimators': (10, 250),
    'min_samples_split': (2, 25),
    'max_features': (0.1, 0.999),
    'max_depth': (5, 50)
}

# 使用贝叶斯优化
optimizer = BayesianOptimization(f=rf_cv, pbounds=params, random_state=1)
optimizer.maximize(n_iter=25, init_points=5)

# 打印最优参数
print("最优参数:", optimizer.max['params'])

# 提取最优参数
params = optimizer.max['params']
params['n_estimators'] = int(params['n_estimators'])
params['min_samples_split'] = int(params['min_samples_split'])
params['max_depth'] = int(params['max_depth'])
# 注意，max_features已经是在0到1之间了，不需要转换

# 使用最优参数创建随机森林回归器
rf_best = RandomForestRegressor(**params)

# 使用全部训练数据训练模型
rf_best.fit(X_train, y_train)

# 现在rf_best就是使用最优参数训练出的模型，可以用于预测或进一步分析

y_pred = rf_best.predict(X_test)

# 评估预测性能，这里可以使用适合回归问题的评估指标，如RMSE
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"测试集上的RMSE: {rmse}")

# 在测试集上进行预测

y_pred_day1 = rf_best.predict(X_test_day1)
y_pred_day2 = rf_best.predict(X_test_day2)
y_pred_day3 = rf_best.predict(X_test_day3)
y_pred_day4 = rf_best.predict(X_test_day4)

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

