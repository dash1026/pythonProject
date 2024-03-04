from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from bayes_opt import BayesianOptimization

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


# print(X_test_day4)
# # 载入数据
# boston = load_boston()
# X, y = boston.data, boston.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义XGBoost训练过程的评估函数
def xgb_cv(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree):
    # 模型参数
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'eta': 0.1,
        'objective': 'reg:squarederror',
    }
    # 使用交叉验证的方式评估模型性能
    xgb_model = xgb.XGBRegressor(**params)
    cv_score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    return cv_score

# 使用贝叶斯优化库定义超参数空间
pbounds = {
    'n_estimators': (100, 1000),
    'max_depth': (3, 10),
    'gamma': (0, 1),
    'min_child_weight': (0, 5),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
}

# 初始化贝叶斯优化模型，并设置优化的目标函数和参数空间
optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=42,
)

# 执行优化过程
optimizer.maximize(init_points=5, n_iter=25)

# 输出最优的模型参数
print(optimizer.max)

best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])

# 使用最优超参数重新训练模型
model = xgb.XGBRegressor(**best_params)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)
y_pred_day1 = model.predict(X_test_day1)
y_pred_day2 = model.predict(X_test_day2)
y_pred_day3 = model.predict(X_test_day3)
y_pred_day4 = model.predict(X_test_day4)

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

# 使用模型进行预测的示例
# 这里的 y_pred 就是模型对测试集的预测结果
