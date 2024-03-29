from math import sqrt

import pandas as pd
from bayes_opt import BayesianOptimization
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# 定义文件路径
file_path_Xtrain = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTrain.csv'
file_path_ytrain = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTrain.csv'
file_path_Xtest = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest.csv'
file_path_ytest = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest.csv'

file_path_X_test_day1 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day002.csv'
file_path_y_test_day2 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day002.csv'
file_path_X_test_day3 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day003.csv'
file_path_y_test_day3 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day003.csv'
file_path_X_test_day4 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day004.csv'
file_path_y_test_day4 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day004.csv'

selected_features_names = ['H', 'WS', 'PT', 'RH', 'WD', 'TShear']
# 使用pandas的read_csv函数读取数据
X_train = pd.read_csv(file_path_Xtrain)[selected_features_names]
y_train = pd.read_csv(file_path_ytrain)
X_test = pd.read_csv(file_path_Xtest)[selected_features_names]
y_test = pd.read_csv(file_path_ytest)

X_test_day1 = pd.read_csv(file_path_X_test_day1)[selected_features_names]
y_test_day1 = pd.read_csv(file_path_y_test_day1)
X_test_day2 = pd.read_csv(file_path_X_test_day2)[selected_features_names]
y_test_day2 = pd.read_csv(file_path_y_test_day2)
X_test_day3 = pd.read_csv(file_path_X_test_day3)[selected_features_names]
y_test_day3 = pd.read_csv(file_path_y_test_day3)
X_test_day4 = pd.read_csv(file_path_X_test_day4)[selected_features_names]
y_test_day4 = pd.read_csv(file_path_y_test_day4)

y_train = y_train.values.ravel()
# 使用.ravel()方法将y_test转换为一维数组
y_test = y_test.values.ravel()
y_test_day1 = y_test_day1.values.ravel()
y_test_day2 = y_test_day2.values.ravel()
y_test_day3 = y_test_day3.values.ravel()
y_test_day4 = y_test_day4.values.ravel()

# 对数据进行预处理，将Nan的使用该列的下一个非nan代替
X_train = X_train.fillna(method='bfill')
X_test = X_test.fillna(method='bfill')


# 定义随机森林的性能评估函数
def rf_cv(n_estimators, min_samples_leaf, max_features, max_depth):
    estimator = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(min_samples_leaf),
        max_features=min(max_features, 0.999),  # 因为max_features是比例
        max_depth=int(max_depth),
        random_state=42
    )
    cval = cross_val_score(estimator, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return cval.mean()


# 定义贝叶斯优化的参数空间

params = {
    'n_estimators': (25, 300),
    'max_depth': (5, 50),
    'max_features': (0.1, 0.999),
    'min_samples_leaf': (1, 10),  # 在叶节点处需要的最小样本数

}
# 使用贝叶斯优化
optimizer = BayesianOptimization(f=rf_cv, pbounds=params, random_state=42)
optimizer.maximize(n_iter=25, init_points=5)

# 打印最优参数
print("最优参数:", optimizer.max['params'])

# 提取最优参数
params = optimizer.max['params']
params['n_estimators'] = int(params['n_estimators'])
params['min_samples_leaf'] = int(params['min_samples_leaf'])
params['max_depth'] = int(params['max_depth'])
# 注意，max_features已经是在0到1之间了，不需要转换

# 使用最优参数创建随机森林回归器
rf_best = RandomForestRegressor(**params)

# 使用全部训练数据训练模型
rf_best.fit(X_train, y_train)

# 现在rf_best就是使用最优参数训练出的模型，可以用于预测或进一步分析

y_pred = rf_best.predict(X_test)

# 评估预测性能，这里可以使用适合回归问题的评估指标，如RMSE
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

import os

# 定义目录路径
output_dir = 'rongcheng_outputdata'

# 检查目录是否存在，如果不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 在指定的目录下保存CSV文件
df_pred_day1.to_csv(os.path.join(output_dir, 'BERF_predicted_values1.csv'), index=False)
df_pred_day2.to_csv(os.path.join(output_dir, 'BERF_predicted_values2.csv'), index=False)
df_pred_day3.to_csv(os.path.join(output_dir, 'BERF_predicted_values3.csv'), index=False)
df_pred_day4.to_csv(os.path.join(output_dir, 'BERF_predicted_values4.csv'), index=False)

print("预测结果已经保存到指定的目录中。")


# RMSE
rmse = sqrt(mse)
rmse_day1 = sqrt(mse_day1)
rmse_day2 = sqrt(mse_day2)
rmse_day3 = sqrt(mse_day3)
rmse_day4 = sqrt(mse_day4)


# MAD
mad = mean_absolute_error(y_test, y_pred)
mad_day1 = mean_absolute_error(y_test_day1, y_pred_day1)
mad_day2 = mean_absolute_error(y_test_day2, y_pred_day2)
mad_day3 = mean_absolute_error(y_test_day3, y_pred_day3)
mad_day4 = mean_absolute_error(y_test_day4, y_pred_day4)

# Pearson correlation coefficient
correlation, _ = pearsonr(y_test, y_pred)
correlation_day1, _ = pearsonr(y_test_day1, y_pred_day1)
correlation_day2, _ = pearsonr(y_test_day2, y_pred_day2)
correlation_day3, _ = pearsonr(y_test_day3, y_pred_day3)
correlation_day4, _ = pearsonr(y_test_day4, y_pred_day4)

# Print the results
# 输出统计量，保留四位小数
print(f"all Test RMSE: {rmse:.4f}")
print(f"day1 Test RMSE: {rmse_day1:.4f}")
print(f"day2 Test RMSE: {rmse_day2:.4f}")
print(f"day3 Test RMSE: {rmse_day3:.4f}")
print(f"day4 Test RMSE: {rmse_day4:.4f}")

# ...

print(f"Test MAD: {mad:.4f}")
print(f"Day 1 MAD: {mad_day1:.4f}")
print(f"Day 2 MAD: {mad_day2:.4f}")
print(f"Day 3 MAD: {mad_day3:.4f}")
print(f"Day 4 MAD: {mad_day4:.4f}")

print(f"Test Pearson Correlation: {correlation:.4f}")
print(f"Day 1 Pearson Correlation: {correlation_day1:.4f}")
print(f"Day 2 Pearson Correlation: {correlation_day2:.4f}")
print(f"Day 3 Pearson Correlation: {correlation_day3:.4f}")
print(f"Day 4 Pearson Correlation: {correlation_day4:.4f}")
