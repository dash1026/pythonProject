from math import sqrt

import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bayes_opt import BayesianOptimization
import numpy as np



file_path_Xtrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTrain.csv'
file_path_ytrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTrain.csv'
file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest.csv'

file_path_X_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day002.csv'
file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day002.csv'



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


y_train = y_train.values.ravel()
# 使用.ravel()方法将y_test转换为一维数组
y_test = y_test.values.ravel()
y_test_day1 = y_test_day1.values.ravel()
y_test_day2 = y_test_day2.values.ravel()


# 对数据进行预处理，将Nan的使用该列的下一个非nan代替
X_train = X_train.fillna(method='bfill')
X_test = X_test.fillna(method='bfill')

# 初始化MinMaxScaler
scaler = MinMaxScaler()

# 使用X_train拟合scaler，然后转换X_train

X_train_scaled = scaler.fit_transform(X_train)


# 更新训练和测试数据集

X_train_optimized = X_train[selected_features_names]
X_test_optimized = X_test[selected_features_names]

# 使用相同的scaler转换所有测试集
X_test_scaled = scaler.transform(X_test)
X_test_day1_scaled = scaler.transform(X_test_day1)
X_test_day2_scaled = scaler.transform(X_test_day2)


# 现在，所有的数据集都已经按照X_train的尺度进行了归一化
# 定义SVR模型的超参数范围
pbounds = {
    'C': (1, 100),
    'epsilon': (0.01, 1),
    'gamma': (0.01, 1)
}

# 定义优化的目标函数
def svr_cv(C, epsilon, gamma):
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return -mean_squared_error(y_test, y_pred)  # 负MSE作为优化目标

# 初始化贝叶斯优化
optimizer = BayesianOptimization(
    f=svr_cv,
    pbounds=pbounds,
    random_state=42,
)

# 执行优化
optimizer.maximize(
    init_points=5,
    n_iter=128,
)


# 提取最优参数
params_best = optimizer.max['params']
print('Best Parameters:', params_best)

# 使用最优参数训练模型
model_best = SVR(C=params_best['C'], epsilon=params_best['epsilon'], gamma=params_best['gamma'])
model_best.fit(X_train_scaled, y_train)
y_pred = model_best.predict(X_test_scaled)

# 计算并打印最终的MSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse}, RMSE: {rmse}")

# 使用最佳参数的模型进行预测
best_grid = model_best
predictions = best_grid.predict(X_test_scaled)

# 评估预测性能
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f"测试集上的RMSE: {rmse}")

# 在测试集上进行预测
y_pred = best_grid.predict(X_test_scaled)
y_pred_day1 = best_grid.predict(X_test_day1_scaled)
y_pred_day2 = best_grid.predict(X_test_day2_scaled)


# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mse_day1 = mean_squared_error(y_test_day1, y_pred_day1)
mse_day2 = mean_squared_error(y_test_day2, y_pred_day2)




# 将y_pred转换为DataFrame
df_pred_day1 = pd.DataFrame(y_pred_day1, columns=['Predicted Values'])
df_pred_day2 = pd.DataFrame(y_pred_day2, columns=['Predicted Values'])

# 将DataFrame保存为CSV文件

import os

# 定义目录路径
output_dir = 'taizhou_outputdata'

# 检查目录是否存在，如果不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 在指定的目录下保存CSV文件
df_pred_day1.to_csv(os.path.join(output_dir, 'BESVR_predicted_values1.csv'), index=False)
df_pred_day2.to_csv(os.path.join(output_dir, 'BESVR_predicted_values2.csv'), index=False)

print("预测结果已经保存到指定的目录中。")


# RMSE
rmse = sqrt(mse)
rmse_day1 = sqrt(mse_day1)
rmse_day2 = sqrt(mse_day2)



# MAD
mad = mean_absolute_error(y_test, y_pred)
mad_day1 = mean_absolute_error(y_test_day1, y_pred_day1)
mad_day2 = mean_absolute_error(y_test_day2, y_pred_day2)

# Pearson correlation coefficient
correlation, _ = pearsonr(y_test, y_pred)
correlation_day1, _ = pearsonr(y_test_day1, y_pred_day1)
correlation_day2, _ = pearsonr(y_test_day2, y_pred_day2)


# Print the results
# 输出统计量，保留四位小数
print(f"all Test RMSE: {rmse:.4f}")
print(f"day1 Test RMSE: {rmse_day1:.4f}")
print(f"day2 Test RMSE: {rmse_day2:.4f}")

# ...

print(f"Test MAD: {mad:.4f}")
print(f"Day 1 MAD: {mad_day1:.4f}")
print(f"Day 2 MAD: {mad_day2:.4f}")


print(f"Test Pearson Correlation: {correlation:.4f}")
print(f"Day 1 Pearson Correlation: {correlation_day1:.4f}")
print(f"Day 2 Pearson Correlation: {correlation_day2:.4f}")

