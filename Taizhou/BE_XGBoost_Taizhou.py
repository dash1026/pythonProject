from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from bayes_opt import BayesianOptimization

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr



# 定义文件路径
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

# 定义XGBoost训练过程的评估函数
def xgb_cv(n_estimators, max_depth, gamma, min_child_weight, subsample, colsample_bytree, learning_rate):
    # 模型参数
    params = {
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
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
    'n_estimators': (50, 1000),
    'learning_rate': (0.01, 0.05),
    'max_depth': (3, 10),
    'gamma': (0, 1),
    'min_child_weight': (0, 10),
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
optimizer.maximize(init_points=5, n_iter=125)

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
output_dir = 'Taizhou_outputdata'

# 检查目录是否存在，如果不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 在指定的目录下保存CSV文件
df_pred_day1.to_csv(os.path.join(output_dir, 'BEXG_predicted_values1.csv'), index=False)
df_pred_day2.to_csv(os.path.join(output_dir, 'BEXG_predicted_values2.csv'), index=False)


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

