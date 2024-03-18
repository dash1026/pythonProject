
from math import sqrt

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


# 定义文件路径
file_path_Xtrain = 'D:\data\台州\Program\Taizhou_pythondata\day\\XTrain.csv'
file_path_ytrain = 'D:\data\台州\Program\Taizhou_pythondata\day\\YTrain.csv'
file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\day\\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\day\\YTest.csv'

file_path_X_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\day\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\day\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\day\\XTest_Value_day002.csv'
file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\day\\YTest_Value_day002.csv'

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

# 对数据进行预处理，将Nan的使用该列的下一个非nan代替
X_train = X_train.fillna(method='bfill')
X_test = X_test.fillna(method='bfill')

# 定义模型
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')

# 定义参数网格
param_grid = {
    'n_estimators': [100, 128],
    'learning_rate': [0.01, 0.02, 0.021, 0.022, 0.023],
    'max_depth': [3, 4, 5],
    'colsample_bytree': [0.3, 0.7, 1.0],
    'subsample': [0.5, 0.7, 0.71, 0.72, 0.73, 0.8]
}

# 设置网格搜索
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)

# 训练模型
grid_search.fit(X_train, y_train)

# 打印最佳参数
print("最佳参数:", grid_search.best_params_)

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

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mse_day1 = mean_squared_error(y_test_day1, y_pred_day1)
mse_day2 = mean_squared_error(y_test_day2, y_pred_day2)




# 将y_pred转换为DataFrame
df_pred_day1 = pd.DataFrame(y_pred_day1, columns=['Predicted Values'])
df_pred_day2 = pd.DataFrame(y_pred_day2, columns=['Predicted Values'])


# 将DataFrame保存为CSV文件
df_pred_day1.to_csv('predicted_values1.csv', index=False)
df_pred_day2.to_csv('predicted_values2.csv', index=False)


print("预测结果已经保存到predicted_values.csv")

rmse = sqrt(mse)
rmse_day1 = sqrt(mse_day1)
rmse_day2 = sqrt(mse_day2)



print(f"Test RMSE: {rmse}")
print(f"Test RMSE: {rmse_day1}")
print(f"Test RMSE: {rmse_day2}")


from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression  # 作为元模型


# 随机森林的参数网格
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用GridSearchCV优化随机森林参数
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(),
                              param_grid=param_grid_rf,
                              scoring='neg_mean_squared_error',
                              cv=5,
                              verbose=2,
                              n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# 打印最佳参数
print("随机森林最佳参数:", grid_search_rf.best_params_)

# 获取最佳随机森林模型
best_rf = grid_search_rf.best_estimator_

# 定义堆叠模型，包括优化后的随机森林和XGBoost
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', best_grid),  # 使用之前优化的XGBoost模型
        ('rf', best_rf)      # 使用优化的随机森林模型
    ],
    final_estimator=LinearRegression(),  # 使用线性回归作为元模型
    cv=5  # 使用5折交叉验证
)
# 训练堆叠模型
stacked_model.fit(X_train, y_train)

# 使用堆叠模型进行预测和评估
stacked_predictions = stacked_model.predict(X_test)
stacked_rmse = np.sqrt(mean_squared_error(y_test, stacked_predictions))
print(f"堆叠模型在测试集上的RMSE: {stacked_rmse}")


# 对不同时间段的数据使用堆叠模型进行预测
stacked_pred_day1 = stacked_model.predict(X_test_day1)
stacked_pred_day2 = stacked_model.predict(X_test_day2)



# 评估各时间段堆叠模型的性能
stacked_rmse_day1 = np.sqrt(mean_squared_error(y_test_day1, stacked_pred_day1))
stacked_rmse_day2 = np.sqrt(mean_squared_error(y_test_day2, stacked_pred_day2))


print(f"堆叠模型在第一天的测试集上的RMSE: {stacked_rmse_day1}")
print(f"堆叠模型在第二天的测试集上的RMSE: {stacked_rmse_day2}")


# 将堆叠模型的预测结果转换为DataFrames
df_stacked_pred_day1 = pd.DataFrame(stacked_pred_day1, columns=['Stacked Predictions Day 1'])
df_stacked_pred_day2 = pd.DataFrame(stacked_pred_day2, columns=['Stacked Predictions Day 2'])


# 将DataFrames保存为CSV文件
df_stacked_pred_day1.to_csv('stacked_predictions_day1.csv', index=False)
df_stacked_pred_day2.to_csv('stacked_predictions_day2.csv', index=False)

