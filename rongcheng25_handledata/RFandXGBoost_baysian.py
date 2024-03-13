import os
from math import sqrt

from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

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

# 对数据进行预处理，将Nan的使用该列的下一个非nan代替
X_train = X_train.fillna(method='bfill')
X_test = X_test.fillna(method='bfill')

# 定义模型
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# XGBoost的参数搜索空间
search_spaces_xgb = {
    'n_estimators': Integer(100, 1000),
    'learning_rate': Real(0.01, 0.05),
    'max_depth': Integer(3, 10),
    'gamma': (0, 1),
    'colsample_bytree': Real(0.3, 1.0),
    'subsample': Real(0.5, 1.0)
}

# 随机森林的参数搜索空间
search_spaces_rf = {
    'n_estimators': Integer(5, 300),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5)
}

# XGBoost的贝叶斯搜索
bayes_search_xgb = BayesSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror'),
    search_spaces=search_spaces_xgb,
    n_iter=25,  # 可以调整迭代次数
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 训练模型并找到最佳参数
bayes_search_xgb.fit(X_train, y_train)
print("XGBoost最佳参数:", bayes_search_xgb.best_params_)

# 随机森林的贝叶斯搜索
bayes_search_rf = BayesSearchCV(
    estimator=RandomForestRegressor(),
    search_spaces=search_spaces_rf,
    n_iter=25,  # 可以调整迭代次数
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 训练模型并找到最佳参数
bayes_search_rf.fit(X_train, y_train)
print("随机森林最佳参数:", bayes_search_rf.best_params_)

# 获取最佳模型
best_xgb = bayes_search_xgb.best_estimator_
best_rf = bayes_search_rf.best_estimator_

# 定义堆叠模型
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('rf', best_rf)
    ],
    final_estimator=LinearRegression(),
    cv=5
)

# 训练堆叠模型
stacked_model.fit(X_train, y_train)
# 使用堆叠模型进行预测和评估
stacked_predictions = stacked_model.predict(X_test)
stacked_rmse = np.sqrt(mean_squared_error(y_test, stacked_predictions))
print(f"堆叠模型在测试集上的RMSE: {stacked_rmse}")

# 使用堆叠模型进行预测
stacked_pred_day1 = stacked_model.predict(X_test_day1)
stacked_pred_day2 = stacked_model.predict(X_test_day2)
stacked_pred_day3 = stacked_model.predict(X_test_day3)
stacked_pred_day4 = stacked_model.predict(X_test_day4)

# 评估各时间段堆叠模型的性能
stacked_rmse_day1 = np.sqrt(mean_squared_error(y_test_day1, stacked_pred_day1))
stacked_rmse_day2 = np.sqrt(mean_squared_error(y_test_day2, stacked_pred_day2))
stacked_rmse_day3 = np.sqrt(mean_squared_error(y_test_day3, stacked_pred_day3))
stacked_rmse_day4 = np.sqrt(mean_squared_error(y_test_day4, stacked_pred_day4))

print(f"堆叠模型在第一天的测试集上的RMSE: {stacked_rmse_day1}")
print(f"堆叠模型在第二天的测试集上的RMSE: {stacked_rmse_day2}")
print(f"堆叠模型在第三天的测试集上的RMSE: {stacked_rmse_day3}")
print(f"堆叠模型在第四天的测试集上的RMSE: {stacked_rmse_day4}")

# 保存预测结果到CSV文件
pd.DataFrame(stacked_pred_day1, columns=['Stacked Predictions Day 1']).to_csv('stacked_predictions_day1.csv',
                                                                              index=False)
pd.DataFrame(stacked_pred_day2, columns=['Stacked Predictions Day 2']).to_csv('stacked_predictions_day2.csv',
                                                                              index=False)
pd.DataFrame(stacked_pred_day3, columns=['Stacked Predictions Day 3']).to_csv('stacked_predictions_day3.csv',
                                                                              index=False)
pd.DataFrame(stacked_pred_day4, columns=['Stacked Predictions Day 4']).to_csv('stacked_predictions_day4.csv',
                                                                              index=False)

# 通过使用`BayesSearchCV`，你现在已经将原本基于网格搜索的参数优化更换为了基于贝叶斯优化的方法，这在许多情况下可以更高效地找到更好的参数组合。贝叶斯优化在参数空间中搜索时，会考虑先前评估的结果，这样可以更快地收敛到最优解。
#
# 接下来，你使用这些最优化的模型构建了一个堆叠模型，并对不同时间段的数据进行了预测。最后，你将每个时间段的预测结果保存为CSV文件，方便进一步分析和报告。
#
# 请记住，贝叶斯优化的过程可能需要一些时间来完成，特别是当参数空间较大、模型较复杂或数据集较大时。但通常，它能够找到比传统网格搜索或随机搜索更好的解决方案。
