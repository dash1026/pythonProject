from cmath import sqrt

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 定义文件路径
file_path_Xtrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTrain.csv'
file_path_ytrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTrain.csv'
file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest.csv'

file_path_X_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day002.csv'
file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day002.csv'

selected_features_names = ['H', 'P', 'T', 'WS', 'WShear', 'TShear']

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

# Best parameters from grid search
params = {
    'colsample_bytree': 0.6,
    'gamma': 1,
    'learning_rate': 0.0244,
    'max_depth': 9,
    'min_child_weight': 2,
    'n_estimators': 633,
    'subsample': 0.7973,
    'objective': 'reg:squarederror'
}

# Initialize XGBoost with the best parameters
xg_reg_optimized = xgb.XGBRegressor(**params)

# Fit the model
# Assuming X_train and y_train are already defined
xg_reg_optimized.fit(X_train, y_train)

# After fitting the model, you can perform predictions
# For example, predicting the test set
y_pred = xg_reg_optimized.predict(X_test)

# Evaluating the model performance
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
print(f"Optimized XGBoost Model RMSE on Test Set: {rmse:.4f}")
