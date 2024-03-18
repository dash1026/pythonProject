from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# Assuming you have already loaded and preprocessed your data:
# X_train_scaled, y_train, X_test_scaled, y_test, etc.

# Set the optimized hyperparameters
C_optimized = 86.99786975290928
epsilon_optimized = 0.07756400886238057
gamma_optimized = 0.9473190028663652

# Initialize and train the SVR model with the optimized parameters
model_optimized = SVR(C=C_optimized, epsilon=epsilon_optimized, gamma=gamma_optimized)
model_optimized.fit(X_train_scaled, y_train)

# Predictions
y_pred = model_optimized.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
mad = mean_absolute_error(y_test, y_pred)
correlation, _ = pearsonr(y_test, y_pred)

# Output the evaluation metrics
print(f"Optimized SVR Model Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAD: {mad:.4f}")
print(f"Pearson Correlation: {correlation:.4f}")
