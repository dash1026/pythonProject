import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


# 定义文件路径
file_path_Xtrain = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTrain.csv'
file_path_ytrain = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTrain.csv'
file_path_Xtest = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest.csv'
file_path_ytest = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest.csv'



file_path_X_test_day1 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day001.csv'
file_path_y_test_day1  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day002.csv'
file_path_y_test_day2  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day002.csv'
file_path_X_test_day3 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day003.csv'
file_path_y_test_day3  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day003.csv'
file_path_X_test_day4 = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\XTest_Value_day004.csv'
file_path_y_test_day4  = 'D:\\data\\荣成\荣成25km\\rongchengdata_python\\YTest_Value_day004.csv'

# 使用pandas的read_csv函数读取数据
X_train = pd.read_csv(file_path_Xtrain)
y_train = pd.read_csv(file_path_ytrain)
X_test = pd.read_csv(file_path_Xtest)
y_test = pd.read_csv(file_path_ytest)

X_test_day1 = pd.read_csv(file_path_X_test_day1)
y_test_day1 = pd.read_csv(file_path_y_test_day1)
X_test_day2 = pd.read_csv(file_path_X_test_day2)
y_test_day2 = pd.read_csv(file_path_y_test_day2)
X_test_day3 = pd.read_csv(file_path_X_test_day3)
y_test_day3 = pd.read_csv(file_path_y_test_day3)
X_test_day4 = pd.read_csv(file_path_X_test_day4)
y_test_day4 = pd.read_csv(file_path_y_test_day4)

y_train = y_train.values.ravel()

# 对数据进行预处理，将Nan的使用该列的下一个非nan代替
X_train = X_train.fillna(method='bfill')
X_test = X_test.fillna(method='bfill')
# 定义随机森林模型函数，以便多次使用
def train_rf_and_compute_rmse(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    return rmse

# 定义特征集
feature_sets = {
    'model_noWShear': ['P', 'H', 'WS', 'PT', 'RH', 'WD', 'TShear', 'T'],
    'model_noT': ['P', 'H', 'WS', 'PT', 'RH', 'WD', 'TShear', 'WShear'],
    'model_noP': ['H', 'WS', 'PT', 'RH', 'WD', 'TShear', 'WShear'],
    'model_noWShear&T': ['P', 'H', 'WS', 'PT', 'RH', 'WD', 'WShear'],
    'model_noWShear&T&P': ['H', 'WS', 'PT', 'RH', 'WD', 'TShear']
}

# 扩展特征集以包含一个基准模型，它使用所有特征
feature_sets['model_all_features'] = X_train.columns.tolist()  # 添加所有特征的模型

# 初始化字典来存储各个模型的RMSE
rmse_results = {}

# 对每组特征，训练模型并计算RMSE
for model_name, features in feature_sets.items():
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    rmse = train_rf_and_compute_rmse(X_train_subset, y_train, X_test_subset, y_test)
    rmse_results[model_name] = rmse

# 打印每个模型的RMSE
for model_name, rmse in rmse_results.items():
    print(f'{model_name} RMSE: {rmse}')
