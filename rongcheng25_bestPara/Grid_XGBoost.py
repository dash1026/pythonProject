import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from math import sqrt

# 定义文件路径
file_path_Xtrain = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\XTrain.csv'
file_path_ytrain = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\YTrain.csv'
file_path_Xtest = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\XTest.csv'
file_path_ytest = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\YTest.csv'

file_path_X_test_day1 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\XTest_Value_day002.csv'
file_path_y_test_day2 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\YTest_Value_day002.csv'
file_path_X_test_day3 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\XTest_Value_day003.csv'
file_path_y_test_day3 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\YTest_Value_day003.csv'
file_path_X_test_day4 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\XTest_Value_day004.csv'
file_path_y_test_day4 = 'D:\\data\\荣成\\荣成25km\\rongchengdata_python\\YTest_Value_day004.csv'

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

# 提供的超参数
params = {'colsample_bytree': 0.7, 'gamma': 0.4, 'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 4,
          'n_estimators': 150, 'subsample': 0.5}

# 构建XGBoost模型
model = xgb.XGBRegressor(
    colsample_bytree=params['colsample_bytree'],
    gamma=params['gamma'],
    learning_rate=params['learning_rate'],
    max_depth=params['max_depth'],
    min_child_weight=params['min_child_weight'],
    n_estimators=params['n_estimators'],
    subsample=params['subsample']
)

# 假设X_train, y_train是训练数据集，X_test, y_test是测试数据集
# 你需要加载你的数据集来替换这些变量

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)
y_pred_day1 = model.predict(X_test_day1)
y_pred_day2 = model.predict(X_test_day2)
y_pred_day3 = model.predict(X_test_day3)
y_pred_day4 = model.predict(X_test_day4)

# 如果数组不是一维的，将它们展平为一维
y_test_flattened = y_test.ravel()

y_pred_flattened = y_pred.flatten()

# 计算平均绝对误差（MAD）
mad = mean_absolute_error(y_test_flattened, y_pred_flattened)

# 计算皮尔逊相关系数
correlation, _ = pearsonr(y_test_flattened, y_pred_flattened)

# 对其他天的预测重复这个处理过程


# 计算RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred))

# 计算MAD
mae = mean_absolute_error(y_test, y_pred)

# 计算CC（皮尔逊相关系数）
cc = pearsonr(y_test, y_pred)[0]

# 输出统计量
print(f"RMSE: {rmse}")
print(f"CC: {cc}")
print(f"MAD: {mae}")



import matplotlib.pyplot as plt

# 假设heights是提供的高度层数据，y_pred_values是对应的预测值列表，y_true_values是实际值列表
heights = [i*0.1 for i in range(250)]  # 从0到24.9的高度层，每隔0.1一个高度
y_pred_values_day1 = y_pred_day1 # 第一天的预测值列表
y_true_values_day1 = y_test_day1  # 第一天的实际值列表
y_pred_values_day2 = y_pred_day2 # 第一天的预测值列表
y_true_values_day2 = y_test_day2  # 第一天的实际值列表
y_pred_values_day3 = y_pred_day3 # 第一天的预测值列表
y_true_values_day3 = y_test_day3  # 第一天的实际值列表
y_pred_values_day4 = y_pred_day4 # 第一天的预测值列表
y_true_values_day4 = y_test_day4  # 第一天的实际值列表

# 修改画图函数，将图表转置，并使实际值的线平滑，减小实际值点的大小

def plot_values_on_heights_transposed(heights, y_pred_values, y_true_values, title):
    plt.figure(figsize=(4, 5))
    plt.plot(y_true_values, heights, label='measured data', color='blue', linestyle='--', linewidth=0.5)
    plt.plot(y_pred_values, heights, label='GS-XGBoost', color='#E29135', linestyle='-', linewidth=0.75)
    plt.title(title)
    plt.ylabel('Height /km')
    plt.xlabel(r'$\lg C_n^2  /m^{-\frac{2}{3}}$')
    plt.legend()

    plt.show()

# 使用修改后的函数画出各天的数据
plot_values_on_heights_transposed(heights, y_pred_values_day1, y_true_values_day1, 'Day 1 ')
plot_values_on_heights_transposed(heights, y_pred_values_day2, y_true_values_day2, 'Day 2')
plot_values_on_heights_transposed(heights, y_pred_values_day3, y_true_values_day3, 'Day 3 ')
plot_values_on_heights_transposed(heights, y_pred_values_day4, y_true_values_day4, 'Day 4 ')


