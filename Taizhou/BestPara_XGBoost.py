import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


# 定义文件路径
# 定义文件路径
file_path_Xtrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTrain.csv'
file_path_ytrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTrain.csv'
file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest.csv'

file_path_X_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\XTest_Value_day002.csv'
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



df_y_pred = pd.DataFrame(y_pred, columns=['Predicted Values'])
df_y_test = pd.DataFrame(y_test, columns=['True Values'])



import os

# 定义目录路径
output_dir = 'rongcheng25_bestPara'

# 检查目录是否存在，如果不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 在指定的目录下保存CSV文件
df_y_pred.to_csv(os.path.join(output_dir, 'GSXGB_predicted.csv'), index=False)
df_y_test.to_csv(os.path.join(output_dir, 'GSXGB_True.csv'), index=False)

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



#  假设heights是提供的高度层数据，y_pred_values是对应的预测值列表，y_true_values是实际值列表

heights = [i*0.1 for i in range(250)]  # 从0到24.9的高度层，每隔0.1一个高度
y_pred_values_day1 = y_pred_day1 # 第一天的预测值列表
y_true_values_day1 = y_test_day1  # 第一天的实际值列表
y_pred_values_day2 = y_pred_day2 # 第一天的预测值列表
y_true_values_day2 = y_test_day2  # 第一天的实际值列表


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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# 假设 y_test 和 y_pred 是你的数据
# y_test = np.array([...])  # 实际测量值
# y_pred = np.array([...])  # 预测值

# 清除NaN和无穷大值
y_test = y_test[np.isfinite(y_test) & np.isfinite(y_pred)]
y_pred = y_pred[np.isfinite(y_test) & np.isfinite(y_pred)]

# 计算误差统计量
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
bias = np.mean(y_pred - y_test)  # 偏差
correlation = pearsonr(y_test, y_pred)[0]  # 相关系数

# 设置图形样式
plt.style.use('seaborn-darkgrid')  # 使用Seaborn的暗网格样式

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.6, marker='o', edgecolors='w', linewidth=0.5)  # 使数据点半透明
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', linewidth=2)  # 绘制红色的拟合直线

# 添加误差统计量的注释
stats_text = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nBias = {bias:.4f}\nR = {correlation * 100:.2f}%'
plt.text(np.percentile(y_test, 3), np.percentile(y_pred, 97), stats_text,
         verticalalignment='top', horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

# 设置图形属性
plt.xlabel('Measured lg $C_n^2$ / m$^{-2/3}$', fontsize=16)
plt.ylabel('Estimated lg $C_n^2$ / m$^{-2/3}$', fontsize=16)
plt.title('(a)', fontsize=18, fontweight='bold')
plt.axis('equal')
plt.xlim([min(y_test), max(y_test)])
plt.ylim([min(y_pred), max(y_pred)])
plt.tick_params(labelsize=12)  # 设置刻度字体大小
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 假设 y_test 和 y_pred 是您的数据
# y_test = np.array([...])  # 实际测量值
# y_pred = np.array([...])  # 预测值

# 数据集准备
data = [y_pred, y_test]

# 创建箱型图
fig, ax = plt.subplots(figsize=(8, 6))
boxprops = dict(linestyle='-', linewidth=2, color='darkgoldenrod')
whiskerprops = dict(linestyle='-', linewidth=2, color='orange')
capprops = dict(linestyle='-', linewidth=2, color='black')
medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
flierprops = dict(marker='o', markerfacecolor='green', markersize=12, linestyle='none')

# 绘制箱型图
box = ax.boxplot(data, patch_artist=True, notch=False, showmeans=False,
                 boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
                 medianprops=medianprops, flierprops=flierprops)

# 设置颜色
colors = ['lightblue', 'lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# 添加关键点数据展示
stats_labels = ['Min', 'Q1', 'Median', 'Q3', 'Max']
for i, (box_data, color) in enumerate(zip(data, colors)):
    # 计算统计量
    min_ = np.min(box_data)
    max_ = np.max(box_data)
    median = np.median(box_data)
    q1 = np.percentile(box_data, 25)
    q3 = np.percentile(box_data, 75)

    stats_values = [min_, q1, median, q3, max_]

    # 绘制文本
    for stat_label, stat_value in zip(stats_labels, stats_values):
        ax.text(i + 1, stat_value, f'{stat_label}: {stat_value:.2e}',
                verticalalignment='center', horizontalalignment='left' if i == 0 else 'right',
                color='black', fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# 添加标签
plt.xticks([1, 2], ['Estimation', 'Measurement'], fontsize=16)
plt.ylabel('Range')

# 添加标题和网格
plt.title('(b)', fontsize=18, fontweight='bold')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 直接显示图形
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 假设 y_test 和 y_pred 是您的数据
# y_test = np.array([...])  # 实际测量值
# y_pred = np.array([...])  # 预测值

# 设置直方图的参数
bins = np.linspace(-20, -14, 13)  # 根据实际数据调整

# 创建直方图并计算直方图值
hist_test, edges_test = np.histogram(y_test, bins=bins)
hist_pred, edges_pred = np.histogram(y_pred, bins=bins)

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制直方图，改进颜色和透明度
bars1 = ax1.bar(edges_test[:-1], hist_test, width=np.diff(edges_test), color='#FF5733', alpha=0.7, label='GS-XGBoost estimation')
bars2 = ax1.bar(edges_pred[:-1], hist_pred, width=np.diff(edges_pred), color='#33CFFF', alpha=0.7, label='Measurement')

# 计算累积频率
cum_freq_test = np.cumsum(hist_test) / np.sum(hist_test)
cum_freq_pred = np.cumsum(hist_pred) / np.sum(hist_pred)

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制累积频率曲线
line1, = ax2.plot(bins[:-1], cum_freq_test, 'o-', color='darkred', label='GS-XGBoost estimation')
line2, = ax2.plot(bins[:-1], cum_freq_pred, 'o-', color='darkblue', label='Measurement')

# 设置图形属性
ax1.set_xlabel('lg $C_n^2$ / m$^{-2/3}$', fontsize=16)
ax1.set_ylabel('Count', fontsize=16)
ax1.set_title('(c)', fontsize=18,  fontweight='bold')

# 设置图例
# 合并两个图例
lines = [bars1, bars2, line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=False)

# 移除网格
ax1.grid(False)  # 移除背景网格
ax2.grid(False)  # 移除背景网格

# 设置y轴标签
ax2.set_ylabel('Cumulative Frequency', fontsize=16)

# 显示图形
plt.show()
