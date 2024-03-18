from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error

# 数据路径定义
file_path_Xtrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTrain.csv'
file_path_ytrain = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTrain.csv'
file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest.csv'

file_path_X_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day001.csv'
file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day001.csv'
file_path_X_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\XTest_Value_day002.csv'
file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day002.csv'

file_path_y_waichidu_day1 = 'C:/Users\Administrator\PycharmProjects\pythonProject\Taizhou\y_waichidu_day1.csv'
file_path_y_waichidu_day2 = 'C:/Users\Administrator\PycharmProjects\pythonProject\Taizhou\y_waichidu_day2.csv'

y_waichidu_day1 = pd.read_csv(file_path_y_waichidu_day1)
y_waichidu_day2 = pd.read_csv(file_path_y_waichidu_day2)
# 选定的特征名称
# selected_features_names = ['H', 'WS', 'PT', 'RH', 'WD', 'TShear']
selected_features_names = ['H', 'P', 'T', 'WS', 'WShear', 'TShear']

# 读取数据
X_train = pd.read_csv(file_path_Xtrain)[selected_features_names].fillna(method='bfill')
y_train = pd.read_csv(file_path_ytrain).values.ravel()
X_test = pd.read_csv(file_path_Xtest)[selected_features_names].fillna(method='bfill')
y_test = pd.read_csv(file_path_ytest).values.ravel()
X_test_day1 = pd.read_csv(file_path_X_test_day1)[selected_features_names].fillna(method='bfill')
y_test_day1 = pd.read_csv(file_path_y_test_day1).values.ravel()
X_test_day2 = pd.read_csv(file_path_X_test_day2)[selected_features_names].fillna(method='bfill')
y_test_day2 = pd.read_csv(file_path_y_test_day2).values.ravel()

# 初始化并应用MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_day1_scaled = scaler.transform(X_test_day1)
X_test_day2_scaled = scaler.transform(X_test_day2)
# 定义并训练XGBoost和SVR模型（保留原始代码的贝叶斯优化部分）

# 定义模型
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# XGBoost的参数搜索空间
search_spaces_xgb = {
    'n_estimators': Integer(50, 500),
    'learning_rate': Real(0.01, 0.05),
    'max_depth': Integer(3, 8),
    'gamma': (0, 1),
    'colsample_bytree': Real(0.3, 1.0),
    'subsample': Real(0.5, 1.0)
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
bayes_search_xgb.fit(X_train_scaled, y_train)
print("XGBoost最佳参数:", bayes_search_xgb.best_params_)

# 获取最佳的 XGBoost 模型
best_xgb = bayes_search_xgb.best_estimator_

# 定义 SVR 和其参数搜索空间
search_spaces_svr = {
    'C': Real(0.1, 100),
    'epsilon': Real(0.01, 1),
    'gamma': Real(0.01, 1)
}

# 贝叶斯搜索的 SVR
bayes_search_svr = BayesSearchCV(
    estimator=SVR(),
    search_spaces=search_spaces_svr,
    n_iter=25,  # 可以调整迭代次数
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# 训练 SVR 并找到最佳参数
bayes_search_svr.fit(X_train_scaled, y_train)
print("SVR最佳参数:", bayes_search_svr.best_params_)
print("XGBoost最佳参数:", bayes_search_xgb.best_params_)
# 获取最佳的 SVR 模型
best_svr = bayes_search_svr.best_estimator_

# 定义堆叠模型
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),  # 假设best_xgb是前面贝叶斯优化找到的最佳XGBoost模型
        ('svr', best_svr)  # 假设best_svr是前面贝叶斯优化找到的最佳SVR模型
    ],
    final_estimator=LinearRegression(),
    cv=5
)

# 使用缩放后的训练数据拟合堆叠模型
stacked_model.fit(X_train_scaled, y_train)

XGB_predictions = best_xgb.predict(X_test_scaled)
XGB_rmse = np.sqrt(mean_squared_error(y_test, XGB_predictions))
print(f"XGBoost在测试集上的RMSE: {XGB_rmse}")

SVR_predictions = best_svr.predict(X_test_scaled)
SVR_rmse = np.sqrt(mean_squared_error(y_test, SVR_predictions))
print(f"SVR在测试集上的RMSE: {SVR_rmse}")

# 使用堆叠模型进行预测和评估
stacked_predictions = stacked_model.predict(X_test_scaled)
stacked_rmse = np.sqrt(mean_squared_error(y_test, stacked_predictions))
print(f"堆叠模型在测试集上的RMSE: {stacked_rmse}")

# 使用堆叠模型进行预测
stacked_pred_day1 = stacked_model.predict(X_test_day1_scaled)
stacked_pred_day2 = stacked_model.predict(X_test_day2_scaled)

# 评估各时间段堆叠模型的性能
stacked_rmse_day1 = np.sqrt(mean_squared_error(y_test_day1, stacked_pred_day1))
stacked_rmse_day2 = np.sqrt(mean_squared_error(y_test_day2, stacked_pred_day2))

print(f"堆叠模型在第一天的测试集上的RMSE: {stacked_rmse_day1}")
print(f"堆叠模型在第二天的测试集上的RMSE: {stacked_rmse_day2}")

# 保存预测结果到CSV文件
pd.DataFrame(stacked_pred_day1, columns=['Stacked Predictions Day 1']).to_csv('stacked_predictions_day1.csv',
                                                                              index=False)
pd.DataFrame(stacked_pred_day2, columns=['Stacked Predictions Day 2']).to_csv('stacked_predictions_day2.csv',
                                                                              index=False)

# 计算XGBoost最优模型的性能指标
XGB_predictions = best_xgb.predict(X_test_scaled)
XGB_rmse = sqrt(mean_squared_error(y_test, XGB_predictions))
XGB_mae = mean_absolute_error(y_test, XGB_predictions)
XGB_cc, _ = pearsonr(y_test, XGB_predictions)
print(f"XGBoost最优模型的RMSE: {XGB_rmse:.4f}, MAE: {XGB_mae:.4f}, CC: {XGB_cc:.4f}")

# 计算SVR最优模型的性能指标
SVR_predictions = best_svr.predict(X_test_scaled)
SVR_rmse = sqrt(mean_squared_error(y_test, SVR_predictions))
SVR_mae = mean_absolute_error(y_test, SVR_predictions)
SVR_cc, _ = pearsonr(y_test, SVR_predictions)
print(f"SVR最优模型的RMSE: {SVR_rmse:.4f}, MAE: {SVR_mae:.4f}, CC: {SVR_cc:.4f}")

# 使用堆叠模型进行预测和计算性能指标
stacked_predictions = stacked_model.predict(X_test_scaled)
stacked_rmse = sqrt(mean_squared_error(y_test, stacked_predictions))
stacked_mae = mean_absolute_error(y_test, stacked_predictions)
stacked_cc, _ = pearsonr(y_test, stacked_predictions)
print(f"融合模型的RMSE: {stacked_rmse:.4f}, MAE: {stacked_mae:.4f}, CC: {stacked_cc:.4f}")

#  假设heights是提供的高度层数据，y_pred_values是对应的预测值列表，y_true_values是实际值列表

heights = [i * 0.1 for i in range(250)]  # 从0到24.9的高度层，每隔0.1一个高度
y_pred_values_day1 = stacked_pred_day1  # 第一天的预测值列表
y_true_values_day1 = y_test_day1  # 第一天的实际值列表
y_pred_values_day2 = stacked_pred_day2  # 第一天的预测值列表
y_true_values_day2 = y_test_day2  # 第一天的实际值列表


# 修改画图函数，将图表转置，并使实际值的线平滑，减小实际值点的大小

def plot_values_on_heights_transposed(heights, y_pred_values_1, y_pred_values_2, y_true_values, title):
    plt.figure(figsize=(4, 5))
    plt.plot(y_true_values, heights, label='measured data', color='blue', linestyle='--', linewidth=0.75)
    plt.plot(y_pred_values_1, heights, label='Stacking-SVR&XGB', color='red', linestyle='-', linewidth=1)
    plt.plot(y_pred_values_2, heights, label='HMNSP99', color='#E29135', linestyle='-', linewidth=0.75)
    plt.title(title)
    plt.ylabel('Height /km')
    plt.xlabel(r'$\lg C_n^2  /m^{-\frac{2}{3}}$')
    plt.legend()

    plt.show()


# 使用修改后的函数画出各天的数据
plot_values_on_heights_transposed(heights, y_pred_values_day1, y_waichidu_day1, y_true_values_day1, 'Day 1 ')
plot_values_on_heights_transposed(heights, y_pred_values_day2, y_waichidu_day2, y_true_values_day2, 'Day 2')

# 计算误差统计量
mae = mean_absolute_error(y_test, stacked_predictions)
rmse = np.sqrt(mean_squared_error(y_test, stacked_predictions))
bias = np.mean(stacked_predictions - y_test)  # 偏差
correlation = pearsonr(y_test, stacked_predictions)[0]  # 相关系数

# 设置图形样式
plt.style.use('seaborn-darkgrid')  # 使用Seaborn的暗网格样式

# 绘制散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, stacked_predictions, color='dodgerblue', alpha=0.6, marker='o', edgecolors='w',
            linewidth=0.5)  # 使数据点半透明
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r-', linewidth=2)  # 绘制红色的拟合直线

# 添加误差统计量的注释
stats_text = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nBias = {bias:.4f}\nR = {correlation * 100:.2f}%'
plt.text(np.percentile(y_test, 3), np.percentile(stacked_predictions, 97), stats_text,
         verticalalignment='top', horizontalalignment='left', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

# 设置图形属性
plt.xlabel('Measured lg $C_n^2$ / m$^{-2/3}$', fontsize=16)
plt.ylabel('Estimated lg $C_n^2$ / m$^{-2/3}$', fontsize=16)
plt.title('(a)', fontsize=18, fontweight='bold')
plt.axis('equal')
plt.xlim([min(y_test), max(y_test)])
plt.ylim([min(stacked_predictions), max(stacked_predictions)])
plt.tick_params(labelsize=12)  # 设置刻度字体大小
plt.show()

# 数据集准备
data = [stacked_predictions, y_test]

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

# 设置直方图的参数
bins = np.linspace(-20, -14, 13)  # 根据实际数据调整

# 创建直方图并计算直方图值
hist_test, edges_test = np.histogram(y_test, bins=bins)
hist_pred, edges_pred = np.histogram(stacked_predictions, bins=bins)

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制直方图，改进颜色和透明度
bars1 = ax1.bar(edges_test[:-1], hist_test, width=np.diff(edges_test), color='#FF5733', alpha=0.7,
                label='Stacking-SVR&XGB estimation')
bars2 = ax1.bar(edges_pred[:-1], hist_pred, width=np.diff(edges_pred), color='#33CFFF', alpha=0.7, label='Measurement')

# 计算累积频率
cum_freq_test = np.cumsum(hist_test) / np.sum(hist_test)
cum_freq_pred = np.cumsum(hist_pred) / np.sum(hist_pred)

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制累积频率曲线
line1, = ax2.plot(bins[:-1], cum_freq_test, 'o-', color='darkred', label='Stacking-SVR&XGB estimation')
line2, = ax2.plot(bins[:-1], cum_freq_pred, 'o-', color='darkblue', label='Measurement')

# 设置图形属性
ax1.set_xlabel('lg $C_n^2$ / m$^{-2/3}$', fontsize=16)
ax1.set_ylabel('Count', fontsize=16)
ax1.set_title('(c)', fontsize=18, fontweight='bold')

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
