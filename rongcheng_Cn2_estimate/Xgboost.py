import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os


# 定义文件路径
file_path_feature = 'D:\\data\\Python_data\\rongcheng_six_all.csv'
file_path_label = 'D:\\data\\Python_data\\rongcheng_one_all.csv'

# 使用pandas的read_csv函数读取数据
data_feature = pd.read_csv(file_path_feature)
data_label = pd.read_csv(file_path_label)

# # 查看前几行数据以确认正确导入
# print(data_feature.head())
# print(data_label.head())

X_train = data_feature.iloc[:4605]
y_train = data_label.iloc[:4605]
X_test = data_feature[4606:]
y_test = data_label[4606:]
X_test_day1 = data_feature[4606:4910]
Y_test_day1 = data_label[4606:4910]
X_test_day2 = data_feature[4911:5196]
Y_test_day2 = data_label[4911:5196]
X_test_day3 = data_feature[5197:5500]
Y_test_day3 = data_label[5197:5500]
X_test_day4 = data_feature[5501:]
Y_test_day4 = data_label[5501:]


#定义模型的训练参数
reg_mod = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.08,
    subsample=0.75,
    colsample_bytree=1,
    max_depth=7,
    gamma=0,
    eval_metric='rmse',

)
# 训练模型并指定评估数据集
eval_set = [(X_train, y_train), (X_test, y_test)]
reg_mod.fit(X_train, y_train, eval_set=eval_set,verbose=False)
# 绘制损失曲线
sns.set_style("white")
palette = sns.color_palette("Set2", n_colors=2)

plt.plot(reg_mod.evals_result()['validation_0']['rmse'], label='train', color=palette[0], linewidth=2)
plt.plot(reg_mod.evals_result()['validation_1']['rmse'], label='test', color=palette[1], linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('Loss.png')
plt.show()