import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\YTest.csv'
file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day001.csv'
file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day002.csv'
X_test = pd.read_csv(file_path_Xtest).fillna(method='bfill')
y_test = pd.read_csv(file_path_ytest).fillna(method='bfill')
y_test_day1 = pd.read_csv(file_path_y_test_day1).fillna(method='bfill')
y_test_day2 = pd.read_csv(file_path_y_test_day2).fillna(method='bfill')


h = X_test['H']
T = X_test['T']
P = X_test['P']
WS = X_test['WShear']
TS = X_test['TShear']
PTS = X_test['PTS']

def cn2_model(h, T, P, WS, TS, PTS):

    P[P == 0] = np.nan  # 将P为0的情况设置为nan以避免除以0

    # 初始化 cn2 数组
    cn2 = np.zeros_like(h, dtype=np.float64)
    # [0.362  16.728 192.347   0.757  13.819  57.784]


    # 对流层条件
    mask_troposphere = (h <= 14600)
    L0_factor_troposphere = 0.14 * 10**(0.362 + 16.728 * WS[mask_troposphere] - 192.347 * TS[mask_troposphere])
    cn2[mask_troposphere] = 2.8 * (L0_factor_troposphere * ((79e-6 * P[mask_troposphere] / T[mask_troposphere]**2) * PTS[mask_troposphere])**2)

    # 平流层条件
    mask_stratosphere = (h > 14600)
    L0_factor_stratosphere = 0.14 * 10**(0.757 + 13.819 * WS[mask_stratosphere] - 57.784 * TS[mask_stratosphere])
    cn2[mask_stratosphere] = 2.8 * L0_factor_stratosphere * (((-79e-6 * P[mask_stratosphere] / T[mask_stratosphere]**2) * PTS[mask_stratosphere])**2)

    # 确保 cn2 中没有0或负值
    cn2[cn2 <= 0] = np.nan  # 将小于等于0的值设为NaN

    # # 在取对数之前处理NaN
    # with np.errstate(invalid='ignore'):
    #     log_cn2 = np.log10(cn2)  # 使用log10来避免对负数取对数

    return cn2

# 调用函数得到y_pred
y_pred = cn2_model(h, T, P, WS, TS, PTS)

# 在对数转换前处理NaN，确保不对NaN或负数取对数
y_pred_non_negative = np.where(y_pred <= 0, np.nan, y_pred)  # 替换所有非正数为NaN
y_pred_log = np.log10(y_pred_non_negative)  # 安全取对数
y_test_log = np.log10(y_test['Cn2'].replace(0, np.nan))  # 先将0替换为NaN，再取对数

# 计算性能指标之前，将NaN替换为对数后的数组最小值
min_log_y_pred = np.nanmin(y_pred_log[np.isfinite(y_pred_log)])  # 只考虑非NaN值
y_pred_log = np.nan_to_num(y_pred_log, nan=min_log_y_pred)
min_log_y_test = np.nanmin(y_test_log[np.isfinite(y_test_log)])
y_test_log = np.nan_to_num(y_test_log, nan=min_log_y_test)
half = len(y_pred_log) // 2

# 创建比较DataFrame
comparison = pd.DataFrame({'Actual': y_test_log, 'Predicted': y_pred_log})
print(comparison)

# 计算整个测试集的评价指标
mse_total = mean_squared_error(y_test_log, y_pred_log)
rmse_total = math.sqrt(mse_total)
mae_total = mean_absolute_error(y_test_log, y_pred_log)
cc_total = np.corrcoef(y_test_log, y_pred_log)[0, 1]  # 取相关系数矩阵的非对角元素

print(f"总体 R Mean Squared Error: {rmse_total}")
print(f"总体 Mean Absolute Error: {mae_total}")
print(f"总体 Correlation Coefficient: {cc_total}")

# 创建包含y_pred_log的DataFrame
y_pred_log_df = pd.DataFrame({'Predicted_Cn2_Log': y_pred_log})

# 存储DataFrame到CSV文件
y_pred_log_df.to_csv('waichidu.csv', index=False)

print("对数转换后的估算值已存储到 waichidu.csv 中。")

import pandas as pd

# 假设已经正确分割了 y_pred_log
y_waichidu_day1 = y_pred_log[:half]
y_waichidu_day2 = y_pred_log[half:]

# 将这两个数组分别保存到CSV文件中
y_waichidu_day1_df = pd.DataFrame(y_waichidu_day1, columns=['y_waichidu_day1'])
y_waichidu_day2_df = pd.DataFrame(y_waichidu_day2, columns=['y_waichidu_day2'])

# 保存到CSV文件
y_waichidu_day1_df.to_csv('y_waichidu_day1.csv', index=False)
y_waichidu_day2_df.to_csv('y_waichidu_day2.csv', index=False)

