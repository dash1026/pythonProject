# import pandas as pd
# from scipy.optimize import curve_fit
#
# # file_path_Xtrain = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\XTrain.csv'
# # file_path_ytrain = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\YTrain.csv'
# file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\XTest.csv'
# file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\YTest.csv'
# #
# #
# # file_path_X_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\XTest_Value_day001.csv'
# # file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\YTest_Value_day001.csv'
# # file_path_X_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\XTest_Value_day002.csv'
# # file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\YTest_Value_day002.csv'
#
#
# # # 读取数据
# # X_train = pd.read_csv(file_path_Xtrain).fillna(method='bfill')
# # y_train = pd.read_csv(file_path_ytrain).fillna(method='bfill')
# X_test = pd.read_csv(file_path_Xtest).fillna(method='bfill')
# y_test = pd.read_csv(file_path_ytest).fillna(method='bfill')
# # X_test_day1 = pd.read_csv(file_path_X_test_day1).fillna(method='bfill')
# # y_test_day1 = pd.read_csv(file_path_y_test_day1).fillna(method='bfill')
# # X_test_day2 = pd.read_csv(file_path_X_test_day2).fillna(method='bfill')
# # y_test_day2 = pd.read_csv(file_path_y_test_day2).fillna(method='bfill')
#
# file_Data_Mat_all = 'D:\data\台州\Program\Data_Mat_All.csv'
#
# Data_Mat_all = pd.read_csv(file_Data_Mat_all).fillna(method='bfill')
#
# # 定义模型函数
# import numpy as np
#
#
# # def cn2_model(h, T, P, WS, PS, TS,  a1, a2, a3, b1, b2, b3):
# #     theta_gradient = (1000 / P)**0.286 * TS + T * 0.286 * (1000 / P)**0.286 * 1000 / P**2 * PS  # 你需要提供 theta 对 h 的导数的计算方法
# #     # 初始化cn2数组
# #     cn2 = np.zeros_like(h)
# #     # 对流层条件
# #     mask_troposphere = h <= 11000
# #     L0_factor_troposphere = 0.14 * 10 ** (a1 + a2* WS + a3 * (T[mask_troposphere] / P[mask_troposphere]))
# #     cn2[mask_troposphere] = 2.8 * (L0_factor_troposphere * (
# #                 (79e-6 * P[mask_troposphere] / T[mask_troposphere] ** 2) * theta_gradient) ** 2)
# #
# #     # 平流层条件
# #     mask_stratosphere = h > 11000
# #     L0_factor_stratosphere = 0.14 * 10 ** (b1 + b2 * WS + b3 * (T[mask_stratosphere] / P[mask_stratosphere]))
# #     cn2[mask_stratosphere] = 2.8 * (L0_factor_stratosphere * (
# #                 (79e-6 * P[mask_stratosphere] / T[mask_stratosphere] ** 2) * theta_gradient) ** 2)
# #
# #     return np.log(cn2)
#
# def cn2_model(h, T, P, WS, PS, TS, a1, a2, a3, b1, b2, b3):
#     # 初始化 cn2 数组
#     cn2 = np.zeros_like(h)
#
#     # 对流层和平流层的条件分开计算
#     for i in range(len(h)):
#         if h[i] <= 11000:  # 对流层条件
#             L0_factor = 0.14 * 10 ** (a1 + a2 * WS[i] - a3 * TS[i])
#             cn2[i] = 2.8 * (L0_factor * ((79e-6 * P[i] / T[i] ** 2) * theta_gradient) ** 2)
#         else:  # 平流层条件
#             L0_factor = 0.14 * 10 ** (b1 + b2 * WS[i] - b3 * TS[i])
#             cn2[i] = 2.8 * (L0_factor * ((79e-6 * P[i] / T[i] ** 2) * theta_gradient) ** 2)
#
#     cn2 = np.maximum(cn2, 1e-10)  # Replace zero and negative values with a small positive number
#     return np.log10(cn2)
#
#
# # 你的数据
# h_data = Data_Mat_all['H']
# T_data = Data_Mat_all['T']
# P_data = Data_Mat_all['P']
# WS_data = Data_Mat_all['WShear']
# TS_data = Data_Mat_all['TShear']
#
# cn2_data = Data_Mat_all['Cn2']  # C_n^2 实测值
# # 这里你需要提供这些数据
#
# # curve_fit需要的xdata应该包括所有的自变量，我们这里将它们打包成一个数组
# # 注意，curve_fit并不直接支持多个自变量，所以我们需要稍微变通处理
# def fit_func(xdata, a1, a2, a3, b1, b2, b3):
#     # 从 xdata 中获取 h, T, P, PT，这里 xdata 是一个 2D 数组
#     h = xdata[:, 0]
#     T = xdata[:, 1]
#     P = xdata[:, 2]
#     WS = xdata[:, 3]
#     PS = xdata[:, 4]
#     TS = xdata[:, 5]
#
#     return cn2_model(h, T, P, WS, PS, TS, a1, a2, a3, b1, b2, b3)
#
#
#
# xdata = np.vstack((h_data, T_data, P_data, WS_data,,TS_data)).T  # T 表示转置，转换为正确的格式
# ydata = cn2_data  # Make sure the ydata is on the correct scale for fitting
# # 进行拟合
# # 注意这里需要以正确的形式提供xdata和ydata
# initial_guess = [0.362, 16.728, 192.347, 0.757, 13.819, 57.784]  # 根据模型情况进行修改
#
# try:
#     popt, pcov = curve_fit(fit_func, xdata, ydata, p0=initial_guess, maxfev=10000)
# except RuntimeError as e:
#     print(e)
#
#
# # 打印拟合结果
# print("拟合得到的参数值：", popt)
#
# # 假设您已经完成了前面的拟合工作，并且已经有了拟合参数 popt
# # ...
#
# # 使用拟合出来的参数来进行预测
# def predict(xdata, popt):
#     h, T, P, WS, PS, TS = xdata[:, 0], xdata[:, 1], xdata[:, 2], xdata[:, 3], xdata[:, 4], xdata[:, 5]
#     return cn2_model(h, T, P, WS, PS, TS, *popt)
#
# # 准备测试数据
# xdata_test = np.vstack((X_test['H'].values, X_test['T'].values, X_test['P'].values, X_test['WShear'].values, X_test['PShear'].values, X_test['TShear'].values)).T
#
# # 进行预测
# y_pred = predict(xdata_test, popt)
#
#
# # 预测完成后，您可以将预测结果与实际值进行比较
# # 注意：您的 y_test 应该是在相同的尺度上
# comparison = pd.DataFrame({'Actual': y_test['Cn2'], 'Predicted': y_pred})
# print(comparison)
#
# # 您可以计算一些性能指标，例如均方误差（MSE）或决定系数（R^2）
# from sklearn.metrics import mean_squared_error, r2_score
#
# mse = mean_squared_error(y_test['Cn2'], y_pred)
# r2 = r2_score(y_test['Cn2'], y_pred)
#
# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)
