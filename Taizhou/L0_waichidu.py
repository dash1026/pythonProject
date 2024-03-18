import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


file_path_Xtest = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\XTest.csv'
file_path_ytest = 'D:\data\台州\Program\Taizhou_pythondata\\nightwaichidu\YTest.csv'
X_test = pd.read_csv(file_path_Xtest).fillna(method='bfill')
y_test = pd.read_csv(file_path_ytest).fillna(method='bfill')

h = X_test['H']
# T = X_test['T']
# P = X_test['P']
WS = X_test['WShear']
TS = X_test['TShear']
M = X_test['M']
# 定义拟合函数
def model1(x, a, b, c):
    WS, TS = x
    return a + b*WS + c*TS

def model2(x, d, e, f):
    WS, TS = x
    return d + e*WS + f*TS

# 数据分段
mask = h <= 11000
WS1, TS1, M1 = WS[mask], TS[mask], M[mask]
WS2, TS2, M2 = WS[~mask], TS[~mask], M[~mask]

# 拟合第一个模型
popt1, _ = curve_fit(model1, (WS1, TS1), M1)

# 拟合第二个模型
popt2, _ = curve_fit(model2, (WS2, TS2), M2)

# 输出拟合参数
print("第一个模型参数 (a, b, c):", popt1)
print("第二个模型参数 (d, e, f):", popt2)
