# Correcting the error by defining numpy as np and retrying the calculation
import numpy as np
import pandas as pd

file_path_y_test_day1 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day001.csv'
file_path_y_test_day2 = 'D:\data\台州\Program\Taizhou_pythondata\\night\\YTest_Value_day002.csv'


y_test_day1 = pd.read_csv(file_path_y_test_day1).values.ravel()
y_test_day2 = pd.read_csv(file_path_y_test_day2).values.ravel()


file_path_y_stacking_day1 = 'C:/Users\Administrator\PycharmProjects\pythonProject\Taizhou\Stacking\stacked_predictions_day1.csv'
file_path_y_stacking_day2 = 'C:/Users\Administrator\PycharmProjects\pythonProject\Taizhou\Stacking\stacked_predictions_day2.csv'

y_stacking_day1 = pd.read_csv(file_path_y_stacking_day1).values.ravel()
y_stacking_day2 = pd.read_csv(file_path_y_stacking_day2).values.ravel()

file_path_height = 'D:\data\台州\Program\Taizhou_pythondata/night\height_day001_temp.csv'

y_height = pd.read_csv(file_path_height).values.ravel()


# 假设已经加载了数据到 y_test_day1, y_test_day2, y_stacking_day1, y_stacking_day2

def clean_data_and_calculate_Cn2(data):
    """
    清理数据中的 NaN 值并计算 Cn2 值。

    参数:
    - data: 包含对数Cn2值的numpy数组，可能包含NaN值。

    返回:
    - Cn2: 清理NaN后的Cn2值的数组。
    """
    # 检查数据中的 NaN 值并计算非 NaN 值的平均数
    data_without_nan = data[~np.isnan(data)]  # 移除 NaN 值
    average_value = np.mean(data_without_nan)  # 计算平均值

    # 替换 NaN 值为平均数
    clean_data = np.where(np.isnan(data), average_value, data)

    # 计算 Cn2 值
    Cn2 = 10 ** clean_data

    return Cn2


# 应用清理和计算过程
Cn2_day1 = clean_data_and_calculate_Cn2(y_test_day1)
Cn2_day2 = clean_data_and_calculate_Cn2(y_test_day2)
Cn2_stacking_day1 = clean_data_and_calculate_Cn2(y_stacking_day1)
Cn2_stacking_day2 = clean_data_and_calculate_Cn2(y_stacking_day2)

# 常量定义
wavelength = 550e-9  # 波长，单位：米
k = 2 * np.pi / wavelength  # 波数
solar_altitude_angle_degrees = -12.57  # 太阳高度角，单位：度
zenith_angle_radians = np.radians(90 + abs(solar_altitude_angle_degrees))  # 天顶角，单位：弧度
sec_z = 1 / np.cos(zenith_angle_radians)  # 天顶角的正割

# 示例高度数组，应根据实际高度数据调整
altitudes = y_height
integral_Cn2_day1 = np.trapz(Cn2_day1, altitudes)
integral_Cn2_day2 = np.trapz(Cn2_day2, altitudes)
integral_Cn2_stacking_day1 = np.trapz(Cn2_stacking_day1, altitudes)
integral_Cn2_stacking_day2 = np.trapz(Cn2_stacking_day2, altitudes)
print(integral_Cn2_day1)
print(integral_Cn2_day2)
print(integral_Cn2_stacking_day1)
print(integral_Cn2_stacking_day2)
# # 计算每组数据的相干长度r0
# def calculate_r0(Cn2, altitudes, k, sec_z):
#     integral_Cn2 = np.trapz(Cn2, altitudes)
#     return (0.423 * k**2 * sec_z * integral_Cn2)**(-3/5)
#
# r0_day1 = calculate_r0(Cn2_day1, altitudes, k, sec_z)
# r0_day2 = calculate_r0(Cn2_day2, altitudes, k, sec_z)
# r0_stacking_day1 = calculate_r0(Cn2_stacking_day1, altitudes, k, sec_z)
# r0_stacking_day2 = calculate_r0(Cn2_stacking_day2, altitudes, k, sec_z)
#
# integral_Cn2 = np.trapz(Cn2_day1, altitudes)
#
# # 打印结果
# print("r0_day1:", r0_day1)
# print("r0_day2:", r0_day2)
# print("r0_stacking_day1:", r0_stacking_day1)
# print("r0_stacking_day2:", r0_stacking_day2)

