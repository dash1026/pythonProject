# 将给定的RMSE结果转换为Python代码中的数据结构
rmse_values = {
    'M_noWShear': 0.41516647136331175,
    'M_noT': 0.39970783229280016,
    'M_noP': 0.40217803225325066,
    'M_noWShear&T': 0.406783934149697,
    'M_noWShear&T&P': 0.39885349466337516,
    'M_all_features': 0.4184713644387481
}

# 转换为DataFrame以方便排序
import pandas as pd
import matplotlib.pyplot as plt

rmse_df = pd.DataFrame(list(rmse_values.items()), columns=['Model', 'RMSE'])

# 确保基准模型总是在第一位
rmse_df['sort'] = rmse_df['Model'] == 'model_all_features'
# 对数据进行排序，确保其他模型按RMSE值降序排列

rmse_df = rmse_df.sort_values(by=['sort', 'RMSE'], ascending=[False, False]).drop(columns='sort')


# 绘制柱状图
plt.figure(figsize=[10, 6])
plt.barh(rmse_df['Model'], rmse_df['RMSE'], color='#1E90FF')
plt.xlabel('RMSE', fontsize=8)
plt.ylabel('Model', fontsize=8)
# plt.title('Comparison of Model RMSE')
plt.gca().invert_yaxis()  # 保证基准模型在图表顶端

# 显示RMSE数值
for index, value in enumerate(rmse_df['RMSE']):
    plt.text(value, index, str(round(value, 5)))

plt.show()
