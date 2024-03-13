# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 给定的RMSE结果
# rmse_values = {
#     'M_noWShear': 0.41516647136331175,
#     'M_noT': 0.39970783229280016,
#     'M_noP': 0.40217803225325066,
#     'M_noWShear&T': 0.406783934149697,
#     'M_noWShear&T&P': 0.39885349466337516,
#     'M_all_features': 0.4184713644387481
# }
#
# # 转换为DataFrame
# rmse_df = pd.DataFrame(list(rmse_values.items()), columns=['Model', 'RMSE'])
#
# # 定义柱状图的颜色列表
# colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD', '#FF69B4']
#
# # 绘制柱状图
# plt.figure(figsize=[8, 6])
# bar_plot = plt.bar(rmse_df['Model'], rmse_df['RMSE'], color=colors)
# plt.ylabel('RMSE', fontsize=14)
# plt.xlabel('Model', fontsize=14)
# plt.xticks(rotation=45)  # 模型名称旋转45度
#
# # 设置y轴的范围以放大差异，你可以根据实际的RMSE值调整这里的范围
# plt.ylim(min(rmse_df['RMSE']) * 0.99, max(rmse_df['RMSE']) * 1.01)
#
# # 显示RMSE数值
# for bar, value in zip(bar_plot, rmse_df['RMSE']):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{value:.5f}', ha='center', va='bottom')
#
# plt.show()
import matplotlib.pyplot as plt
import pandas as pd

# 给定的RMSE结果
rmse_values = {
    'M_noWShear': 0.41516647136331175,
    'M_noT': 0.39970783229280016,
    'M_noP': 0.40217803225325066,
    'M_noWShear&T': 0.406783934149697,
    'M_noWShear&T&P': 0.39885349466337516,
    'M_all_features': 0.4184713644387481
}

# 转换为DataFrame
rmse_df = pd.DataFrame(list(rmse_values.items()), columns=['Model', 'RMSE'])


# 按照提供的颜色方案分配颜色
colors = ['#A19AD0', '#F0988C', '#B883D4', '#9E9E9E', '#CFEAF1', '#C4A5DE']  # 您可以按照图中的颜色添加或更换

# 确保数据按照RMSE值升序排列
rmse_df = rmse_df.sort_values(by='RMSE', ascending=False)


# 绘制柱状图
plt.figure(figsize=[8, 6])
bar_plot = plt.bar(rmse_df['Model'], rmse_df['RMSE'], color=colors)
plt.ylabel('RMSE', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.xticks(rotation=45)  # 模型名称旋转45度

# 设置y轴的范围以放大差异，你可以根据实际的RMSE值调整这里的范围
plt.ylim(min(rmse_df['RMSE']) * 0.99, max(rmse_df['RMSE']) * 1.01)

# 显示RMSE数值
for bar, value in zip(bar_plot, rmse_df['RMSE']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{value:.5f}', ha='center', va='bottom')

plt.show()
