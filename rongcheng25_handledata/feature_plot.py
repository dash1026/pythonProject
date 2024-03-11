import matplotlib.pyplot as plt

# # 特征名称及其占比(RF)
# features = ['P', 'H', 'RH', 'WS', 'T', 'WD', 'PT', 'TShear', 'WShear']
# importances = [0.278630, 0.242236, 0.107697, 0.096431, 0.081169, 0.075963, 0.048447, 0.034861, 0.034564]

# Given data(Permutation)
features = ['P', 'H', 'WS', 'PT', 'RH', 'WD', 'TShear', 'WShear', 'T']
importances = [0.426966, 0.208137, 0.110124, 0.082728, 0.046944, 0.027365, 0.011900, -0.000430, -0.052258]
# 创建条形图
plt.figure(figsize=(10, 6))  # 可以根据需要调整大小
plt.barh(features, importances, color='#1E90FF')  # 可以选择你喜欢的颜色
plt.xlabel('Importance')
plt.ylabel('Features')
# plt.title('Feature Importances - RF Chose')  # 根据需要调整标题

# 为每个条形添加数值标签
for index, value in enumerate(importances):
    plt.text(value, index, str(round(value, 4)))

plt.gca().invert_yaxis()  # 翻转Y轴，使得最重要的特征显示在顶部
plt.show()

# Python code to generate a feature importance bar chart similar to the one described

# import matplotlib.pyplot as plt
# import pandas as pd
#
# # Given data
# features = ['P', 'H', 'WS', 'PT', 'RH', 'WD', 'TShear', 'WShear', 'T']
# importances = [0.426966, 0.208137, 0.110124, 0.082728, 0.046944, 0.027365, 0.011900, -0.000430, -0.052258]
#
# # Create a pandas DataFrame
# df = pd.DataFrame({'Feature': features, 'Importance': importances})
#
# # Sort the DataFrame based on importance
# df = df.sort_values('Importance', ascending=True)
#
# # Create the plot
# plt.figure(figsize=(10, 6))
# bars = plt.barh(df['Feature'], df['Importance'], color='#1E90FF')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.title('Feature Importances')
#
# # Add text labels to the bars
# for bar in bars:
#     width = bar.get_width()
#     label_x_pos = width if width > 0 else width - 0.03
#     plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, '{:.5f}'.format(width), va='center')
#
# # Show the plot
# plt.tight_layout()
# plt.show()

