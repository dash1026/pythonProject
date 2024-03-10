import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 创建示例数据
data = np.random.rand(10, 10)  # 创建一个10x10的随机矩阵
df = pd.DataFrame(data, columns=[f'Var{i}' for i in range(1, 11)], index=[f'Sample{i}' for i in range(1, 11)])

# 绘制热图，并自定义坐标轴标签
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df,
                 cmap='coolwarm',
                 annot=True,
                 xticklabels=df.columns,
                 yticklabels=df.index)

# 设置坐标轴标签的字体大小
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize='x-small')
ax.set_yticklabels(ax.get_yticklabels(), fontsize='small')

# 设置坐标轴的名称
plt.xlabel('Variables')
plt.ylabel('Samples')

plt.show()
