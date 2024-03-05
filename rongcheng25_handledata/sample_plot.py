import numpy as np
from matplotlib import pyplot as plt
#==============================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签，黑体的 name 为 SimHei
plt.rcParams['font.size'] = 16  # 设置字体大小
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号，跟是否显示中文没关系，你可以考虑加或不加
#==============================================

data = np.random.randn(1000, 2)


plt.hist(x=data,  # 绘图数据
         bins=20,  # 指定直方图的条形数为20个
         edgecolor='w',  # 指定直方图的边框色
         color=['#7CA98C', '#228DAE'],  # 指定直方图的填充色
         label=['数据一', '数据二'],  # 为直方图呈现图例
         density=False,  # 是否将纵轴设置为密度，即频率
         alpha=0.6,  # 透明度
         rwidth=1,  # 直方图宽度百分比：0-1
         stacked=False)  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放

ax = plt.gca()  # 获取当前子图
ax.spines['right'].set_color('none')  # 右边框设置无色
ax.spines['top'].set_color('none')  # 上边框设置无色
# 显示图例
plt.legend()
# 显示图形
plt.show()

data = np.random.randn(1000)
plt.hist(data,  # 绘图数据
         bins=20,  # 指定直方图的组距
         density=True,  # 设置为频率直方图
         cumulative=True,  # 积累直方图
         color='#135471',  # 指定填充色
         edgecolor='w',  # 指定直方图的边界色
         label='直方图')  # 为直方图呈现标签

# 设置坐标轴标签和标题
plt.title('累计频率直方图')
plt.xlabel('x轴')
plt.ylabel('累计频率')

# 显示图例
plt.legend(loc='best')
# 显示图形
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 生成随机数据
x = np.random.randn(9852)
data = pd.Series(x)  # 将数据由数组转换成Series形式

# 绘制直方图，设置颜色为蓝色
plt.hist(data, density=True, edgecolor='w', label='频数直方图', color='#228DAE')

# 绘制密度图，设置颜色为红色
data.plot(kind='kde', label='概率密度图', color='#DC143C')

# 显示图例
plt.legend()

# 显示图形
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 生成一些随机数据
data = np.random.randn(100)

# 绘制箱型图
plt.boxplot(data)


# 显示图形
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 生成一些随机数据
x = np.random.rand(50) * 100  # 假设这是一些特征值
y = np.random.rand(50) * 100  # 假设这是与特征值相关的另一组数据

# 绘制散点图
plt.scatter(x, y)

# 添加标题和轴标签
plt.title('Scatter Plot Example')
plt.xlabel('Feature')
plt.ylabel('Value')

# 显示图形
plt.show()
