import pandas as pd
import numpy as np

# 创建一个示例DataFrame
df = pd.DataFrame({
    'A': [1, np.nan, 3, 4, np.nan],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, np.nan, np.nan, 5]
})

# 打印原始DataFrame
print("原始DataFrame:")
print(df)

# 用每列的下一个非NaN值来填充NaN
df_filled = df.fillna(method='bfill')

# 打印处理后的DataFrame
print("\n处理后的DataFrame:")
print(df_filled)
