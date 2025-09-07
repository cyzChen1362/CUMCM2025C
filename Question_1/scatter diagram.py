"""
绘制散点图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取数据
file_path = "D:/LearningDeepLearning/2025Modeling/Data_Processing/男胎检测数据_处理后_filtered.xlsx"
df = pd.read_excel(file_path)

# 计算BMI
df["BMI"] = df["体重"] / (df["身高"]/100)**2

# 提取需要的列
x1 = df["检测孕周"].values.reshape(-1, 1)
x2 = df["BMI"].values.reshape(-1, 1)
y = df["Y染色体浓度"].values

# 散点图 + 回归拟合函数
def plot_scatter_with_fit(x, y, xlabel, title):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)

    # 小的实心蓝点
    plt.scatter(x, y, s=3, c="blue", marker="o", label="Data")

    # 细的红线
    plt.plot(x, y_pred, color="red", linewidth=1, label=f"Fit line (R²={r2:.3f})")

    plt.xlabel(xlabel)
    plt.ylabel("Y chromosome concentration")
    plt.title(title)
    plt.legend()
    plt.show()
    return r2

# 绘制两个散点图
r2_week = plot_scatter_with_fit(x1, y, "Gestational week", "Y chromosome concentration vs Gestational week")
r2_bmi  = plot_scatter_with_fit(x2, y, "Maternal BMI", "Y chromosome concentration vs Maternal BMI")

print(f"Gestational week vs Y chromosome concentration R² = {r2_week:.3f}")
print(f"Maternal BMI vs Y chromosome concentration R² = {r2_bmi:.3f}")
