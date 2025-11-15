import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读入数据
file_path = "../Data/男胎检测数据_处理后.xlsx"
df = pd.read_excel(file_path)

# 2. 需要做 IQR 异常值检测和画箱线图的变量名
cols = [
    "年龄",
    "身高",
    "体重",
    "孕妇BMI",
    "检测孕周",
    "GC含量",
    "13号染色体的Z值",
    "18号染色体的Z值",
    "21号染色体的Z值"
]

# 3. IQR 方法检测异常值
outlier_info = {}

for col in cols:
    data = df[col].dropna()  # 去掉缺失值
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # 异常值
    outliers = data[(data < lower) | (data > upper)]
    num_outliers = outliers.shape[0]
    pct_outliers = num_outliers / data.shape[0] * 100

    outlier_info[col] = (num_outliers, pct_outliers)

    # 异常值位置输出到控制台
    print(f"{col}: 异常值 {num_outliers} 个，占 {pct_outliers:.2f}%")
    print(outliers)

# 4. 画 3×3 九合一箱线图
plt.rcParams["font.sans-serif"] = ["SimHei"]     # 支持中文
plt.rcParams["axes.unicode_minus"] = False       # 负号正常显示

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
fig.suptitle("主要变量异常值检测箱线图（IQR方法）", fontsize=16, y=0.98)

for i, col in enumerate(cols):
    ax = axes[i // 3, i % 3]

    # 当前变量数据
    data = df[col].dropna()

    # 画箱线图
    bp = ax.boxplot(
        data,
        vert=True,
        showfliers=True,    # 显示异常值点
        patch_artist=True   # 允许填充颜色
    )

    # 简单设置一下箱体颜色（可省略）
    for patch in bp['boxes']:
        patch.set_alpha(0.7)

    ax.set_title(col, fontsize=12)
    ax.set_xticks([])  # 横轴刻度去掉，只保留箱线图

    # 添加异常值数量和比例标注
    n_out, pct_out = outlier_info[col]
    label = f"异常值: {n_out} ({pct_out:.1f}%)"
    ax.text(
        0.03, 0.95, label,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fce6c9", ec="peru", alpha=0.9)
    )

# 5. 布局 & 保存
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给总标题留一点空间
plt.savefig("./Images/picture7.png", dpi=300, bbox_inches="tight")
plt.show()
