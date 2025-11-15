import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


path = "../Data/附件.xlsx"
male = pd.read_excel(path, sheet_name="男胎检测数据")
female = pd.read_excel(path, sheet_name="女胎检测数据")

def plot_missing_bar(series, title, y_label, save_path, xtick_step=200):
    """
    缺失值柱状图绘制函数
    """
    arr = series.isna().astype(int).to_numpy()[None, :]
    cmap = ListedColormap(["#1f77b4", "#d62728"])

    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_yticks([0])
    ax.set_yticklabels([y_label], fontsize=12)

    # Add index ticks
    n = len(series)
    xticks = np.arange(0, n, xtick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize=9)

    # Legend
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Present"),
        Patch(facecolor="#d62728", label="Missing")
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.0, 1.6), ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

# 男胎数据集——末次月经缺失值
plot_missing_bar(
    male["末次月经"],
    title="Missingness Map — Male Dataset (Last Menstruation)",
    y_label="Last Menstruation",
    save_path="missing_male_last_menstruation.png"
)

# 女胎数据集——末次月经缺失值
plot_missing_bar(
    female["末次月经"],
    title="Missingness Map — Female Dataset (Last Menstruation)",
    y_label="Last Menstruation",
    save_path="missing_female_last_menstruation.png"
)

# 女胎数据集——BMI缺失值
plot_missing_bar(
    female["孕妇BMI"],
    title="Missingness Map — Female Dataset (Maternal BMI)",
    y_label="Maternal BMI",
    save_path="missing_female_bmi.png"
)
