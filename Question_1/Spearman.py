"""
计算三列的 Spearman 相关，并绘制热力图
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False        # 解决负号显示为方块的问题

# ========= 配置（按需修改） =========
FILE_PATH = r"D:\LearningDeepLearning\2025Modeling\Data_Processing\男胎检测数据_处理后_filtered.xlsx"
SHEET_NAME = 0  # 默认第一个sheet

COL_Y   = "Y染色体浓度"
COL_W   = "检测孕周"
COL_BMI = "孕妇BMI"

# ========= 主流程 =========
def main():
    # 1) 读取数据
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

    # 2) 取三列并转成数值；非数值→NaN
    sub = pd.DataFrame({
        COL_Y:   pd.to_numeric(df[COL_Y], errors="coerce"),
        COL_W:   pd.to_numeric(df[COL_W], errors="coerce"),
        COL_BMI: pd.to_numeric(df[COL_BMI], errors="coerce"),
    }).dropna(how="any")

    if len(sub) < 3:
        raise RuntimeError("清洗后可用样本数不足（<3）。请检查三列是否为数值且缺失太多。")

    cols = [COL_Y, COL_W, COL_BMI]
    data = sub[cols].to_numpy()

    # 3) 计算 Spearman：返回相关矩阵与 p 值矩阵
    corr, pval = spearmanr(data, axis=0, nan_policy='omit')
    corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    pval_df = pd.DataFrame(pval, index=cols, columns=cols)

    # 4) 输出结果
    print("样本量 N =", len(sub))
    print("\n两两 Spearman 相关系数与 p 值：")
    pairs = [(COL_Y, COL_W), (COL_Y, COL_BMI), (COL_W, COL_BMI)]
    for a, b in pairs:
        print(f"- {a} vs {b}: ρ={corr_df.loc[a,b]:.4f}, p={pval_df.loc[a,b]:.4g}")

    print("\nSpearman 相关系数矩阵：")
    print(corr_df.round(4))

    print("\nSpearman p 值矩阵：")
    print(pval_df.applymap(lambda x: round(x, 6)))

    # 5) 保存矩阵为 CSV
    corr_df.to_csv("spearman_corr_matrix.csv", encoding="utf-8-sig")
    pval_df.to_csv("spearman_pval_matrix.csv", encoding="utf-8-sig")

    # 6) 绘制热力图（matplotlib）
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr_df.values, cmap="YlOrRd", vmin=-1, vmax=1)

    # 坐标轴与刻度
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=0)
    ax.set_yticklabels(cols)

    # 在每个格子里写上数值
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_df.values[i, j]
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center",
                    color="white" if abs(val) > 0.6 else "black")

    # 颜色条
    cbar = plt.colorbar(im)

    plt.tight_layout()
    plt.savefig("spearman_heatmap.png", dpi=200)
    plt.show()

    print(r"\n已保存：spearman_corr_matrix.csv, spearman_pval_matrix.csv, spearman_heatmap.png")

if __name__ == "__main__":
    main()
