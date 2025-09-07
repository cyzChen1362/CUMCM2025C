# -*- coding: utf-8 -*-
import pandas as pd
from itertools import combinations

def main(
    input_path=r"D:\LearningDeepLearning\2025Modeling\Data_Processing\男胎检测数据_处理后_filtered.xlsx",
    output_path=r"男胎检测数据_处理后_filtered_pairwise.xlsx",
    sheet_name=0  # 如需指定工作表名可改为字符串
):
    # ---------- 1) 只保留指定列 ----------
    cols_keep = [
        "Y染色体浓度",
        "检测孕周",
        "孕妇BMI",
        "孕周*BMI",
        "原始读段数",
        "在参考基因组上比对的比例",
        "重复读段的比例",
        "唯一比对的读段数",
        "GC含量",
        "13号染色体的GC含量",
        "18号染色体的GC含量",
        "21号染色体的GC含量",
        "被过滤掉读段数的比例",
        "生产次数",
        "检测抽血次数",
        "年龄",
        "身高",
        "体重",
    ]

    df_raw = pd.read_excel(input_path, sheet_name=sheet_name)
    # 提示缺失列但继续运行（缺哪列就不保留哪列）
    missing = [c for c in cols_keep if c not in df_raw.columns]
    if missing:
        print("警告：下列指明要保留的列在输入表中未找到，将被忽略：", missing)

    existing_cols = [c for c in cols_keep if c in df_raw.columns]
    df = df_raw[existing_cols].copy()

    # ---------- 2) 列转数值（尽可能），非数值置为 NaN ----------
    # 两两相乘需要数值型
    for c in df.columns:
        # 仅尝试将“百分比”“比例”“读段数”等转成数值；如果是字符串百分号也能处理
        df[c] = pd.to_numeric(
            df[c].astype(str)
                 .str.replace('%', '', regex=False)
                 .str.replace(',', '', regex=False)
                 .str.strip(),
            errors='coerce'
        )

    # ---------- 3) 生成两两相乘的新列（排除 Y染色体浓度） ----------
    target_exclude = "Y染色体浓度"
    base_cols = [c for c in df.columns if c != target_exclude]

    # 逐对生成乘积列，列名形如 "列A*列B"
    for a, b in combinations(base_cols, 2):
        new_col = f"{a}*{b}"
        df[new_col] = df[a] * df[b]

    # ---------- 4) 特别保证“检测孕周*孕妇BMI”一致（若存在同名列则覆盖计算值） ----------
    col_a, col_b = "检测孕周", "孕妇BMI"
    if col_a in df.columns and col_b in df.columns:
        prod_name = f"{col_a}*{col_b}"  # 即“检测孕周*孕妇BMI”
        df[prod_name] = df[col_a] * df[col_b]

        # 如果你希望同步一个更短名字“孕周*BMI”，也可以额外写入（可选）：
        short_name = "孕周*BMI"
        df[short_name] = df[prod_name]

    # ---------- 5) 保存结果 ----------
    df.to_excel(output_path, index=False)
    print(f"已完成。输出文件：{output_path}")

if __name__ == "__main__":
    main()
