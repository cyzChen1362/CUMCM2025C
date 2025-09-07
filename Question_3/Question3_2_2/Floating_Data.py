import numpy as np
import pandas as pd

input_path  = r"男胎检测数据_处理后_with_flags.xlsx"
output_path = None
random_seed = 42

# ===== 列名容错=====
Y_COL_CANDIDATES  = ["Y染色体浓度", "Y 染色体浓度", "y_frac", "Y染色体含量"]
GC_COL_CANDIDATES = ["GC含量", "GC 含量", "gc_global"]

def pick_column_name(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"未在表格中找到目标列，候选名：{candidates}\n现有列：{list(df.columns)}")

def main():
    # 读取
    df = pd.read_excel(input_path)

    # 找到目标列
    y_col  = pick_column_name(df, Y_COL_CANDIDATES)
    gc_col = pick_column_name(df, GC_COL_CANDIDATES)

    # 随机种子
    if random_seed is not None:
        np.random.seed(random_seed)

    # --- 1) Y染色体浓度 下调 10% ---
    y_new_col = "Y染色体浓度_下调10%"
    df[y_new_col] = df[y_col] * 0.90

    # --- 2) GC含量 上下 10% 浮动 ---
    gc_new_col = "GC含量_浮动±10%"
    # 对每行生成一个 1 + U(-0.1, 0.1) 的乘子
    multipliers = 1.0 + np.random.uniform(-0.1, 0.1, size=len(df))
    df[gc_new_col] = df[gc_col] * multipliers
    # （可选）将 GC 限制在 [0,1] 区间
    df[gc_new_col] = df[gc_new_col].clip(lower=0.0, upper=1.0)

    # --- 3) 基于调整后的 Y 判别 NIPT准确性 ---
    cond_y = (df[y_new_col] >= 0.04)
    cond_y = cond_y.fillna(False)  # NaN 视为未达标
    df["NIPT准确性"] = np.where(cond_y, "是", "否")
    df["NIPT准确性0-1变量"] = np.where(cond_y, 1, 0)

    # --- 4) 基于调整后的 GC 判别 检测准确性 ---
    cond_gc = (df[gc_new_col] >= 0.4) & (df[gc_new_col] <= 0.6)
    cond_gc = cond_gc.fillna(False)
    df["检测准确性"] = np.where(cond_gc, "是", "否")
    df["检测准确性0-1变量"] = np.where(cond_gc, 1, 0)

    # 保存
    if output_path is None:
        if "." in input_path:
            base = input_path.rsplit(".", 1)[0]
            ext  = input_path.rsplit(".", 1)[1]
            out  = f"{base}_浮动后.{ext}"
        else:
            out  = f"{input_path}_浮动后.xlsx"
    else:
        out = output_path

    df.to_excel(out, index=False)
    print(f"已完成处理并保存到：{out}")

if __name__ == "__main__":
    main()
