import numpy as np
import pandas as pd
from pathlib import Path


def main():
    # === 配置区域 ===
    input_path = Path(r"男胎检测数据_处理后_with_flags.xlsx")
    output_path = input_path.with_name(input_path.stem + "_浮动后.xlsx")
    seed = None  # 例如设为 42 可复现；保持 None 则每次随机不同

    # === 读取数据 ===
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    df = pd.read_excel(input_path)

    # === 基本校验 ===
    required_cols = ["Y染色体浓度", "GC含量"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必要列：{missing}（需要列：{required_cols}）")

    # === 准备随机源 ===
    rng = np.random.default_rng(seed)

    # === 1) Y染色体浓度 上下5%浮动 ===
    y_scale = 1.0 + rng.uniform(-0.05, 0.05, size=len(df))
    df["Y染色体浓度_浮动"] = df["Y染色体浓度"].astype(float) * y_scale

    # === 2) GC含量 上下5%浮动 ===
    gc_scale = 1.0 + rng.uniform(-0.05, 0.05, size=len(df))
    df["GC含量_浮动"] = df["GC含量"].astype(float) * gc_scale
    # 可选：把 GC 限定在 [0, 1] 区间（常见做法）
    df["GC含量_浮动"] = df["GC含量_浮动"].clip(lower=0, upper=1)

    # === 3) 基于浮动后的 Y 判别 NIPT准确性 ===
    y_thresh = 0.04
    df["NIPT准确性"] = np.where(df["Y染色体浓度_浮动"] >= y_thresh, "是", "否")
    df["NIPT准确性0-1变量"] = np.where(df["Y染色体浓度_浮动"] >= y_thresh, 1, 0)

    # === 4) 基于浮动后的 GC 判别 检测准确性 ===
    gc_low, gc_high = 0.4, 0.6
    in_range = (df["GC含量_浮动"] >= gc_low) & (df["GC含量_浮动"] <= gc_high)
    df["检测准确性"] = np.where(in_range, "是", "否")
    df["检测准确性0-1变量"] = np.where(in_range, 1, 0)

    # === 写出结果 ===
    df.to_excel(output_path, index=False)

    # === 结果提示（可选） ===
    n_yes_nipt = int((df["NIPT准确性"] == "是").sum())
    n_yes_gc = int((df["检测准确性"] == "是").sum())
    print(f"已保存：{output_path}")
    print(f"[统计] NIPT准确性=是：{n_yes_nipt} / {len(df)}")
    print(f"[统计] 检测准确性=是：{n_yes_gc} / {len(df)}")


if __name__ == "__main__":
    main()
