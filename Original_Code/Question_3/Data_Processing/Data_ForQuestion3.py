import pandas as pd
from pathlib import Path

# ===== 1) 输入文件路径 =====
INPUT_PATH = r"男胎检测数据_处理后.xlsx"  # ← 改成你的文件路径

# ===== 2) 读取数据 =====
input_path = Path(INPUT_PATH)
if not input_path.exists():
    raise FileNotFoundError(f"找不到输入文件：{input_path}")

if input_path.suffix.lower() in [".xlsx", ".xls"]:
    df = pd.read_excel(input_path)
elif input_path.suffix.lower() == ".csv":
    df = pd.read_csv(input_path)
else:
    raise ValueError("仅支持 .xlsx / .xls / .csv 文件")

# ===== 3) 确保关键列存在，并转为数值 =====
required_cols = ["Y染色体浓度", "GC含量"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"缺少必要列：{missing}；请检查表头是否为：{required_cols}")

# 将两列安全转为数值，无法转换的设为 NaN
df["Y染色体浓度_num"] = pd.to_numeric(df["Y染色体浓度"], errors="coerce")
df["GC含量_num"]   = pd.to_numeric(df["GC含量"], errors="coerce")

# ===== 4) 规则计算 =====
# 规则 1 & 2：阈值 0.04
cond_nipt = df["Y染色体浓度_num"] >= 0.04
df["NIPT准确性"] = cond_nipt.map({True: "是", False: "否"})
df["NIPT准确性0-1变量"] = cond_nipt.astype(int).astype(str)  # True->1, False->0，再转字符串

# 规则 3 & 4：GC 含量处于 [0.4, 0.6]（含边界）
cond_gc = (df["GC含量_num"] >= 0.4) & (df["GC含量_num"] <= 0.6)
df["检测准确性"] = cond_gc.map({True: "是", False: "否"})
df["检测准确性0-1变量"] = cond_gc.astype(int).astype(str)

# 如果你不想保留中间的 *_num 列，把下面两行取消注释：
# df.drop(columns=["Y染色体浓度_num", "GC含量_num"], inplace=True)

# ===== 5) 导出结果 =====
out_path = input_path.with_stem(input_path.stem + "_with_flags")
if input_path.suffix.lower() in [".xlsx", ".xls"]:
    df.to_excel(out_path, index=False)
else:
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"已处理完成并保存到：{out_path}")
