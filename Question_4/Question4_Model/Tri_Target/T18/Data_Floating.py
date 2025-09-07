import pandas as pd
import numpy as np

# === 1. 读取文件 ===
file_path = "balanced_组合采样.xlsx"
df = pd.read_excel(file_path)

# === 2. 需要浮动的列 ===
cols_to_float = [
    "原始读段数","检测抽血次数",
     "X染色体的Z值", "18号染色体的Z值",
    "13号染色体的Z值", "被过滤掉读段数的比例",
     "X染色体浓度", "21号染色体的Z值"
]

# === 3. 对指定列应用上下浮动10% ===
df_float = df.copy()
for col in cols_to_float:
    if col in df_float.columns:
        noise = np.random.uniform(0.9, 1.1, size=len(df_float))
        df_float[col] = df_float[col] * noise

# === 4. 保存结果 ===
output_path = "balanced_组合采样_浮动后.xlsx"
df_float.to_excel(output_path, index=False)

print(f"处理完成，已保存为: {output_path}")
