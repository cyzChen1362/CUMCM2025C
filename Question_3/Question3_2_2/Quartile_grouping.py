import pandas as pd

# === 1) 读取数据 ===
file_path = "男胎检测数据_处理后_with_flags_浮动后.xlsx"
df = pd.read_excel(file_path)

# === 2) 确保有“孕妇BMI”列===
if "孕妇BMI" not in df.columns and {"身高", "体重"}.issubset(df.columns):
    df["孕妇BMI"] = df["体重"] / (df["身高"] / 100.0) ** 2

# === 3) 按图片给定的固定分组阈值===
# [20.7,29.2], [29.2,37.0], [37.0,38.6], [38.6,42.8], [42.8,46.9]
bins   = [20.7, 29.2, 37.0, 38.6, 42.8, 46.9]
labels = ["Group1", "Group2", "Group3", "Group4", "Group5"]
df["BMI_group"] = pd.cut(
    df["孕妇BMI"],
    bins=bins,
    labels=labels,
    right=True,
    include_lowest=True
)

# === 4) 拆出各组并保存 ===
groups = {lab: df[df["BMI_group"] == lab] for lab in labels}
for i, lab in enumerate(labels, 1):
    groups[lab].to_excel(f"BMI_{lab.lower()}.xlsx", index=False)

# === 5) 统计每组样本数与实际BMI范围 ===
def group_stats(name, data):
    if len(data) > 0:
        print(f"{name}: 样本数={len(data)}, BMI范围=[{data['孕妇BMI'].min():.2f}, {data['孕妇BMI'].max():.2f}]")
    else:
        print(f"{name}: 样本数=0")

interval_names = [
    "Group1 [20.7, 29.2]",
    "Group2 [29.2, 37.0]",
    "Group3 [37.0, 38.6]",
    "Group4 [38.6, 42.8]",
    "Group5 [42.8, 46.9]",
]
for lab, shown in zip(labels, interval_names):
    group_stats(shown, groups[lab])

# === 6) 可选：提示落在区间外的数据量 ===
n_out = df["BMI_group"].isna().sum()
if n_out > 0:
    print(f"注意：有 {n_out} 条样本的BMI不在给定区间[20.7,46.9]内，未被分组。")

print("分组完成！输出文件：BMI_group1.xlsx~BMI_group5.xlsx")
