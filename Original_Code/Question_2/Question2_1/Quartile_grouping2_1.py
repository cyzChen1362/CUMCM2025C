import pandas as pd

# === 1. 读取数据 ===
file_path = "男胎检测数据_处理后.xlsx"  # 输入文件路径
df = pd.read_excel(file_path)

# === 2. 确保存在孕妇BMI列（如果没有则根据身高体重计算） ===
if "孕妇BMI" not in df.columns and {"身高", "体重"}.issubset(df.columns):
    df["孕妇BMI"] = df["体重"] / (df["身高"] / 100.0) ** 2

# === 3. 固定 BMI 分组阈值 ===
bins = [26.6, 34.3, 37.4, 39.7, 43.1, 46.9]
labels = ["Group1", "Group2", "Group3", "Group4", "Group5"]

# === 4. 分组 ===
df["BMI_group"] = pd.cut(df["孕妇BMI"], bins=bins, labels=labels, right=False, include_lowest=True)

group1 = df[df["BMI_group"] == "Group1"]
group2 = df[df["BMI_group"] == "Group2"]
group3 = df[df["BMI_group"] == "Group3"]
group4 = df[df["BMI_group"] == "Group4"]
group5 = df[df["BMI_group"] == "Group5"]

# === 5. 输出五个新的表格 ===
group1.to_excel("BMI_group2_1.xlsx", index=False)
group2.to_excel("BMI_group2_2.xlsx", index=False)
group3.to_excel("BMI_group2_3.xlsx", index=False)
group4.to_excel("BMI_group2_4.xlsx", index=False)
group5.to_excel("BMI_group2_5.xlsx", index=False)

# === 6. 打印每组样本数量和BMI区间范围 ===
def group_stats(name, data):
    if len(data) > 0:
        print(f"{name}: 样本数 = {len(data)}, BMI范围 = [{data['孕妇BMI'].min():.2f}, {data['孕妇BMI'].max():.2f}]")
    else:
        print(f"{name}: 样本数 = 0")

group_stats("Group1 (26.6-34.3)", group1)
group_stats("Group2 (34.3-37.4)", group2)
group_stats("Group3 (37.4-39.7)", group3)
group_stats("Group4 (39.7-43.1)", group4)
group_stats("Group5 (43.1-46.9)", group5)

print("分组完成！输出文件：BMI_group2_1.xlsx, ..., BMI_group2_5.xlsx")
