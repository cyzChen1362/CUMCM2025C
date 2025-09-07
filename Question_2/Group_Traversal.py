import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# -------- 1) 将“检测孕周”统一为数值（小数周） --------
def to_weeks(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    m_plus = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m_plus:
        return round(int(m_plus.group(1)) + int(m_plus.group(2))/7.0, 2)
    m_w = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m_w:
        return float(m_w.group(1))
    m_num = re.match(r"^\s*\d+(\.\d+)?\s*$", s)
    if m_num:
        return float(s)
    return np.nan

# -------- 2) 读取原始文件 --------
file_path = "男胎检测数据_处理后_filtered.xlsx"
df = pd.read_excel(file_path)

# 如果没有 BMI，则根据身高体重计算
if "孕妇BMI" not in df.columns and {"身高", "体重"}.issubset(df.columns):
    df["孕妇BMI"] = df["体重"] / (df["身高"] / 100.0) ** 2

# 转换孕周
df["检测孕周_num"] = df["检测孕周"].apply(to_weeks)

# 处理 NIPT准确性
acc_raw = df["NIPT准确性"].astype(str).str.strip()
map_yes = {"是":1,"否":0,"yes":1,"no":0,"Yes":1,"No":0,
           "TRUE":1,"True":1,"true":1,"FALSE":0,"False":0,"false":0,
           "1":1,"0":0}
df["acc_yes"] = acc_raw.map(map_yes)

df_valid = df.dropna(subset=["检测孕周_num","acc_yes","孕妇BMI"]).copy()

# -------- 3) 固定 BMI 分组阈值 --------
bins = [26.6, 34.3, 37.4, 39.7, 43.1, 46.9]
labels = ["Group1", "Group2", "Group3", "Group4", "Group5"]

df_valid["BMI_group"] = pd.cut(df_valid["孕妇BMI"], bins=bins, labels=labels, right=False, include_lowest=True)

# -------- 4) 遍历孕周并计算比例 --------
thresholds = np.arange(10, 30.01, 0.2).round(2)

results = []
for label in labels:
    sub_df = df_valid[df_valid["BMI_group"] == label]
    props, counts = [], []
    for t in thresholds:
        sub = sub_df[sub_df["检测孕周_num"] >= t]
        n = len(sub)
        counts.append(n)
        if n == 0:
            props.append(np.nan)
        else:
            props.append(sub["acc_yes"].mean())
    result = pd.DataFrame({
        "week_threshold": thresholds,
        "prop_yes": props,
        "n_samples": counts,
        "group": label
    })
    results.append(result)

# 合并所有结果
all_results = pd.concat(results, ignore_index=True)

# -------- 5) 绘制对比折线图 --------
plt.figure(figsize=(9, 6))
for label in labels:
    sub = all_results[all_results["group"] == label]
    plt.plot(sub["week_threshold"], sub["prop_yes"], linewidth=1.5, label=label)

plt.xlim(10, 30)
plt.ylim(0, 1)
plt.xlabel("Gestational age threshold (weeks)")
plt.ylabel("Proportion of 'Yes'")
plt.title("Proportion of NIPT = 'Yes' for GA ≥ threshold (fixed BMI 5 groups)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------- 6) 导出结果表格 --------
all_results.to_csv("prop_yes_by_week_fixed_BMI5.csv", index=False, encoding="utf-8-sig")

print("完成！结果保存到 prop_yes_by_week_fixed_BMI5.csv")
