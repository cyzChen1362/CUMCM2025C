import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ===== 工具：统一孕周格式 =====
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

# ===== 读取数据 =====
file_path = r"男胎检测数据_处理后_filtered.xlsx"
df = pd.read_excel(file_path)

# 如果没有 BMI，则根据身高体重计算
if "孕妇BMI" not in df.columns and {"身高", "体重"}.issubset(df.columns):
    df["孕妇BMI"] = df["体重"] / (df["身高"]/100.0)**2

df["检测孕周_num"] = df["检测孕周"].apply(to_weeks)

# 处理 NIPT准确性
acc_raw = df["NIPT准确性"].astype(str).str.strip()
map_yes = {"是":1,"否":0,"yes":1,"no":0,"Yes":1,"No":0,
           "TRUE":1,"True":1,"true":1,"FALSE":0,"False":0,"false":0,
           "1":1,"0":0}
df["acc_yes"] = acc_raw.map(map_yes)

df_valid = df.dropna(subset=["检测孕周_num","acc_yes","孕妇BMI"]).copy()

# ===== BMI 阈值范围 =====
bmi_min, bmi_max = df_valid["孕妇BMI"].min(), df_valid["孕妇BMI"].max()

# 选取候选分界点（比如 5 个 q1，5 个 q2 -> 最多 25 种分法）
q1_candidates = np.linspace(bmi_min+1, bmi_max-2, 5)  # 第一分界点候选
q2_candidates = np.linspace(bmi_min+2, bmi_max-1, 5)  # 第二分界点候选

thresholds = np.arange(10, 30.01, 0.2).round(2)

# ===== 遍历候选分界点，画图 =====
plot_count = 0
for q1 in q1_candidates:
    for q2 in q2_candidates:
        if q2 <= q1:
            continue
        plot_count += 1
        if plot_count > 20:  # 限制最多画 20 张
            break

        # 三组划分
        group_low  = df_valid[df_valid["孕妇BMI"] <= q1]
        group_mid  = df_valid[(df_valid["孕妇BMI"] > q1) & (df_valid["孕妇BMI"] <= q2)]
        group_high = df_valid[df_valid["孕妇BMI"] > q2]

        groups = {"Low BMI": group_low, "Mid BMI": group_mid, "High BMI": group_high}

        plt.figure(figsize=(8,5))
        for gname, sub_df in groups.items():
            props = []
            for t in thresholds:
                sub = sub_df[sub_df["检测孕周_num"] >= t]
                if len(sub) == 0:
                    props.append(np.nan)
                else:
                    props.append(sub["acc_yes"].mean())
            plt.plot(thresholds, props, label=gname)

        plt.xlim(10,30)
        plt.ylim(0,1)
        plt.xlabel("Gestational age threshold (weeks)")
        plt.ylabel("Proportion of 'Yes'")
        plt.title(f"NIPT Proportion (q1={q1:.1f}, q2={q2:.1f})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

print(f"共绘制 {plot_count} 张图（每张对应一种分法）。")
