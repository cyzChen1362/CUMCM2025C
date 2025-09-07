import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from math import sqrt

# ===== 参数可调 =====
w_min, w_max = 11, 30     # 遍历区间
step = 0.02                # 遍历步长，支持小数
alpha = 0.7               # 权重：越大越重视准确率
min_n = 1                 # 最少样本数门槛
z = 1.96                  # Wilson 置信下界（95%）

# ===== 工具：解析孕周 =====
def to_weeks(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m:
        return round(int(m.group(1)) + int(m.group(2))/7.0, 2)
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m:
        return float(m.group(1))
    m = re.match(r"^\s*\d+(\.\d+)?\s*$", s)
    if m:
        return float(s)
    return np.nan

# ===== 风险计算函数 =====
def process_file(file_path, label):
    df = pd.read_excel(file_path)
    if "检测孕周" not in df.columns or "NIPT准确性" not in df.columns:
        raise RuntimeError(f"{file_path} 缺少必要列：检测孕周 / NIPT准确性")

    df["检测孕周_num"] = df["检测孕周"].apply(to_weeks)
    map_yes = {"是":1,"否":0,"yes":1,"no":0,"Yes":1,"No":0,
               "TRUE":1,"True":1,"true":1,"FALSE":0,"False":0,"false":0,
               "1":1,"0":0}
    df["acc_yes"] = df["NIPT准确性"].astype(str).str.strip().map(map_yes)
    df = df.dropna(subset=["检测孕周_num","acc_yes"]).copy()

    weeks = np.arange(w_min, w_max+step, step).round(2)  # 遍历小数孕周
    rows = []
    for w in weeks:
        sub = df[df["检测孕周_num"] >= w]
        n = len(sub)
        p_hat = sub["acc_yes"].mean() if n>0 else np.nan
        # Wilson 下界
        if n>0 and not np.isnan(p_hat):
            denom = 1 + z**2/n
            center = p_hat + z**2/(2*n)
            radius = z*sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n)
            p_lb = (center - radius)/denom
            p_lb = max(0.0, min(1.0, p_lb))
        else:
            p_lb = np.nan
        rows.append((w, n, p_hat, p_lb))

    res = pd.DataFrame(rows, columns=["week","n","p_hat","p_lb"])
    def norm_week(w): return (w - w_min) / (w_max - w_min)
    res["R_practical"]   = alpha*(1-res["p_hat"]) + (1-alpha)*res["week"].map(norm_week)
    res["R_conservative"] = alpha*(1-res["p_lb"])  + (1-alpha)*res["week"].map(norm_week)
    res.loc[res["n"] < min_n, ["R_practical","R_conservative"]] = np.nan
    res["group"] = label

    # === 找出最佳孕周（Practical / Conservative） ===
    best_prac = res.loc[res["R_practical"].idxmin()] if res["R_practical"].notna().any() else None
    best_cons = res.loc[res["R_conservative"].idxmin()] if res["R_conservative"].notna().any() else None

    print(f"\n===== {label} =====")
    if best_prac is not None:
        print(f"Practical 推荐孕周: {best_prac['week']:.2f} 周 "
              f"(样本={int(best_prac['n'])}, p̂={best_prac['p_hat']:.3f}, R={best_prac['R_practical']:.4f})")
    else:
        print("Practical 推荐孕周: 无可用值")
    if best_cons is not None:
        print(f"Conservative 推荐孕周: {best_cons['week']:.2f} 周 "
              f"(样本={int(best_cons['n'])}, p_LB={best_cons['p_lb']:.3f}, R={best_cons['R_conservative']:.4f})")
    else:
        print("Conservative 推荐孕周: 无可用值")

    return res

# ===== 循环处理五个分组文件 =====
files = {
    "Group1": "BMI_group1.xlsx",
    "Group2": "BMI_group2.xlsx",
    "Group3": "BMI_group3.xlsx",
    "Group4": "BMI_group4.xlsx",
    "Group5": "BMI_group5.xlsx"
}

results = []
for label, path in files.items():
    res = process_file(path, label)
    results.append(res)

all_res = pd.concat(results, ignore_index=True)

# ===== 画图（每组两条曲线） =====
plt.figure(figsize=(10,6))
for label in files.keys():
    sub = all_res[all_res["group"] == label]
    plt.plot(sub["week"], sub["R_practical"], label=f"{label} - Practical")
    plt.plot(sub["week"], sub["R_conservative"], linestyle="--", label=f"{label} - Conservative")

plt.xlabel("Gestational age threshold (weeks)")
plt.ylabel("Risk")
plt.title("Time–Accuracy Tradeoff: Risk vs Week (5 BMI groups)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ===== 保存结果 =====
all_res.to_csv("risk_by_week_all_groups.csv", index=False, encoding="utf-8-sig")
