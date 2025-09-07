import pandas as pd
import numpy as np
import statsmodels.api as sm

# ======== 1) 读数 ========
file_path = r"D:/LearningDeepLearning/2025Modeling/Data_Processing/男胎检测数据_处理后_filtered.xlsx"
df = pd.read_excel(file_path)

# 若无 BMI，则计算 BMI
if "BMI" not in df.columns and {"身高", "体重"}.issubset(df.columns):
    df["BMI"] = df["体重"] / (df["身高"] / 100.0) ** 2

df = df[["检测孕周", "BMI", "Y染色体浓度"]].dropna().copy()

GW  = df["检测孕周"].to_numpy(dtype=float)
BMI = df["BMI"].to_numpy(dtype=float)
y   = df["Y染色体浓度"].to_numpy(dtype=float)

# ======== 2) 将 GW/BMI 映射到角度 [0, 2π] ========
def to_angle(x):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax == xmin:
        return np.zeros_like(x), xmin, xmax
    theta = 2.0 * np.pi * (x - xmin) / (xmax - xmin)
    return theta, xmin, xmax

theta_gw,  gw_min,  gw_max  = to_angle(GW)
theta_bmi, bmi_min, bmi_max = to_angle(BMI)

# ======== 3) 生成三角基特征 ========
def fourier_design(theta_gw, theta_bmi, K_gw=3, K_bmi=3, include_interactions=True):
    n = len(theta_gw)
    feats = [np.ones(n)]
    names = ["const"]

    # GW 一维谐波
    for k in range(1, K_gw + 1):
        feats.append(np.sin(k * theta_gw)); names.append(f"sin({k}*θ_gw)")
        feats.append(np.cos(k * theta_gw)); names.append(f"cos({k}*θ_gw)")

    # BMI 一维谐波
    for l in range(1, K_bmi + 1):
        feats.append(np.sin(l * theta_bmi)); names.append(f"sin({l}*θ_bmi)")
        feats.append(np.cos(l * theta_bmi)); names.append(f"cos({l}*θ_bmi)")

    # 交互项（适度）
    if include_interactions:
        KK = min(K_gw, K_bmi, 3)  # 控制复杂度，避免爆炸
        for k in range(1, KK + 1):
            for l in range(1, KK + 1):
                feats.append(np.sin(k*theta_gw) * np.sin(l*theta_bmi))
                names.append(f"sin({k}*θ_gw)*sin({l}*θ_bmi)")
                feats.append(np.cos(k*theta_gw) * np.cos(l*theta_bmi))
                names.append(f"cos({k}*θ_gw)*cos({l}*θ_bmi)")

    X = np.column_stack(feats)
    return X, names

# ======== 4) 设定阶数（你可以改大/改小） ========
K_GW  = 500   # GW 的谐波最高阶
K_BMI = 500   # BMI 的谐波最高阶
INCLUDE_INTERACTIONS = True  # 是否加入交互项

X, names = fourier_design(theta_gw, theta_bmi, K_GW, K_BMI, INCLUDE_INTERACTIONS)

# ======== 5) 拟合（最大化训练 R²） ========
model = sm.OLS(y, X).fit()
print("训练 R² = %.4f, 调整R² = %.4f" % (model.rsquared, model.rsquared_adj))
print(model.summary(xname=names))

# ======== 6) 打印“可直接写进报告”的显式公式 ========
# θ_gw = 2π*(GW - gw_min)/(gw_max - gw_min)
# θ_bmi = 2π*(BMI - bmi_min)/(bmi_max - bmi_min)
def explicit_formula(names, params, gw_min, gw_max, bmi_min, bmi_max):
    lines = []
    lines.append("显式三角函数模型：")
    lines.append("令 θ_gw = 2π*(GW - %.6g)/(%.6g - %.6g)" % (gw_min, gw_max, gw_min))
    lines.append("    θ_bmi = 2π*(BMI - %.6g)/(%.6g - %.6g)" % (bmi_min, bmi_max, bmi_min))
    lines.append("则：")
    # y = Σ β_j * basis_j(θ_gw, θ_bmi)
    terms = []
    for name, beta in zip(names, model.params):
        coef = float(beta)
        if abs(coef) < 1e-12:
            continue
        if name == "const":
            terms.append(f"{coef:.6g}")
        else:
            terms.append(f"({coef:.6g})*{name}")
    if not terms:
        body = "0"
    else:
        body = " + ".join(terms)
    lines.append("y = " + body)
    return "\n".join(lines)

print("\n" + explicit_formula(names, model.params, gw_min, gw_max, bmi_min, bmi_max))
