# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter, gaussian_filter

# ========= 可调参数 =========
FILE_PATH = r"/Original_Code/Data_Processing/男胎检测数据_处理后_filtered.xlsx"
K_GW  = 12000
K_BMI = 12000
INCLUDE_INTERACTIONS = True

# 平滑强度
MED_WIN_1D = 9
SG_WIN_1D  = 31
SG_POLY_1D = 3
GAUSS_SIGMA_2D = 1.6
GAUSS_PASSES   = 2

# 稳健显示范围
Y_Q_LOW, Y_Q_HIGH = 1, 99
Y_PADDING_RATIO = 0.10

# ========= 1) 读数 =========
df = pd.read_excel(FILE_PATH)
if "BMI" not in df.columns and {"身高", "体重"}.issubset(df.columns):
    df["BMI"] = df["体重"] / (df["身高"] / 100.0) ** 2
df = df[["检测孕周", "BMI", "Y染色体浓度"]].dropna().copy()

GW  = df["检测孕周"].to_numpy(dtype=float)
BMI = df["BMI"].to_numpy(dtype=float)
y   = df["Y染色体浓度"].to_numpy(dtype=float)

# ========= 2) 映射到角度 =========
def to_angle(x):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax == xmin:
        return np.zeros_like(x), xmin, xmax
    theta = 2.0 * np.pi * (x - xmin) / (xmax - xmin)
    return theta, xmin, xmax

theta_gw,  gw_min,  gw_max  = to_angle(GW)
theta_bmi, bmi_min, bmi_max = to_angle(BMI)

# ========= 3) 三角基特征 =========
def fourier_design(theta_gw, theta_bmi, K_gw=3, K_bmi=3, include_interactions=True):
    n = len(theta_gw)
    feats = [np.ones(n)]
    names = ["const"]
    for k in range(1, K_gw + 1):
        feats.append(np.sin(k * theta_gw)); names.append(f"sin({k}*θ_gw)")
        feats.append(np.cos(k * theta_gw)); names.append(f"cos({k}*θ_gw)")
    for l in range(1, K_bmi + 1):
        feats.append(np.sin(l * theta_bmi)); names.append(f"sin({l}*θ_bmi)")
        feats.append(np.cos(l * theta_bmi)); names.append(f"cos({l}*θ_bmi)")
    if include_interactions:
        KK = min(K_gw, K_bmi, 3)
        for k in range(1, KK + 1):
            for l in range(1, KK + 1):
                feats.append(np.sin(k*theta_gw) * np.sin(l*theta_bmi))
                names.append(f"sin({k}*θ_gw)*sin({l}*θ_bmi)")
                feats.append(np.cos(k*theta_gw) * np.cos(l*theta_bmi))
                names.append(f"cos({k}*θ_gw)*cos({l}*θ_bmi)")
    X = np.column_stack(feats)
    return X, names

# ========= 4) 设计矩阵 =========
X, names = fourier_design(theta_gw, theta_bmi, K_GW, K_BMI, INCLUDE_INTERACTIONS)

# ========= 5) 拟合 =========
model = sm.OLS(y, X).fit()
print("训练 R² = %.4f, 调整R² = %.4f" % (model.rsquared, model.rsquared_adj))
y_hat_train = X @ model.params  # 训练点拟合值

# ========= 5.1 稳健显示范围 =========
y_lo, y_hi = np.percentile(y, [Y_Q_LOW, Y_Q_HIGH])
pad = Y_PADDING_RATIO * (y_hi - y_lo + 1e-12)
y_min_plot = y_lo - pad
y_max_plot = y_hi + pad

# ========= 6) 显式可写公式 =========
def explicit_formula(names, params, gw_min, gw_max, bmi_min, bmi_max):
    lines = []
    lines.append("显式三角函数模型：")
    lines.append("令 θ_gw = 2π*(GW - %.6g)/(%.6g - %.6g)" % (gw_min, gw_max, gw_min))
    lines.append("    θ_bmi = 2π*(BMI - %.6g)/(%.6g - %.6g)" % (bmi_min, bmi_max, bmi_min))
    lines.append("则：")
    terms = []
    for name, beta in zip(names, params):
        coef = float(beta)
        if abs(coef) < 1e-12:
            continue
        if name == "const":
            terms.append(f"{coef:.6g}")
        else:
            terms.append(f"({coef:.6g})*{name}")
    lines.append("y = " + (" + ".join(terms) if terms else "0"))
    return "\n" .join(lines)

print("\n" + explicit_formula(names, model.params, gw_min, gw_max, bmi_min, bmi_max))

# ========= 强力平滑工具（供 2D 曲线用） =========
def winsorize(arr, lo=1, hi=99):
    lo_v, hi_v = np.percentile(arr, [lo, hi])
    return np.clip(arr, lo_v, hi_v)

def smooth_1d(series, winsor_lo=1, winsor_hi=99, med_win=9, sg_win=31, sg_poly=3):
    from scipy.ndimage import median_filter as _median_filter
    from scipy.signal import savgol_filter as _savgol
    s = winsorize(np.asarray(series), winsor_lo, winsor_hi)
    med_win = max(1, med_win if med_win % 2 == 1 else med_win + 1)
    if med_win > 1 and len(s) >= med_win:
        s = _median_filter(s, size=med_win, mode='nearest')
    sg_win = max(5, sg_win if sg_win % 2 == 1 else sg_win + 1)
    if len(s) >= sg_win:
        try:
            s = _savgol(s, window_length=sg_win, polyorder=min(sg_poly, sg_win-1))
        except Exception:
            pass
    return s

# ========= 7A) 3D：仅散点（条件着色/隐藏），无曲面 =========
# 训练点拟合值用于显示时的裁剪（与坐标一致）
y_hat_train_s = np.clip(y_hat_train, y_min_plot, y_max_plot)

# ——按相对误差筛选（≤5% → 绿点，且隐藏对应蓝点）——
denom = np.maximum(np.abs(y), 1e-12)          # 防除零
rel_err = np.abs(y_hat_train_s - y) / denom
good = rel_err <= 0.05
bad  = ~good

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绿点（拟合误差 ≤5% 的观测点；对应蓝点隐藏）
if np.any(good):
    ax.scatter(GW[good], BMI[good], y[good], s=8, alpha=0.9, c='green', label='observed (≤5% error)')

# 红点（误差 >5% 的观测点）
if np.any(bad):
    ax.scatter(GW[bad], BMI[bad], y[bad], s=8, alpha=0.9, c='red', label='observed (>5% error)')
    # 蓝点只画“误差 >5%”那一部分
    ax.scatter(GW[bad], BMI[bad], y_hat_train_s[bad], s=10, alpha=0.9, c='blue', marker='^', label='fitted@train (>5% only)')

ax.set_xlabel('Gestational Week (GW)')
ax.set_ylabel('BMI')
ax.set_zlabel('Y-chromosome concentration')
ax.set_title('3D points: green=observed (≤5%), red=observed (>5%), blue▲=fitted (>5% only)')
ax.set_zlim(y_min_plot, y_max_plot)
ax.legend(loc='best')
plt.tight_layout()

# ========= 7B) 2D 切片：y–GW（固定 BMI） =========
bmi_qs = np.quantile(BMI, [0.25, 0.50, 0.75])
gw_line = np.linspace(gw_min, gw_max, 200)
theta_gw_line = 2.0 * np.pi * (gw_line - gw_min) / (gw_max - gw_min + 1e-12)

plt.figure(figsize=(9, 6))
plt.scatter(GW, y, s=10, alpha=0.5, label='data')

for q in bmi_qs:
    theta_bmi_line = 2.0 * np.pi * (np.full_like(gw_line, q) - bmi_min) / (bmi_max - bmi_min + 1e-12)
    X_line, _ = fourier_design(theta_gw_line, theta_bmi_line, K_GW, K_BMI, INCLUDE_INTERACTIONS)
    y_line = X_line @ model.params
    y_line = smooth_1d(y_line, winsor_lo=Y_Q_LOW, winsor_hi=Y_Q_HIGH,
                       med_win=MED_WIN_1D, sg_win=SG_WIN_1D, sg_poly=SG_POLY_1D)
    y_line = np.clip(y_line, y_min_plot, y_max_plot)
    plt.plot(gw_line, y_line, linewidth=2, label=f'fit @ BMI={q:.2f}')

plt.xlabel('Gestational Week (GW)')
plt.ylabel('Y-chromosome concentration')
plt.title('Fitted curves vs GW (robust-smoothed)')
plt.ylim(y_min_plot, y_max_plot)
plt.legend()
plt.tight_layout()

# ========= 7C) 2D 切片：y–BMI（固定 GW） =========
gw_qs = np.quantile(GW, [0.25, 0.50, 0.75])
bmi_line = np.linspace(bmi_min, bmi_max, 200)
theta_bmi_line2 = 2.0 * np.pi * (bmi_line - bmi_min) / (bmi_max - bmi_min + 1e-12)

plt.figure(figsize=(9, 6))
plt.scatter(BMI, y, s=10, alpha=0.5, label='data')

for q in gw_qs:
    theta_gw_line2 = 2.0 * np.pi * (np.full_like(bmi_line, q) - gw_min) / (gw_max - gw_min + 1e-12)
    X_line2, _ = fourier_design(theta_gw_line2, theta_bmi_line2, K_GW, K_BMI, INCLUDE_INTERACTIONS)
    y_line2 = X_line2 @ model.params
    y_line2 = smooth_1d(y_line2, winsor_lo=Y_Q_LOW, winsor_hi=Y_Q_HIGH,
                        med_win=MED_WIN_1D, sg_win=SG_WIN_1D, sg_poly=SG_POLY_1D)
    y_line2 = np.clip(y_line2, y_min_plot, y_max_plot)
    plt.plot(bmi_line, y_line2, linewidth=2, label=f'fit @ GW={q:.2f}')

plt.xlabel('BMI')
plt.ylabel('Y-chromosome concentration')
plt.title('Fitted curves vs BMI (robust-smoothed)')
plt.ylim(y_min_plot, y_max_plot)
plt.legend()
plt.tight_layout()

plt.show()
