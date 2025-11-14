import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)

FILE_PATH = r"../balanced_组合采样.xlsx"
FLOATED_FILE_PATH = r"../balanced_组合采样_浮动后.xlsx"
TARGET_COL = "13号染色体是否正常"

RAW_FEATURES = [
    "原始读段数",
    "检测抽血次数",
    "X染色体的Z值",
    "18号染色体的Z值",
    "13号染色体的Z值",
    "被过滤掉读段数的比例",
    "X染色体浓度",
    "21号染色体的Z值",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

def to_weeks(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2))
        return round(w + d / 7.0, 2)
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m:
        return float(m.group(1))
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1))
    return np.nan

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def latex_escape(s: str) -> str:
    return (s.replace('_', r'\_')
             .replace('%', r'\%')
             .replace('&', r'\&')
             .replace('#', r'\#'))

df = pd.read_excel(FILE_PATH)
df = strip_columns(df)

if TARGET_COL not in df.columns:
    raise KeyError(f"未找到目标列：{TARGET_COL}")
if "检测孕周" in df.columns:
    df["检测孕周"] = df["检测孕周"].apply(to_weeks)
missing_features = [c for c in RAW_FEATURES if c not in df.columns]
if missing_features:
    raise KeyError(f"训练集缺失特征列：{missing_features}")

use_cols = [TARGET_COL] + RAW_FEATURES
df_use = df[use_cols].copy()

y_map = {"是": 1, "否": 0, 1: 1, 0: 0}
y = df_use[TARGET_COL].map(y_map)
if y.isna().any():
    bad_rows = df_use[y.isna()]
    print("警告：训练集中存在无法映射为二元标签的行，这些行将被丢弃。示例：")
    print(bad_rows.head())
    df_use = df_use[~y.isna()].copy()
    y = df_use[TARGET_COL].map(y_map)

X = df_use[RAW_FEATURES].copy()
for c in RAW_FEATURES:
    X[c] = pd.to_numeric(X[c], errors="coerce")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)


pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=2000, random_state=RANDOM_STATE))
])

param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
    "clf__penalty": ["l2"],
}

grid = GridSearchCV(
    pipe, param_grid=param_grid,
    scoring="roc_auc", cv=5, n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\n=== Best Params (by ROC-AUC, CV) ===")
print(grid.best_params_)

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Test Metrics ===")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1-score   : {f1:.4f}")
print(f"ROC-AUC    : {auc:.4f}")
print("\nConfusion Matrix [[TN, FP], [FN, TP]]:\n", cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"LogReg (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("logreg_roc.png", dpi=150)
plt.close()
print("ROC 曲线已保存为: logreg_roc.png")

roc_df = pd.DataFrame({
    "threshold": thresholds,
    "fpr": fpr,
    "tpr": tpr,
})
roc_df["specificity"] = 1 - roc_df["fpr"]
roc_df["youden_J"] = roc_df["tpr"] - roc_df["fpr"]
roc_df["auc"] = auc  # 方便快速查看

best_idx = roc_df["youden_J"].idxmax()
best_row = roc_df.loc[best_idx]
print(f"Best (Youden J) threshold on test set: {best_row['threshold']:.6f}, "
      f"TPR={best_row['tpr']:.4f}, FPR={best_row['fpr']:.4f}")

roc_df.to_csv("logreg_roc_points.csv", index=False, encoding="utf-8-sig")
print("ROC 数据表（测试集）已保存为: logreg_roc_points.csv")

imputer = best_model.named_steps["imputer"]
scaler  = best_model.named_steps["scaler"]
clf     = best_model.named_steps["clf"]

beta_std = clf.coef_[0].copy()
b_std    = clf.intercept_[0]
mu       = scaler.mean_
sigma    = scaler.scale_

beta_raw = beta_std / sigma
b_raw    = b_std - np.sum(beta_std * (mu / sigma))
or_std   = np.exp(beta_std)
or_raw   = np.exp(beta_raw)

params_df = pd.DataFrame({
    "feature": RAW_FEATURES,
    "beta_std(1 SD)": beta_std,
    "OR_std(1 SD)": or_std,
    "beta_raw(per unit)": beta_raw,
    "OR_raw(per unit)": or_raw,
}).sort_values(by="beta_raw(per unit)", key=lambda s: s.abs(), ascending=False)

print("\n=== Logistic Regression Parameters ===")
print(f"Intercept_std (b_std): {b_std:+.6f}")
print(f"Intercept_raw (b_raw): {b_raw:+.6f}")
print(params_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

params_df.to_csv("logreg_params.csv", index=False, encoding="utf-8-sig")
print("参数表已保存为: logreg_params.csv")

latex_lines = []
latex_lines.append(r"\[")
latex_lines.append(r"\operatorname{logit}\,P(y=1\mid x) = " + f"{b_raw:+.6f} ")
for i, (f, br) in enumerate(zip(RAW_FEATURES, beta_raw)):
    sign = "+" if br >= 0 else "-"
    val  = abs(br)
    term = f" {sign} {val:.6f}\\,\\cdot\\,{latex_escape(f)}"
    latex_lines.append(term + (r" \\" if (i+1) % 3 == 0 else ""))
latex_lines.append(r"\]")
latex_text = "\n".join(latex_lines)
with open("logreg_formula_latex.txt", "w", encoding="utf-8") as f:
    f.write(latex_text)
print("LaTeX 公式已保存为: logreg_formula_latex.txt")

all_probs = best_model.predict_proba(X)[:, 1]
all_pred  = (all_probs >= 0.5).astype(int)

out = df_use.copy()
out["pred_prob"]  = all_probs
out["pred_label"] = all_pred
out.to_csv("predictions_logreg.csv", index=False, encoding="utf-8-sig")
print("完整样本预测已保存为: predictions_logreg.csv")

print("\n=== Evaluating on FLOATED dataset ===")
df_f = pd.read_excel(FLOATED_FILE_PATH)
df_f = strip_columns(df_f)

if TARGET_COL not in df_f.columns:
    raise KeyError(f"浮动后数据集未找到目标列：{TARGET_COL}")
if "检测孕周" in df_f.columns:
    df_f["检测孕周"] = df_f["检测孕周"].apply(to_weeks)

missing_f = [c for c in RAW_FEATURES if c not in df_f.columns]
if missing_f:
    raise KeyError(f"浮动后数据集缺失特征列：{missing_f}")

df_f_use = df_f[[TARGET_COL] + RAW_FEATURES].copy()

y_f = df_f_use[TARGET_COL].map({"是":1, "否":0, 1:1, 0:0})
if y_f.isna().any():
    bad_rows_f = df_f_use[y_f.isna()]
    print("警告：浮动集中存在无法映射为二元标签的行，这些行将被丢弃。示例：")
    print(bad_rows_f.head())
    df_f_use = df_f_use[~y_f.isna()].copy()
    y_f = df_f_use[TARGET_COL].map({"是":1, "否":0, 1:1, 0:0})

X_f = df_f_use[RAW_FEATURES].copy()
for c in RAW_FEATURES:
    X_f[c] = pd.to_numeric(X_f[c], errors="coerce")

y_f_prob = best_model.predict_proba(X_f)[:, 1]
y_f_pred = (y_f_prob >= 0.5).astype(int)

acc_f  = accuracy_score(y_f, y_f_pred)
prec_f = precision_score(y_f, y_f_pred, zero_division=0)
rec_f  = recall_score(y_f, y_f_pred, zero_division=0)
f1_f   = f1_score(y_f, y_f_pred, zero_division=0)
try:
    auc_f = roc_auc_score(y_f, y_f_prob)
except ValueError:
    auc_f = float("nan")
cm_f = confusion_matrix(y_f, y_f_pred)

print("\n--- Metrics on FLOATED dataset ---")
print(f"Accuracy   : {acc_f:.4f}")
print(f"Precision  : {prec_f:.4f}")
print(f"Recall     : {rec_f:.4f}")
print(f"F1-score   : {f1_f:.4f}")
print(f"ROC-AUC    : {auc_f:.4f}")
print("\nConfusion Matrix [[TN, FP], [FN, TP]]:\n", cm_f)
print("\nClassification Report (floated):")
print(classification_report(y_f, y_f_pred, digits=4))

out_f = df_f_use.copy()
out_f["pred_prob"]  = y_f_prob
out_f["pred_label"] = y_f_pred
out_f.to_csv("predictions_logreg_floated.csv", index=False, encoding="utf-8-sig")
print("浮动数据集预测已保存为: predictions_logreg_floated.csv")
