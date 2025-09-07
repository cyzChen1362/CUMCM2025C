"""
基于真实表格：
- π(w,x): 逻辑回归 (含 w*BMI 交互、插补、标准化)
- q(w):   方案2 融合：q_fused(w) = η * q_bin(w) + (1-η) * q_kernel(w)
          其中 q_bin 为孕周分箱经验成功率 (Jeffreys 平滑) + 线性插值
               q_kernel 为一维高斯核平滑回归 + 线性插值
          η 在验证集上自动调参（默认以 AUC 最大为准）

标签口径（优先）:
    π 的 y_pi: ['NIPT准确性0-1变量','NIPT准确性']；否则 y_frac>=0.04
    q 的 y_q : ['检测准确性0-1变量','检测准确性']；否则 (0.4<=GC<=0.6)
注意：q 表示“检测准确/成功 概率”，不是失败概率

风险函数:
    R(w)=α(1-π)+β(1-q)+(1-α-β)*(w-wmin)/(wmax-wmin)

"""

import os
import re
import math
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Sequence

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from joblib import dump, load
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ 1) 工具: 孕周解析/列名映射 ------------------

def to_weeks(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 7.0, 2)
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m:
        return float(m.group(1))
    m = re.match(r"^\s*\d+(\.\d+)?\s*$", s)
    if m:
        return float(s)
    return np.nan

COLMAP = {
    "检测孕周": "w", "孕周": "w",
    "孕妇BMI": "BMI", "BMI": "BMI",
    "Y染色体浓度": "y_frac", "Y 染色体浓度": "y_frac",
    "原始读段数": "reads_total",
    "在参考基因组上比对的比例": "align_rate",
    "重复读段的比例": "dup_rate",
    "唯一比对的读段数": "uniq_reads",
    "被过滤掉读段数的比例": "filtered_rate",
    "GC含量": "gc_global",
    "13号染色体的GC含量": "gc_chr13",
    "18号染色体的GC含量": "gc_chr18",
    "21号染色体的GC含量": "gc_chr21",
    "生产次数": "parity",
    "检测抽血次数": "draw_times",
    "年龄": "age",
    "身高": "height",
    "体重": "weight",
}

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        c_strip = str(c).strip()
        new_cols[c] = COLMAP.get(c_strip, c_strip)
    return df.rename(columns=new_cols)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 孕周
    if 'w' not in df.columns:
        cand = [c for c in df.columns if '孕周' in str(c)]
        if cand:
            df = df.rename(columns={cand[0]: 'w'})
    df['w'] = df['w'].apply(to_weeks)


# ------------------ 2) 读入与清洗 ------------------

EXCEL_PATH = r"..\Data_Processing\男胎检测数据_处理后_with_flags.xlsx"
assert os.path.exists(EXCEL_PATH), f"找不到文件: {EXCEL_PATH}"

raw = pd.read_excel(EXCEL_PATH)
df = rename_columns(raw.copy())
df = ensure_columns(df)

need = ['w','BMI','y_frac']
missing = [c for c in need if c not in df.columns]
if missing:
    raise RuntimeError(f"缺少关键列: {missing}")

df = df.dropna(subset=['w','BMI','y_frac']).reset_index(drop=True)

# ------------------ 3) 生成标签（优先用现成列） ------------------

# π 的标签
if 'NIPT准确性0-1变量' in df.columns:
    y_pi = df['NIPT准确性0-1变量'].astype(int).values
elif 'NIPT准确性' in df.columns:
    y_pi_tmp = pd.to_numeric(df['NIPT准确性'], errors='coerce')
    if y_pi_tmp.isna().any():
        y_pi_tmp = df['NIPT准确性'].astype(str).str.strip().map({'是':1,'否':0,'1':1,'0':0})
    y_pi = y_pi_tmp.fillna(0).astype(int).values
else:
    y_pi = (df['y_frac'] >= 0.04).astype(int).values

# q 的标签（成功=1）
if '检测准确性0-1变量' in df.columns:
    y_q_all = df['检测准确性0-1变量'].astype(int).values
elif '检测准确性' in df.columns:
    yq_tmp = pd.to_numeric(df['检测准确性'], errors='coerce')
    if yq_tmp.isna().any():
        yq_tmp = df['检测准确性'].astype(str).str.strip().map({'是':1,'否':0,'1':1,'0':0})
    y_q_all = yq_tmp.fillna(0).astype(int).values
else:
    if 'gc_global' not in df.columns:
        raise RuntimeError("没有 q 的标签列，也没有 GC含量 列以兜底生成。")
    y_q_all = ((df['gc_global'] >= 0.40) & (df['gc_global'] <= 0.60)).astype(int).values

# ------------------ 4) 特征 ------------------

base_features = ['w','BMI']
optional_feats = [
    'age','height','weight','parity','draw_times'
]
extra_features = [f for f in optional_feats if f in df.columns]
X_cols = base_features + extra_features

# ------------------ 5) 训练/验证划分 ------------------

X = df[X_cols].copy()
X_train, X_valid, y_pi_tr, y_pi_va, y_q_tr, y_q_va = train_test_split(
    X, y_pi, y_q_all, test_size=0.25, random_state=42, stratify=y_pi
)
# 说明：q 只用于评估/调参，不训练模型

# ------------------ 6) π(w,x): 带插补的逻辑回归（含 w*BMI 交互） ------------------

def build_pi_pipeline(X_cols):
    inter_cols = ['w','BMI']
    remain_cols = [c for c in X_cols if c not in inter_cols]

    transformers = []
    if inter_cols:
        transformers.append((
            'inter',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                # 二次多项式：包含 w, BMI, w^2, BMI^2, w*BMI
                ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
                ('scaler', StandardScaler())
            ]),
            inter_cols
        ))
    if remain_cols:
        transformers.append((
            'rest',
            Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]),
            remain_cols
        ))
    pre = ColumnTransformer(transformers, remainder='drop')

    logit = LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        class_weight='balanced'
    )
    return Pipeline([('pre', pre), ('clf', logit)])


pipe_pi = build_pi_pipeline(X_cols)
pipe_pi.fit(X_train, y_pi_tr)

def eval_binary_probs(proba, y_true, name):
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(np.nan_to_num(proba, nan=np.nanmean(proba)), 0.0, 1.0)
    auc = roc_auc_score(y_true, proba)
    bs  = brier_score_loss(y_true, proba)
    print(f"{name}: AUC={auc:.3f}, Brier={bs:.4f}")
    return auc, bs

print("=== 验证集性能 ===")
pi_valid_prob = pipe_pi.predict_proba(X_valid)[:,1]
eval_binary_probs(pi_valid_prob, y_pi_va, "pi(w,x) 达标概率")

# ------------------ 7) q(w,BMI)：内部组件（仅供融合用） ------------------

from typing import Optional

def _build_binned_q_2d(df_all: pd.DataFrame,
                       w_col: str = 'w',
                       bmi_col: str = 'BMI',
                       y_cols: Tuple[str, str] = ('检测准确性0-1变量','检测准确性'),
                       gc_col: str = 'gc_global',
                       w_bin_width: float = 0.20,
                       bmi_bin_width: float = 1.5):
    # 标签（成功=1）
    if y_cols[0] in df_all.columns:
        yq = pd.to_numeric(df_all[y_cols[0]], errors='coerce')
    elif y_cols[1] in df_all.columns:
        tmp = pd.to_numeric(df_all[y_cols[1]], errors='coerce')
        if tmp.isna().any():
            tmp = df_all[y_cols[1]].astype(str).str.strip().map({'是':1,'否':0,'1':1,'0':0})
        yq = pd.to_numeric(tmp, errors='coerce')
    else:
        if gc_col not in df_all.columns:
            raise RuntimeError("没有 q 的标签列，也没有 GC含量列以兜底生成。")
        yq = ((df_all[gc_col] >= 0.40) & (df_all[gc_global] <= 0.60)).astype(int)

    w   = pd.to_numeric(df_all[w_col], errors='coerce')
    bmi = pd.to_numeric(df_all[bmi_col], errors='coerce')
    ok  = w.notna() & bmi.notna()
    w, bmi, yq = w[ok].astype(float), bmi[ok].astype(float), yq[ok].fillna(0).clip(0,1)

    wmin, wmax = float(w.min()), float(w.max())
    bmin, bmax = float(bmi.min()), float(bmi.max())
    wbins = np.arange(wmin - 1e-9, wmax + w_bin_width, w_bin_width)
    bbins = np.arange(bmin - 1e-9, bmax + bmi_bin_width, bmi_bin_width)

    # 二维分箱
    wbin = pd.cut(w, bins=wbins, right=False)
    bbin = pd.cut(bmi, bins=bbins, right=False)
    df_g = pd.DataFrame({"wbin": wbin, "bbin": bbin, "yq": yq.values})
    grp  = df_g.groupby(["wbin","bbin"], observed=True)["yq"]

    k = grp.sum(); n = grp.count()
    q_emp = (k + 0.5) / (n + 1.0)  # Jeffreys 平滑

    w_centers = np.array([iv.left + w_bin_width/2.0 for iv in pd.Categorical(df_g["wbin"]).categories])
    b_centers = np.array([iv.left + bmi_bin_width/2.0 for iv in pd.Categorical(df_g["bbin"]).categories])
    W, B = len(w_centers), len(b_centers)
    grid = np.full((W, B), np.nan, dtype=float)

    wcats = {c:i for i,c in enumerate(pd.Categorical(df_g["wbin"]).categories)}
    bcats = {c:j for j,c in enumerate(pd.Categorical(df_g["bbin"]).categories)}
    for (wcat, bcat), val in q_emp.items():
        i = wcats.get(wcat, None); j = bcats.get(bcat, None)
        if i is not None and j is not None:
            grid[i, j] = float(val)

    # 行/列均值 -> 全局均值 -> 0.5 兜底
    if np.isnan(grid).any():
        row_means = np.nanmean(grid, axis=1)
        for i in range(W):
            sel = np.isnan(grid[i, :])
            if sel.any(): grid[i, sel] = row_means[i]
        col_means = np.nanmean(grid, axis=0)
        for j in range(B):
            sel = np.isnan(grid[:, j])
            if sel.any(): grid[sel, j] = col_means[j]
        grid = np.nan_to_num(grid, nan=0.5)

    return (w_centers, b_centers, grid)

w_centers, b_centers, _q_grid_bin2d = _build_binned_q_2d(df, w_bin_width=0.20, bmi_bin_width=1.5)

def _bilinear_interp(x, y, xs, ys, Z):
    i = np.searchsorted(xs, x) - 1
    j = np.searchsorted(ys, y) - 1
    i = np.clip(i, 0, len(xs)-2)
    j = np.clip(j, 0, len(ys)-2)
    x1, x2 = xs[i], xs[i+1]; y1, y2 = ys[j], ys[j+1]
    z11, z21 = Z[i, j],   Z[i+1, j]
    z12, z22 = Z[i, j+1], Z[i+1, j+1]
    tx = 0.0 if x2==x1 else (x - x1)/(x2 - x1)
    ty = 0.0 if y2==y1 else (y - y1)/(y2 - y1)
    z1 = z11*(1-tx) + z21*tx
    z2 = z12*(1-tx) + z22*tx
    return float(np.clip(z1*(1-ty) + z2*ty, 0.0, 1.0))

def _q_bin2d_hat(w: float, bmi: float) -> float:
    if (w is None) or (bmi is None) or not np.isfinite(w) or not np.isfinite(bmi):
        return 0.5
    wq = float(np.clip(w,   w_centers.min(), w_centers.max()))
    bq = float(np.clip(bmi, b_centers.min(), b_centers.max()))
    return _bilinear_interp(wq, bq, w_centers, b_centers, _q_grid_bin2d)


# ------------------ 8) q(w,BMI)：二维核 + 带宽CV（仅供融合用） ------------------

def _fit_kernel_q_2d(df_all: pd.DataFrame,
                     y_cols=('检测准确性0-1变量','检测准确性'),
                     gc_col='gc_global',
                     bandwidth_grid=(0.15, 0.20, 0.25, 0.30, 0.40),
                     n_splits=5,
                     select_metric='brier', lambda_combo=0.7):
    # 标签
    if y_cols[0] in df_all.columns:
        yq = pd.to_numeric(df_all[y_cols[0]], errors='coerce')
    elif y_cols[1] in df_all.columns:
        tmp = pd.to_numeric(df_all[y_cols[1]], errors='coerce')
        if tmp.isna().any():
            tmp = df_all[y_cols[1]].astype(str).str.strip().map({'是':1,'否':0,'1':1,'0':0})
        yq = pd.to_numeric(tmp, errors='coerce')
    else:
        if gc_col not in df_all.columns:
            raise RuntimeError("没有 q 的标签列，也没有 GC含量列以兜底生成。")
        yq = ((df_all[gc_col] >= 0.40) & (df_all[gc_col] <= 0.60)).astype(int)

    w   = pd.to_numeric(df_all['w'],   errors='coerce').astype(float)
    bmi = pd.to_numeric(df_all['BMI'], errors='coerce').astype(float)
    ok  = np.isfinite(w) & np.isfinite(bmi)
    w, bmi, y = w[ok].values, bmi[ok].values, yq[ok].fillna(0).clip(0,1).astype(float).values
    if len(w) == 0:
        wmin, wmax = 12.0, 16.0
        bmin, bmax = 18.0, 32.0
        W, B = 60, 40
        return (np.linspace(wmin, wmax, W), np.linspace(bmin, bmax, B), np.full((W,B), 0.5), bandwidth_grid[0])

    def k(u): return (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*u*u)

    # CV 选带宽
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best = None
    for h in bandwidth_grid:
        preds = np.zeros_like(y)
        for tr, va in kf.split(w):
            wt, bt, yt = w[tr], bmi[tr], y[tr]
            wv, bv     = w[va], bmi[va]
            for ii in range(len(va)):
                zw = (wt - wv[ii]) / h
                zb = (bt - bv[ii]) / h
                wts = k(zw) * k(zb)
                s   = wts.sum()
                pv  = (np.dot(wts, yt) / s) if s > 1e-12 else yt.mean()
                preds[va[ii]] = float(np.clip(pv, 0.0, 1.0))
        preds = np.clip(np.nan_to_num(preds, nan=float(y.mean())), 0.0, 1.0)
        brier = brier_score_loss(y, preds)
        try:
            auc = roc_auc_score(y, preds)
        except ValueError:
            auc = 0.5
        score = (auc if select_metric=='auc'
                 else (-brier if select_metric=='brier'
                       else auc - lambda_combo*brier))
        if (best is None) or (score > best['score']):
            best = {'h': h, 'auc': auc, 'brier': brier, 'score': score}

    h = best['h']

    # 预计算规则网格（供快速查询/可视化）
    wmin, wmax = float(w.min()), float(w.max())
    bmin, bmax = float(bmi.min()), float(bmi.max())
    w_grid = np.linspace(wmin, wmax, 120)
    b_grid = np.linspace(bmin, bmax, 120)
    Q = np.zeros((len(w_grid), len(b_grid)), dtype=float)
    kw_cache = np.zeros((len(w_grid), len(w)), dtype=float)
    for i, w0 in enumerate(w_grid):
        kw_cache[i, :] = k((w - w0) / h)
    for j, b0 in enumerate(b_grid):
        kb = k((bmi - b0) / h)
        for i in range(len(w_grid)):
            wts = kw_cache[i, :] * kb
            s   = wts.sum()
            pv  = (np.dot(wts, y) / s) if s > 1e-12 else y.mean()
            Q[i, j] = float(np.clip(pv, 0.0, 1.0))

    return w_grid, b_grid, Q, h

_w_grid2d, _b_grid2d, _q_grid_kernel2d, _best_h2d = _fit_kernel_q_2d(df, select_metric='brier')

def _q_kernel2d_hat(w: float, bmi: float) -> float:
    if (w is None) or (bmi is None) or not np.isfinite(w) or not np.isfinite(bmi):
        return 0.5
    wq = float(np.clip(w,   _w_grid2d.min(), _w_grid2d.max()))
    bq = float(np.clip(bmi, _b_grid2d.min(), _b_grid2d.max()))
    return _bilinear_interp(wq, bq, _w_grid2d, _b_grid2d, _q_grid_kernel2d)

# ------------------ 9) q_fused2d(w,BMI)：η 在验证集上自动调参（仅输出融合） ------------------

def _tune_eta_2d(w_valid: Sequence[float], bmi_valid: Sequence[float], y_valid: Sequence[int],
                 eta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
                 metric='auc', lambda_combo=0.7):
    best = None
    for eta in eta_grid:
        qb  = np.array([_q_bin2d_hat(w, b)    for w, b in zip(w_valid, bmi_valid)], dtype=float)
        qk  = np.array([_q_kernel2d_hat(w, b) for w, b in zip(w_valid, bmi_valid)], dtype=float)
        qfu = np.clip(eta*qb + (1-eta)*qk, 0.0, 1.0)
        try: auc = roc_auc_score(y_valid, qfu)
        except ValueError: auc = 0.5
        brier = brier_score_loss(y_valid, qfu)
        score = (auc if metric=='auc' else (-brier if metric=='brier' else auc - lambda_combo*brier))
        if (best is None) or (score > best['score']):
            best = {'eta': eta, 'auc': auc, 'brier': brier, 'score': score}
    return best

_eta2d_res = _tune_eta_2d(X_valid['w'].values, X_valid['BMI'].values, y_q_va,
                          eta_grid=(0.0, 0.25, 0.5, 0.75, 1.0), metric='auc')
ETA_2D = _eta2d_res['eta']
print(f"q_fused2d: 选择 η={ETA_2D:.2f} (VA AUC={_eta2d_res['auc']:.3f}, Brier={_eta2d_res['brier']:.4f})")

def q_hat(w: float, bmi: float=None, feat: Dict[str, Any]=None) -> float:
    """外部唯一可用的 q：二维融合 q_fused2d(w,BMI)。BMI 缺失时返回 0.5。"""
    if (bmi is None) or (not np.isfinite(bmi)):
        return 0.5
    qb = _q_bin2d_hat(w, bmi)
    qk = _q_kernel2d_hat(w, bmi)
    return float(np.clip(ETA_2D * qb + (1 - ETA_2D) * qk, 0.0, 1.0))

# ------------------ 10) 导出可调用函数（π 的接口保持不变） ------------------

def _make_feature_row(w: float, bmi: float, feat: Dict[str, Any]) -> pd.DataFrame:
    row = {col: np.nan for col in X_cols}
    row['w'] = float(w) if (w is not None and not math.isnan(w)) else np.nan
    row['BMI'] = float(bmi) if (bmi is not None and not math.isnan(bmi)) else np.nan
    if feat:
        for k, v in feat.items():
            if k in row: row[k] = v
    return pd.DataFrame([row])

def pi_hat(w: float, bmi: float, feat: Dict[str, Any] = None) -> float:
    Xrow = _make_feature_row(w, bmi, feat or {})
    return float(pipe_pi.predict_proba(Xrow)[0, 1])

# ------------------ 11) 风险函数（仅使用 q_fused2d） ------------------

def recommend_week(bmi: float, feat: Dict[str, Any] = None,
                   alpha: float = 0.7, beta: float = 0.1,
                   w_min: float = None, w_max: float = None,
                   step: float = 0.1,
                   feasible: Tuple[float, float] = None):
    """
    最小化: R(w)=α(1-π)+β(1-q_fused2d)+(1-α-β)*((w-wmin)/(wmax-wmin))
    """
    feat = feat or {}
    wmin = float(df['w'].min()) if w_min is None else w_min
    wmax = float(df['w'].max()) if w_max is None else w_max
    grid = np.arange(wmin, wmax + 1e-9, step)

    pis = np.array([pi_hat(w, bmi, feat) for w in grid])
    qs  = np.array([q_hat(w, bmi) for w in grid])
    norm_t = (grid - wmin) / (wmax - wmin + 1e-12)
    R = alpha*(1 - pis) + beta*(1 - qs) + (1 - alpha - beta)*norm_t

    if feasible is not None:
        mask = (grid >= feasible[0]) & (grid <= feasible[1])
        if mask.any(): R[~mask] = R[mask].max() + 10.0

    i = int(np.argmin(R))
    return {
        "w_star": float(grid[i]),
        "pi": float(pis[i]),
        "q": float(qs[i]),
        "Rmin": float(R[i]),
        "curve": pd.DataFrame({"w": grid, "pi": pis, "q": qs, "R": R}),
        "bmi": float(bmi)
    }

# ------------------ 12) 示例与可视化（只画融合曲线） ------------------

example = recommend_week(bmi=32.0, feat={"gc_global": 0.5, "align_rate": 0.9}, alpha=0.7, beta=0.1)
print(f"推荐孕周 w*={example['w_star']:.2f}, pi={example['pi']:.3f}, q={example['q']:.3f}, R={example['Rmin']:.3f}")

try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.set_constrained_layout(True)   # 避免标题/标签被裁切（比 tight_layout 更稳）
    # 或者：最后 plt.tight_layout() 也可以

    wmin, wmax = float(df['w'].min()), float(df['w'].max())
    w_plot = np.linspace(wmin, wmax, 400)
    bmi0 = example["bmi"]

    q_fus_slice = [q_hat(w, bmi0) for w in w_plot]  # fused (η=ETA_2D)

    # 左图：q_fused2d vs w
    axes[0].plot(w_plot, q_fus_slice, lw=2.4, label=f"q_fused2d (η={ETA_2D:.2f})")
    axes[0].set_xlabel("Gestational week (w)")
    axes[0].set_ylabel(f"q(w, BMI={bmi0:.1f}) accuracy probability")
    axes[0].set_title("q_fused2d slice over week")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    # 右图：风险曲线 + 最优周
    axes[1].plot(example["curve"]["w"], example["curve"]["R"], lw=2)
    axes[1].axvline(example["w_star"], ls='--', lw=1)
    axes[1].set_xlabel("Gestational week (w)")
    axes[1].set_ylabel("Risk R(w)")
    axes[1].set_title("Personalized risk and optimal week")
    axes[1].grid(True, alpha=0.3)

    # 如果你要存图，推荐用 bbox_inches='tight' 防裁切
    # plt.savefig("figure.png", dpi=200, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"画图跳过: {e}")


# ------------------ 13) 持久化（只导出融合曲线） ------------------

dump(pipe_pi, "pi_model.joblib")
w_out = np.linspace(float(df['w'].min()), float(df['w'].max()), 400)
bmi_out = example["bmi"]
out = pd.DataFrame({
    "w": w_out,
    "BMI": bmi_out,
    "q_fused2d":  [q_hat(w, bmi_out) for w in w_out],
})
out.to_csv("q_fused2d_by_week_bmi_slice.csv", index=False, encoding="utf-8-sig")
with open("q_fused2d_meta.txt", "w", encoding="utf-8") as f:
    f.write(f"eta_2d={ETA_2D:.4f}\n")
    f.write(f"bmi_slice={bmi_out:.3f}\n")
print("已保存: pi_model.joblib, q_fused2d_by_week_bmi_slice.csv, q_fused2d_meta.txt")

def load_models(pi_path="pi_model.joblib", q_csv="q_fused2d_by_week_bmi_slice.csv"):
    pi = load(pi_path)
    qtab = pd.read_csv(q_csv)
    return pi, qtab

# ------------------ 14) π(w,x) 参数（标准化空间） ------------------

# 提取预处理后特征名（顺序与 coef_ 对齐）
pre = pipe_pi.named_steps['pre']
feat_names = pre.get_feature_names_out(input_features=X_cols)

# 取逻辑回归的权重和截距
clf = pipe_pi.named_steps['clf']
coef = clf.coef_.ravel()
intercept = clf.intercept_[0]

# 打印
print("=== π(w,x) 参数（标准化空间）===")
for name, w in zip(feat_names, coef):
    print(f"{name:25s}  {w:+.6f}")
print(f"{'intercept':25s}  {intercept:+.6f}")

# 交互分支的插补中位数 & 标准化均值方差
inter_imputer = pre.named_transformers_['inter'].named_steps['imputer']
inter_scaler  = pre.named_transformers_['inter'].named_steps['scaler']
print("inter imputer medians:", inter_imputer.statistics_)
print("inter scaler means:", inter_scaler.mean_)
print("inter scaler vars:",  inter_scaler.var_)

# 其它特征分支
rest_imputer = pre.named_transformers_['rest'].named_steps['imputer']
rest_scaler  = pre.named_transformers_['rest'].named_steps['scaler']
print("rest imputer medians:", rest_imputer.statistics_)
print("rest scaler means:",   rest_scaler.mean_)
print("rest scaler vars:",    rest_scaler.var_)

# ------------------ 15) q_fused2d 参数 ------------------

print("=== q_fused2d 参数 ===")
print(f"ETA_2D (融合权重): {ETA_2D:.4f}")
print(f"best bandwidth (kernel h): {_best_h2d:.4f}")

print(f"binned grid: w centers={len(w_centers)} [{w_centers.min():.2f}, {w_centers.max():.2f}], "
      f"BMI centers={len(b_centers)} [{b_centers.min():.2f}, {b_centers.max():.2f}]")

print(f"kernel grid: w points={len(_w_grid2d)} [{_w_grid2d.min():.2f}, {_w_grid2d.max():.2f}], "
      f"BMI points={len(_b_grid2d)} [{_b_grid2d.min():.2f}, {_b_grid2d.max():.2f}]")

# 分箱 Jeffreys 平滑表导出
bin_table = pd.DataFrame(_q_grid_bin2d, index=w_centers, columns=b_centers)
bin_table.to_csv("q_bin2d_grid.csv", encoding="utf-8-sig")

# 核平滑表导出
ker_table = pd.DataFrame(_q_grid_kernel2d, index=_w_grid2d, columns=_b_grid2d)
ker_table.to_csv("q_kernel2d_grid.csv", encoding="utf-8-sig")
