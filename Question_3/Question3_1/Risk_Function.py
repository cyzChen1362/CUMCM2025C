import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Optional, Tuple

from Calclating_Pi_Q import (
    pipe_pi,
    X_cols,
    ETA_2D,
    _w_grid2d, _b_grid2d, _q_grid_kernel2d,
    w_centers, b_centers, _q_grid_bin2d,
    df as df_models
)

W_MIN = float(pd.to_numeric(df_models['w'], errors='coerce').min())
W_MAX = float(pd.to_numeric(df_models['w'], errors='coerce').max())
step = 0.10
alpha = 0.70
beta  = 0.10
min_n = 1
batch_rows = 150_000

def to_weeks(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)): return float(val)
    s = str(val).strip()
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m: return round(int(m.group(1)) + int(m.group(2))/7.0, 2)
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m: return float(m.group(1))
    m = re.match(r"^\s*\d+(\.\d+)?\s*$", s)
    if m: return float(s)
    return np.nan

def _observed_week_range_for_group(
    df_group: pd.DataFrame,
    bmi_series: np.ndarray
) -> Optional[Tuple[float, float]]:

    for cand in ["检测孕周", "孕周", "w", "W"]:
        if cand in df_group.columns:
            ww = pd.to_numeric(df_group[cand].apply(to_weeks), errors='coerce')
            ww = ww[np.isfinite(ww)]
            if ww.size > 0:
                return float(ww.min()), float(ww.max())

    if bmi_series.size == 0:
        return None
    bmin, bmax = float(np.nanmin(bmi_series)), float(np.nanmax(bmi_series))
    eps = 1e-6
    dfg = df_models.copy()
    if 'BMI' not in dfg.columns:
        return None
    dfg['w_num'] = pd.to_numeric(dfg['w'], errors='coerce')
    dfg['BMI_num'] = pd.to_numeric(dfg['BMI'], errors='coerce')
    sub = dfg[(dfg['BMI_num'] >= bmin - eps) & (dfg['BMI_num'] <= bmax + eps)]
    ww = sub['w_num'].dropna().to_numpy(dtype=float)
    if ww.size == 0:
        return None
    return float(np.min(ww)), float(np.max(ww))

def _bilinear_vec(Wv, Bv, xs, ys, Z):

    Wv = np.clip(Wv, xs[0], xs[-1])
    Bv = np.clip(Bv, ys[0], ys[-1])

    i = np.searchsorted(xs, Wv, side='right') - 1
    j = np.searchsorted(ys, Bv, side='right') - 1
    i = np.clip(i, 0, len(xs)-2)
    j = np.clip(j, 0, len(ys)-2)

    x1 = xs[i];   x2 = xs[i+1]
    y1 = ys[j];   y2 = ys[j+1]
    tx = np.divide(Wv - x1, (x2 - x1), out=np.zeros_like(Wv), where=(x2!=x1))
    ty = np.divide(Bv - y1, (y2 - y1), out=np.zeros_like(Bv), where=(y2!=y1))

    z11 = Z[i,   j]
    z21 = Z[i+1, j]
    z12 = Z[i,   j+1]
    z22 = Z[i+1, j+1]

    z1 = z11*(1-tx) + z21*tx
    z2 = z12*(1-tx) + z22*ty
    out = z1*(1-ty) + z2*ty
    return np.clip(out, 0.0, 1.0)

def q_fused2d_vec(W_mesh, BMI_mesh):
    # 两套网格各做一次双线性，再加权融合；保持输出与输入网格同形状
    qb = _bilinear_vec(W_mesh.ravel(), BMI_mesh.ravel(), w_centers, b_centers, _q_grid_bin2d)
    qk = _bilinear_vec(W_mesh.ravel(), BMI_mesh.ravel(), _w_grid2d, _b_grid2d, _q_grid_kernel2d)
    qf = ETA_2D*qb + (1.0-ETA_2D)*qk
    return qf.reshape(W_mesh.shape)

def _make_X_batch(w_vec, bmi_vec):

    n = len(w_vec)
    data = {col: np.full(n, np.nan, dtype=float) for col in X_cols}
    if 'w' in data:   data['w']   = w_vec.astype(float)
    if 'BMI' in data: data['BMI'] = bmi_vec.astype(float)
    return pd.DataFrame(data, columns=X_cols)

def _predict_pi_mesh_fast(weeks, bmi_list):

    W, B = len(weeks), len(bmi_list)
    w_flat = np.repeat(weeks, B)
    b_flat = np.tile(bmi_list, W)

    pis = np.empty_like(w_flat, dtype=float)
    for start in range(0, len(w_flat), batch_rows):
        end = min(start + batch_rows, len(w_flat))
        Xb = _make_X_batch(w_flat[start:end], b_flat[start:end])
        proba = pipe_pi.predict_proba(Xb)[:, 1]
        pis[start:end] = np.clip(proba, 0.0, 1.0)
    return pis.reshape(W, B)

def process_group_file_fast(file_path: str, label: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    if "孕妇BMI" in df.columns:
        bmi = pd.to_numeric(df["孕妇BMI"], errors="coerce").to_numpy(dtype=float)
    elif "BMI" in df.columns:
        bmi = pd.to_numeric(df["BMI"], errors="coerce").to_numpy(dtype=float)
    elif {"身高","体重"}.issubset(df.columns):
        h = pd.to_numeric(df["身高"], errors="coerce").to_numpy(dtype=float)
        h_m = np.where(h > 10, h/100.0, h)
        w_kg = pd.to_numeric(df["体重"], errors="coerce").to_numpy(dtype=float)
        bmi = w_kg / (h_m**2)
    else:
        raise RuntimeError(f"{file_path} lacks BMI and cannot be derived.")

    bmi = bmi[np.isfinite(bmi)]
    n_group = len(bmi)
    if n_group == 0:
        raise RuntimeError(f"{file_path} has no valid BMI values.")

    weeks = np.arange(W_MIN, W_MAX + 1e-12, step, dtype=float)

    Wm, Bm = np.meshgrid(weeks, bmi, indexing='ij')

    q_mat = q_fused2d_vec(Wm, Bm)                   # (W, B)

    pi_mat = _predict_pi_mesh_fast(weeks, bmi)      # (W, B)

    norm_t = (weeks - W_MIN) / max(1e-12, (W_MAX - W_MIN))
    R_rows = alpha*(1.0 - pi_mat).mean(axis=1) + beta*(1.0 - q_mat).mean(axis=1) + (1.0 - alpha - beta)*norm_t

    obs = _observed_week_range_for_group(df, bmi)
    if obs is not None:
        wmin_g, wmax_g = obs
        mask = (weeks >= wmin_g) & (weeks <= wmax_g)
        if np.any(mask):
            R_rows = np.where(mask, R_rows, np.inf)

    res = pd.DataFrame({
        "week": weeks,
        "n": n_group,
        "pi_mean": pi_mat.mean(axis=1),
        "q_mean":  q_mat.mean(axis=1),
        "R_model": R_rows,
        "group":  label
    })

    idx = int(np.nanargmin(R_rows))
    best = res.iloc[idx]
    print(f"\n===== {label} =====")
    print(f"Optimal week (aligned): {best['week']:.2f}  | "
          f"n={int(best['n'])}, pi_mean={best['pi_mean']:.3f}, "
          f"q_mean={best['q_mean']:.3f}, R={best['R_model']:.4f}")

    return res

files = {
    "Group1": "BMI_group1.xlsx",
    "Group2": "BMI_group2.xlsx",
    "Group3": "BMI_group3.xlsx",
    "Group4": "BMI_group4.xlsx",
    "Group5": "BMI_group5.xlsx",
}

results = []
for label, path in files.items():
    results.append(process_group_file_fast(path, label))

all_res = pd.concat(results, ignore_index=True)

plt.figure(figsize=(10, 6))
for label in files.keys():
    sub = all_res[all_res["group"] == label]
    plt.plot(sub["week"], sub["R_model"], label=f"{label} - Aligned")
plt.xlabel("Gestational age (weeks)")
plt.ylabel("Expected risk R(w)")
plt.title("Time–Accuracy Tradeoff (Aligned to dynamic-grouping settings)")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

all_res.to_csv("risk_by_week_all_groups_aligned.csv", index=False, encoding="utf-8-sig")
print("Saved: risk_by_week_all_groups_aligned.csv")
