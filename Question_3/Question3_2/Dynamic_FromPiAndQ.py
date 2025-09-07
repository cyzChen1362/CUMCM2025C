from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False  # ensure minus sign renders

MIN_GROUP_N = 5

# === 1) 引入来自“第一段”的模型接口 ===
try:
    from Calclating_Pi_Q import pi_hat, q_hat, df as df_models  # 若你保存的模块名不同，请改这里
except Exception:
    import builtins
    pi_hat = builtins.__dict__.get('pi_hat', None)
    q_hat  = builtins.__dict__.get('q_hat', None)
    df_models = builtins.__dict__.get('df', None)

if (pi_hat is None) or (q_hat is None) or (df_models is None) or ('w' not in df_models.columns):
    raise RuntimeError("未能找到 pi_hat/q_hat/df_models")

# 全局 w 范围（用于 norm_t 的归一化）
W_MIN = float(df_models['w'].min())
W_MAX = float(df_models['w'].max())

# === 2) 读取/准备数据 ===
if 'BMI' not in df_models.columns:
    raise RuntimeError("df_models 缺少 'BMI' 列")

DATA = df_models[['BMI', 'w']].dropna().rename(columns={'w': 'gestational_weeks'}).copy()
BMI_arr = DATA['BMI'].to_numpy(float)
W_arr   = DATA['gestational_weeks'].to_numpy(float)

# ====== 2.5) 预计算加速：按 BMI 排序 + 生成 w 网格 + 计算 PI/Q 矩阵 + 前缀和 ======
order = np.argsort(BMI_arr)
BMI_sorted = BMI_arr[order]
W_sorted   = W_arr[order]  # 若需限制到观测范围会用到它

def make_w_grid(w_min=W_MIN, w_max=W_MAX, step=0.1):
    return np.arange(float(w_min), float(w_max) + 1e-9, float(step), dtype=float)

# 注意：W_GRID 的步长应与主流程期望一致
W_GRID = make_w_grid(step=0.1)
NW = W_GRID.size
NS = BMI_sorted.size

# 2) 构建 PI/Q 矩阵（只算一次）
PI = np.empty((NW, NS), dtype=float)
Q  = np.empty((NW, NS), dtype=float)

for wi, w in enumerate(W_GRID):
    pi_row = [pi_hat(w, float(b)) for b in BMI_sorted]
    q_row  = [q_hat (w, float(b)) for b in BMI_sorted]
    pi_row = np.array(pi_row, dtype=float)
    q_row  = np.array(q_row,  dtype=float)

    if np.isnan(pi_row).all(): pi_row[:] = 0.5
    if np.isnan(q_row ).all(): q_row[:]  = 0.5
    pi_row = np.clip(np.nan_to_num(pi_row, nan=np.nanmean(pi_row)), 0.0, 1.0)
    q_row  = np.clip(np.nan_to_num(q_row,  nan=np.nanmean(q_row )), 0.0, 1.0)

    PI[wi] = pi_row
    Q [wi] = q_row

# 3) 列方向前缀和（便于 O(1) 取任意 BMI 连续区间的和）
PI_cumsum = np.cumsum(PI, axis=1)  # (NW, NS)
Q_cumsum  = np.cumsum(Q,  axis=1)

# 4) 预计算 norm_t(w)
NORM_T = np.clip((W_GRID - W_MIN) / (W_MAX - W_MIN + 1e-12), 0.0, 1.0)

# === 3) 加速后的组最优孕周搜索工具 ===
def _interval_sum(cumsum_row: np.ndarray, l: int, r: int) -> np.ndarray:
    if l == 0:
        return cumsum_row[:, r]
    return cumsum_row[:, r] - cumsum_row[:, l-1]

def _group_mean_over_w(l: int, r: int):
    n = (r - l + 1)
    if n <= 0:
        return None, None, 0
    pi_sum = _interval_sum(PI_cumsum, l, r)
    q_sum  = _interval_sum(Q_cumsum,  l, r)
    return pi_sum / n, q_sum / n, n

def group_best_week_by_range(bmi_l: float, bmi_r: float,
                             alpha: float, beta: float,
                             restrict_to_observed: bool = True) -> dict:

    l = int(np.searchsorted(BMI_sorted, bmi_l, side='left'))
    r = int(np.searchsorted(BMI_sorted, bmi_r, side='right')) - 1
    if l > r:
        return {"w_star": np.nan, "R_min": np.inf, "n": 0}

    mean_pi_w, mean_q_w, n = _group_mean_over_w(l, r)
    if n == 0:
        return {"w_star": np.nan, "R_min": np.inf, "n": 0}

    R_w = alpha*(1.0 - mean_pi_w) + beta*(1.0 - mean_q_w) + (1.0 - alpha - beta)*NORM_T

    if restrict_to_observed:
        w_g = W_sorted[l:r+1]
        wmin_g, wmax_g = float(np.min(w_g)), float(np.max(w_g))
        mask = (W_GRID >= wmin_g) & (W_GRID <= wmax_g)
        if not np.any(mask):
            return {"w_star": np.nan, "R_min": np.inf, "n": n}
        R_w = np.where(mask, R_w, np.inf)

    wi_star = int(np.argmin(R_w))
    return {"w_star": float(W_GRID[wi_star]), "R_min": float(R_w[wi_star]), "n": int(n)}

# === 4) 评估一个分组方案（向量化版 + 最小样本数约束） ===
def evaluate_grouping(boundaries: np.ndarray,
                      alpha: float,
                      beta: float) -> dict:

    K = len(boundaries) - 1
    groups = []

    for g in range(K):
        lo, hi = float(boundaries[g]), float(boundaries[g+1])
        # 与 searchsorted 对齐处理右端点闭区间
        if g == K - 1:
            hi_adj = np.nextafter(hi, float('inf'))
        else:
            hi_adj = np.nextafter(hi, -float('inf'))

        # 先计算该区间样本数，若 < MIN_GROUP_N，直接判无效
        l = int(np.searchsorted(BMI_sorted, lo, side='left'))
        r = int(np.searchsorted(BMI_sorted, hi_adj, side='right')) - 1
        n_here = (r - l + 1) if (l <= r) else 0
        if n_here < MIN_GROUP_N:
            return {"ok": False}

        best = group_best_week_by_range(lo, hi_adj, alpha, beta, restrict_to_observed=True)
        if (not np.isfinite(best["R_min"])) or (best["n"] < MIN_GROUP_N):
            return {"ok": False}

        groups.append({
            "bmi_range": (lo, hi),
            "n": best["n"],
            "w_star": best["w_star"],
            "R_min": best["R_min"],
        })

    weights = np.array([g["n"] for g in groups], dtype=float)
    weights = weights / weights.sum()
    overall = float(np.sum(weights * np.array([g["R_min"] for g in groups], dtype=float)))

    return {
        "ok": True,
        "overall_R": overall,
        "groups": groups,
        "boundaries": boundaries.astype(float)
    }

# === 5) 蒙特卡洛搜索分组边界（保持接口） ===
def sample_boundaries(n_groups: int,
                      bmi_min: float,
                      bmi_max: float,
                      min_gap: float) -> np.ndarray | None:

    if n_groups < 2:
        raise ValueError("n_groups 至少为 2。")
    inner = n_groups - 1
    for _ in range(200):
        pts = np.sort(np.random.uniform(bmi_min, bmi_max, size=inner))
        bnds = np.concatenate([[bmi_min], pts, [bmi_max]])
        if np.all(np.diff(bnds) >= min_gap - 1e-9):
            return bnds
    return None

def monte_carlo_search(n_groups: int = 5,
                       trials: int = 15000,
                       alpha: float = 0.7,
                       beta: float  = 0.05,
                       min_gap_ratio: float = 0.5):

    bmi_min, bmi_max = float(BMI_sorted.min()), float(BMI_sorted.max())
    span = bmi_max - bmi_min
    min_gap = (span / n_groups) * float(min_gap_ratio)

    best = None
    for t in range(1, trials + 1):
        bnds = sample_boundaries(n_groups, bmi_min, bmi_max, min_gap)
        if bnds is None:
            continue
        res = evaluate_grouping(bnds, alpha=alpha, beta=beta)
        if not res["ok"]:
            continue
        if (best is None) or (res["overall_R"] < best["overall_R"]):
            best = res
        if t % 100 == 0:
            print(f"[{t}/{trials}] current best overall R = {best['overall_R']:.6f}" if best else f"[{t}/{trials}] searching...")

    if best is None:
        raise RuntimeError("未找到有效分组。")
    return best

# === 6) 主流程 ===
if __name__ == "__main__":
    N_GROUPS  = 5
    TRIALS    = 15000
    ALPHA     = 0.7
    BETA      = 0.1
    STEP_W    = 0.1
    GAP_RATIO = 0.3

    best = monte_carlo_search(
        n_groups=N_GROUPS,
        trials=TRIALS,
        alpha=ALPHA,
        beta=BETA,
        min_gap_ratio=GAP_RATIO
    )

    print("\n=== Best result ===")
    print(f"Overall weighted risk: {best['overall_R']:.6f}")
    print("BMI boundaries:", " | ".join([f"{b:.2f}" for b in best["boundaries"]]))

    print("\nGrp | BMI range          |  n  | best week (w*) | min R_in_group")
    print("-" * 70)
    for i, g in enumerate(best["groups"], 1):
        lo, hi = g["bmi_range"]
        print(f"{i:>3d} | [{lo:5.1f},{hi:5.1f}] | {g['n']:4d} | {g['w_star']:13.2f} | {g['R_min']:13.6f}")

    try:
        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, g in enumerate(best["groups"], 1):
            lo, hi = g["bmi_range"]
            # 索引区间
            hi_adj = np.nextafter(hi, float('inf')) if i == len(best["groups"]) else np.nextafter(hi, -float('inf'))
            l = int(np.searchsorted(BMI_sorted, lo, side='left'))
            r = int(np.searchsorted(BMI_sorted, hi_adj, side='right')) - 1
            if l > r:
                continue
            mean_pi_w, mean_q_w, n = _group_mean_over_w(l, r)
            if n < MIN_GROUP_N:
                continue  # 保守：图上也跳过不满足约束的组
            R_w = ALPHA*(1.0 - mean_pi_w) + BETA*(1.0 - mean_q_w) + (1.0 - ALPHA - BETA)*NORM_T
            ax.plot(W_GRID, R_w, lw=2,
                    label=f"Group {i}: BMI[{lo:.1f},{hi:.1f}] (n={n})",
                    color=colors[(i-1) % len(colors)])
            wi_star = int(np.argmin(R_w))
            ax.scatter([W_GRID[wi_star]], [R_w[wi_star]], s=60,
                       color=colors[(i-1) % len(colors)], edgecolors='k', zorder=3)

        ax.set_xlabel("Gestational week (w)")
        ax.set_ylabel("Risk R(w)")
        ax.set_title(f"Risk vs Week by BMI groups  (alpha={ALPHA}, beta={BETA})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(Visualization skipped) {e}")
