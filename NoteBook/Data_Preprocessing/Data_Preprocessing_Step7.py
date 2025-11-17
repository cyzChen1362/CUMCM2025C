import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

import matplotlib.pyplot as plt
from typing import Union

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False

INPUT_FILE   = "../Data/附件_女胎检测数据_处理后.xlsx"
sheet = "女胎检测数据"
TARGET_COL   = "染色体是否正常0-1变量"
STATUS_COL   = "染色体是否正常"

PREDICTOR_COLS = []

RANDOM_STATE  = 42
OUTPUT_FILE   = "../Data/balanced_组合采样.xlsx"
OUTPUT_FILE_EXAMPLE = "../Data/balanced_组合采样_example.xlsx"

# 将一些奇奇怪怪的数字格式统统转成SMOTE/Tomek可以使用的
def to_numeric_maybe_percent(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

# 去掉NAN INF 全空列这种
def robust_numeric_cleanup(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    med = X.median(numeric_only=True)
    X = X.apply(lambda s: s.fillna(med.get(s.name, np.nan)))

    still_all_nan = [c for c in X.columns if X[c].isna().all()]
    if still_all_nan:
        X[still_all_nan] = X[still_all_nan].fillna(0.0)

    return X

# “组合采样 + 导出”封装成函数
def combo_sample_and_export(
        X: pd.DataFrame,
        y: pd.Series,
        output_file: str,
        status_col: str,
        verbose_tag: str = "",
        plot_info: Union[dict, None] = None
):
    """
    1）SMOTE 只对少数类过采样到多数类一样多；
    2）TomekLinks 只在少数类上做清理（删掉一部分少数类），不删多数类。
    """

    tag = f"[{verbose_tag}]" if verbose_tag else ""

    # 组合采样前类别分布
    cnt_before = Counter(y)
    print(f"\n{tag} 组合采样之前类别分布：", dict(cnt_before))

    # 识别多数类 / 少数类
    minority_class = min(cnt_before, key=cnt_before.get)
    majority_class = max(cnt_before, key=cnt_before.get)
    minority_n = cnt_before[minority_class]
    majority_n = cnt_before[majority_class]

    # 根据少数类自动设置 SMOTE 的 k_neighbors
    k_neighbors = max(1, min(5, minority_n - 1))
    if k_neighbors < 1:
        raise RuntimeError(f"{tag} 少数类样本太少，无法进行 SMOTE（需要至少 2 个少数类样本）。")
    print(f"{tag} SMOTE 使用 k_neighbors={k_neighbors}（少数类样本数={minority_n}）")

    # SMOTE 过采样
    smote_strategy = {minority_class: majority_n}
    smote = SMOTE(
        sampling_strategy=smote_strategy,
        k_neighbors=k_neighbors,
        random_state=RANDOM_STATE
    )

    X_sm, y_sm = smote.fit_resample(X, y)
    cnt_after_smote = Counter(y_sm)
    print(f"{tag} SMOTE 之后类别分布：", dict(cnt_after_smote))

    # Tomek Links 欠采样，只删掉少数类
    tomek = TomekLinks(sampling_strategy=[minority_class])
    X_res, y_res = tomek.fit_resample(X_sm, y_sm)

    # 组合采样后类别分布
    cnt_after = Counter(y_res)
    print(f"{tag} TomekLinks 之后类别分布：", dict(cnt_after))

    # 加入目标列与“染色体是否正常”同步列
    df_res = pd.concat([X_res.reset_index(drop=True),
                        pd.Series(y_res, name=TARGET_COL)], axis=1)

    df_res[status_col] = df_res[TARGET_COL].map({1: "是", 0: "否"}).astype(object)

    out_path = Path(output_file).resolve()
    df_res.to_excel(out_path, index=False)
    print(f"\n{tag} 已导出均衡数据：{out_path}")

    # 画图
    if plot_info is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(plot_info.get("title", "女胎数据建模准备流程分析"), fontsize=14)

        # 图 1：采样前类别饼图
        normal_before = cnt_before.get(1, 0)
        abnormal_before = cnt_before.get(0, 0)
        axes[0].pie(
            [normal_before, abnormal_before],
            labels=["正常", "异常"],
            autopct="%.1f%%",
            startangle=90
        )
        axes[0].set_title(f"原始类别分布（{normal_before + abnormal_before} 样本）")

        # 图 2：采样后类别饼图
        normal_after = cnt_after.get(1, 0)
        abnormal_after = cnt_after.get(0, 0)
        axes[1].pie(
            [normal_after, abnormal_after],
            labels=["正常", "异常"],
            autopct="%.1f%%",
            startangle=90
        )
        axes[1].set_title(f"平衡后类别分布（{normal_after + abnormal_after} 样本）")

        plt.tight_layout(rect=[0, 0, 1, 0.9])

        fig.savefig("./Images/picture14.png",
                        dpi=300, bbox_inches="tight")

        plt.show()

def main():
    xls = pd.ExcelFile(INPUT_FILE)
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"使用工作表：{sheet}")

    if TARGET_COL not in df.columns:
        raise RuntimeError(f"当前工作表不含目标列：{TARGET_COL}")

    def to01(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).strip()
        if s in ["1", "是", "true", "True", "TRUE"]:
            return 1
        if s in ["0", "否", "false", "False", "FALSE"]:
            return 0
        try:
            f = float(s)
            return int(round(f))
        except:
            return np.nan

    y = df[TARGET_COL].map(to01)
    if y.isna().any():
        bad_rows = y[y.isna()].index.tolist()[:10]
        raise ValueError(
            f"目标列存在无法转为0/1的值，示例行索引：{bad_rows}（最多显示10行）。请先清洗后再运行。"
        )

    if PREDICTOR_COLS:
        missing = [c for c in PREDICTOR_COLS if c not in df.columns]
        if missing:
            print("警告：以下指定特征列不存在，将忽略：", missing)
        X = df[[c for c in PREDICTOR_COLS if c in df.columns]].copy()
    else:
        X = df.drop(columns=[TARGET_COL]).copy()

    percent_like_cols = {
        "在参考基因组上比对的比例", "重复读段的比例", "GC含量",
        "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
        "被过滤掉读段数的比例"
    }
    for c in X.columns:
        if c in percent_like_cols or X[c].dtype == "object":
            X[c] = X[c].apply(to_numeric_maybe_percent)

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_numeric_before_cleanup = X.copy()
    X_clean = robust_numeric_cleanup(X)
    n_total = len(X_clean)

    # 原始数据直接组合采样（不画图）
    combo_sample_and_export(
        X_clean,
        y,
        OUTPUT_FILE,
        STATUS_COL,
        verbose_tag="原始数据"
    )

    # GC 过滤后再采样 + 画两个饼图
    if "GC含量" not in X_numeric_before_cleanup.columns:
        print("\n[警告] X 中不存在列 'GC含量'，无法生成 balanced_组合采样_example.xlsx")
        return

    gc_series = X_numeric_before_cleanup["GC含量"]
    mask_gc_ok = gc_series.between(0.4, 0.6, inclusive="both")

    print("\n[GC过滤] 总样本数：", len(gc_series))
    print("[GC过滤] 保留 GC∈[0.4,0.6] 的样本数：", mask_gc_ok.sum())

    X_gc = X_numeric_before_cleanup[mask_gc_ok].copy()
    y_gc = y[mask_gc_ok].copy()
    n_gc = len(y_gc)

    if len(y_gc.unique()) < 2:
        raise RuntimeError("GC过滤后只剩下单一类别，无法进行组合采样。请检查数据或放宽过滤条件。")

    X_gc_clean = robust_numeric_cleanup(X_gc)

    plot_info = {
        "title": "女胎数据建模准备流程分析",
        "n_orig_total": n_total,
        "n_after_gc": n_gc,
        "n_after_selection": n_gc
    }

    combo_sample_and_export(
        X_gc_clean,
        y_gc,
        OUTPUT_FILE_EXAMPLE,
        STATUS_COL,
        verbose_tag="GC∈[0.4,0.6] 过滤后",
        plot_info=plot_info
    )

if __name__ == "__main__":
    main()
