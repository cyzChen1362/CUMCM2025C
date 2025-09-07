import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE

# ============ 可按需修改的参数 ============
INPUT_FILE   = r"附件_女胎检测数据_处理后.xlsx"
TARGET_COL   = "染色体是否正常0-1变量"
STATUS_COL   = "染色体是否正常"

# 如果你只想用某些特征
PREDICTOR_COLS = []

RANDOM_STATE  = 42
OUTPUT_FILE   = "balanced_组合采样.xlsx"
COMBO_METHOD  = "smoteenn"
V_VERBOSE     = True


def to_weeks(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 7.0, 2)
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except:
        return np.nan


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


def robust_numeric_cleanup(X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if verbose:
        print("\n[诊断] 各列非NaN计数：")
        print(X.notna().sum().sort_values())

    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print("\n[处理] 发现全NaN列，已删除：", all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    med = X.median(numeric_only=True)
    X = X.apply(lambda s: s.fillna(med.get(s.name, np.nan)))

    still_all_nan = [c for c in X.columns if X[c].isna().all()]
    if still_all_nan:
        print("\n[处理] 仍全NaN列，用0兜底：", still_all_nan)
        X[still_all_nan] = X[still_all_nan].fillna(0.0)

    return X


def find_sheet_with_target(xls: pd.ExcelFile, target_col: str) -> str:
    for sheet in xls.sheet_names:
        head = pd.read_excel(xls, sheet_name=sheet, nrows=5)
        if target_col in head.columns:
            return sheet
    t = target_col.strip()
    for sheet in xls.sheet_names:
        head = pd.read_excel(xls, sheet_name=sheet, nrows=5)
        cols_strip = [str(c).strip() for c in head.columns]
        if t in cols_strip:
            return sheet
    return None


def main():
    # 1) 读取并定位工作表
    xls = pd.ExcelFile(INPUT_FILE)
    sheet = find_sheet_with_target(xls, TARGET_COL)
    if sheet is None:
        raise RuntimeError(f"未在工作簿中找到目标列“{TARGET_COL}”，请检查列名或工作表。")
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"使用工作表：{sheet}")

    # 2) 目标列转 0/1
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
        raise ValueError(f"目标列存在无法转为0/1的值，示例行索引：{bad_rows}（最多显示10行）。请先清洗后再运行。")

    # 3) 特征选择与数值化
    if PREDICTOR_COLS:
        missing = [c for c in PREDICTOR_COLS if c not in df.columns]
        if missing:
            print("警告：以下指定特征列不存在，将忽略：", missing)
        X = df[[c for c in PREDICTOR_COLS if c in df.columns]].copy()
    else:
        X = df.drop(columns=[TARGET_COL]).copy()

    if "检测孕周" in X.columns:
        X["检测孕周"] = X["检测孕周"].apply(to_weeks)

    percent_like_cols = {
        "在参考基因组上比对的比例", "重复读段的比例", "GC含量",
        "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
        "被过滤掉读段数的比例"
    }
    for c in X.columns:
        if c in percent_like_cols or X[c].dtype == "object":
            X[c] = X[c].apply(to_numeric_maybe_percent)

    # 全列数值化
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 稳健清洗
    X = robust_numeric_cleanup(X, verbose=V_VERBOSE)

    # 4) 组合采样前类别分布
    cnt_before = Counter(y)
    print("\n组合采样之前类别分布：", dict(cnt_before))

    # 5) 根据少数类自动设置 SMOTE 的 k_neighbors
    minority_class = min(cnt_before, key=cnt_before.get)
    minority_n = cnt_before[minority_class]
    k_neighbors = max(1, min(5, minority_n - 1))  # k_neighbors 必须 < 少数类数量
    if k_neighbors < 1:
        raise RuntimeError("少数类样本太少，无法进行 SMOTE（需要至少 2 个少数类样本）。")
    print(f"SMOTE 使用 k_neighbors={k_neighbors}（少数类样本数={minority_n}）")

    # 6) 构造组合采样器
    smote_base = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    if COMBO_METHOD.lower() == "smotetomek":
        sampler = SMOTETomek(random_state=RANDOM_STATE, smote=smote_base)
        method_name = "SMOTETomek"
    else:
        sampler = SMOTEENN(random_state=RANDOM_STATE, smote=smote_base)
        method_name = "SMOTEENN"

    print(f"组合采样方法：{method_name}")

    # 7) 运行组合采样
    X_res, y_res = sampler.fit_resample(X, y)

    # 8) 组合采样后类别分布
    cnt_after = Counter(y_res)
    print("组合采样之后类别分布：", dict(cnt_after))

    # 9) 组装并导出：加入目标列与“染色体是否正常”同步列
    df_res = pd.concat([X_res.reset_index(drop=True),
                        pd.Series(y_res, name=TARGET_COL)], axis=1)

    # 同步文本列
    df_res[STATUS_COL] = df_res[TARGET_COL].map({1: "是", 0: "否"}).astype(object)

    out_path = Path(OUTPUT_FILE).resolve()
    df_res.to_excel(out_path, index=False)
    print(f"\n已导出均衡数据：{out_path}")


if __name__ == "__main__":
    main()
