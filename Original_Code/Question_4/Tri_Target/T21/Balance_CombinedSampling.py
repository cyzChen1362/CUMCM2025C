import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import Optional

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE

# ============ 参数 ============
INPUT_FILE   = r"附件_女胎检测数据_处理后.xlsx"
TARGET_COL   = "21号染色体是否正常0-1变量"
STATUS_COL   = "21号染色体是否正常"  # 文本列：1->“是”，0->“否”
PREDICTOR_COLS = []

RANDOM_STATE  = 42
OUTPUT_FILE   = "balanced_组合采样.xlsx"
COMBO_METHOD  = "smoteenn"
# ========================================


def find_sheet_with_target(xls: pd.ExcelFile, target_col: str) -> Optional[str]:
    t = str(target_col).strip()
    for sheet in xls.sheet_names:
        head = pd.read_excel(xls, sheet_name=sheet, nrows=5)
        cols_strip = [str(c).strip() for c in head.columns]
        if t in cols_strip:
            return sheet
    return None


def to01(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return int(round(float(v)))
    s = str(v).strip()
    if s in {"1", "是", "true", "True", "TRUE"}:
        return 1
    if s in {"0", "否", "false", "False", "FALSE"}:
        return 0
    try:
        f = float(s)
        return int(round(f))
    except:
        return np.nan


def main():
    # 1) 读取表和定位工作表
    xls = pd.ExcelFile(INPUT_FILE)
    sheet = find_sheet_with_target(xls, TARGET_COL)
    if sheet is None:
        raise RuntimeError(f"未在工作簿中找到目标列“{TARGET_COL}”。")
    df = pd.read_excel(xls, sheet_name=sheet)
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"当前工作表不含目标列：{TARGET_COL}")

    # 2) 目标列规范为 0/1
    y = df[TARGET_COL].apply(to01)
    if y.isna().any():
        bad_idx = y[y.isna()].index.tolist()[:10]
        raise ValueError(f"目标列存在无法转为0/1的值，示例行索引：{bad_idx}")

    df[TARGET_COL] = y.astype(int)

    # 3) 构造特征矩阵 X：
    if PREDICTOR_COLS:
        missing = [c for c in PREDICTOR_COLS if c not in df.columns]
        if missing:
            print("警告：以下指定特征列不存在，将忽略：", missing)
        X = df[[c for c in PREDICTOR_COLS if c in df.columns]].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    else:
        X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL], errors="ignore").copy()

    if X.shape[1] == 0:
        raise RuntimeError("没有可用的数值型特征列。请在 PREDICTOR_COLS 中显式指定。")

    # 4) 丢弃含 NaN 的样本
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]
    if len(y_clean) < 2:
        raise RuntimeError("有效样本太少，无法进行采样。")

    # 5) 采样前类别分布
    cnt_before = Counter(y_clean)
    print("\n组合采样之前类别分布：", dict(cnt_before))

    # 6) 配置 SMOTE 基础器
    minority_class = min(cnt_before, key=cnt_before.get)
    minority_n = cnt_before[minority_class]
    k_neighbors = max(1, min(5, minority_n - 1))  # k_neighbors 必须 < 少数类数量
    if k_neighbors < 1:
        raise RuntimeError("少数类样本太少，无法进行 SMOTE")
    print(f"SMOTE 使用 k_neighbors={k_neighbors}（少数类样本数={minority_n}）")

    smote_base = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    if COMBO_METHOD.lower() == "smotetomek":
        sampler = SMOTETomek(random_state=RANDOM_STATE, smote=smote_base)
        method_name = "SMOTETomek"
    else:
        sampler = SMOTEENN(random_state=RANDOM_STATE, smote=smote_base)
        method_name = "SMOTEENN"
    print(f"组合采样方法：{method_name}")

    # 7) 运行组合采样
    X_res, y_res = sampler.fit_resample(X_clean, y_clean)

    # 8) 采样后类别分布
    cnt_after = Counter(y_res)
    print("组合采样之后类别分布：", dict(cnt_after))

    # 9) 组装输出：特征 + 目标 + 同步文本列
    X_res_df = pd.DataFrame(X_res, columns=X_clean.columns)
    df_res = pd.concat([X_res_df.reset_index(drop=True),
                        pd.Series(y_res, name=TARGET_COL)], axis=1)
    df_res[STATUS_COL] = df_res[TARGET_COL].map({1: "是", 0: "否"}).astype(object)

    out_path = Path(OUTPUT_FILE).resolve()
    df_res.to_excel(out_path, index=False)
    print(f"\n已导出均衡数据：{out_path}")


if __name__ == "__main__":
    main()
