import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ========= 参数 =========
INPUT_FILE = r"附件_女胎检测数据_处理后.xlsx"
TARGET_COL = "21号染色体是否正常0-1变量"
PREDICTOR_COLS = [
    "检测抽血次数", "检测孕周", "孕妇BMI", "原始读段数", "在参考基因组上比对的比例",
    "重复读段的比例", "唯一比对的读段数", "GC含量", "13号染色体的Z值", "18号染色体的Z值",
    "21号染色体的Z值", "X染色体的Z值", "X染色体浓度", "13号染色体的GC含量",
    "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例"
]
VIF_THRESHOLD = 15.0
OUTPUT_FILE = "VIF_21号染色体_选择结果.xlsx"
# ========================

def to_weeks(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m: return round(int(m.group(1)) + int(m.group(2))/7.0, 2)
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m: return float(m.group(1))
    try: return float(s)
    except: return np.nan

def to_numeric_maybe_percent(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        try: return float(s[:-1]) / 100.0
        except: return np.nan
    try: return float(s)
    except: return np.nan

def find_sheet_with_columns(xls: pd.ExcelFile, required_cols: list[str]) -> Optional[str]:
    best_sheet, best_score = None, -1
    for sheet in xls.sheet_names:
        head = pd.read_excel(xls, sheet_name=sheet, nrows=5)
        score = len(set(head.columns) & set(required_cols))
        if score > best_score:
            best_sheet, best_score = sheet, score
    return best_sheet


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """返回按 VIF 从高到低排序的表"""
    arr = X.values
    vif_vals = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
    return pd.DataFrame({"feature": X.columns, "VIF": vif_vals}).sort_values("VIF", ascending=False)

def stepwise_vif_selection(X: pd.DataFrame, threshold: float = 10.0, max_steps: int = 100):
    """
    逐步剔除：
      - 计算VIF；若最高VIF>阈值且特征数>1，则删除该特征，循环
      - 返回：最终X、最终VIF、每轮VIF历史、剔除顺序
    """
    kept = list(X.columns)
    history = []
    dropped = []
    for _ in range(max_steps):
        vif_df = compute_vif_table(X[kept])
        history.append(vif_df.copy())
        top = vif_df.iloc[0]
        if top["VIF"] > threshold and len(kept) > 1:
            kept.remove(top["feature"])
            dropped.append(top["feature"])
        else:
            break
    final_vif = compute_vif_table(X[kept])
    return X[kept].copy(), final_vif, history, dropped

def main():
    # 1) 读取并定位表
    xls = pd.ExcelFile(INPUT_FILE)
    sheet = find_sheet_with_columns(xls, [TARGET_COL] + PREDICTOR_COLS)
    if sheet is None:
        raise RuntimeError("未找到包含所需列的工作表，请检查列名。")
    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"使用工作表：{sheet}")

    # 2) 取自变量；若缺列，提示但继续
    missing = [c for c in PREDICTOR_COLS if c not in df.columns]
    if missing:
        print("警告：以下列不存在，将被忽略：", missing)
    cols_in = [c for c in PREDICTOR_COLS if c in df.columns]
    if len(cols_in) < 2:
        raise RuntimeError("可用自变量少于2个，无法计算VIF。")

    X = df[cols_in].copy()

    # 3) 轻度清洗与数值化
    if "检测孕周" in X.columns:
        X["检测孕周"] = X["检测孕周"].apply(to_weeks)

    percent_like = {
        "在参考基因组上比对的比例", "重复读段的比例", "GC含量",
        "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
        "被过滤掉读段数的比例"
    }
    for c in X.columns:
        if c in percent_like or X[c].dtype == "object":
            X[c] = X[c].apply(to_numeric_maybe_percent)
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 4) 替换 inf、删除全NaN列、缺失填补（列内中位数）
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print("提示：以下列全为缺失，已删除：", all_nan_cols)
        X.drop(columns=all_nan_cols, inplace=True)
    med = X.median(numeric_only=True)
    X = X.apply(lambda s: s.fillna(med.get(s.name, np.nan)))
    still_nan = [c for c in X.columns if X[c].isna().all()]
    if still_nan:
        print("提示：以下列仍全NaN，使用0兜底：", still_nan)
        X[still_nan] = 0.0

    # 5) 删除方差为0的常数列（避免VIF异常）
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print("提示：以下常数列已删除：", const_cols)
        X.drop(columns=const_cols, inplace=True)

    # 6) 逐步VIF筛选
    kept_X, final_vif, history, dropped = stepwise_vif_selection(X, threshold=VIF_THRESHOLD, max_steps=200)

    # 7) 打印摘要
    print("\n=== 逐步VIF筛选日志（显示每轮最高VIF） ===")
    for i, vif_df in enumerate(history, 1):
        top = vif_df.iloc[0]
        print(f"第{i:02d}轮：最高VIF={top['VIF']:.3f}（{top['feature']}），特征数={len(vif_df)}")
    print("\n被剔除特征顺序：", " -> ".join(dropped) if dropped else "无（全部<=阈值）")
    print("\n=== 最终保留特征及其VIF ===")
    print(final_vif.to_string(index=False))

    # 8) 导出
    out_path = Path(OUTPUT_FILE).resolve()
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for i, vif_df in enumerate(history, 1):
            vif_df.to_excel(writer, sheet_name=f"round_{i:02d}_vif", index=False)
        final_vif.to_excel(writer, sheet_name="final_vif", index=False)
        pd.DataFrame({"dropped_order": dropped}).to_excel(writer, sheet_name="dropped_order", index=False)
        pd.DataFrame({"kept_features": kept_X.columns}).to_excel(writer, sheet_name="kept_features", index=False)
    print(f"\n结果已导出：{out_path}")

if __name__ == "__main__":
    main()
