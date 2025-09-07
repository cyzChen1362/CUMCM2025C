import re
import numpy as np
import pandas as pd
from pathlib import Path

from statsmodels.stats.outliers_influence import variance_inflation_factor

# ========= 参数 =========
INPUT_FILE = r"附件_女胎检测数据_处理后.xlsx"
TARGET_COL = "染色体是否正常0-1变量"
PREDICTOR_COLS = [
    "检测抽血次数", "检测孕周", "孕妇BMI", "原始读段数", "在参考基因组上比对的比例",
    "重复读段的比例", "唯一比对的读段数", "GC含量", "13号染色体的Z值", "18号染色体的Z值",
    "21号染色体的Z值", "X染色体的Z值", "X染色体浓度", "13号染色体的GC含量",
    "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例"
]
VIF_THRESHOLD = 10.0   # 常用阈值：5 或 10
OUTPUT_FILE = "VIF_selection_results.xlsx"
# ====================================


def to_weeks(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # 13w+5 / 13W+5
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 7.0, 2)
    # 13w / 13W
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m:
        return float(m.group(1))
    # 纯数字字符串
    m = re.match(r"^\s*(\d+(\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1))
    return np.nan


def to_numeric_maybe_percent(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    # 百分号
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return np.nan
    # 普通数字
    try:
        return float(s)
    except:
        return np.nan


def find_sheet_with_columns(xls: pd.ExcelFile, required_cols: list[str]) -> str:
    best_sheet = None
    best_score = -1
    for sheet in xls.sheet_names:
        df_tmp = pd.read_excel(xls, sheet_name=sheet, nrows=5)  # 只读前几行加速
        cols = set(df_tmp.columns)
        score = len(cols.intersection(set(required_cols)))
        if score > best_score:
            best_sheet, best_score = sheet, score
    return best_sheet


def compute_vif_table(x: pd.DataFrame) -> pd.DataFrame:
    """
    计算当前自变量矩阵的 VIF 表。
    """
    arr = x.values
    vif_list = []
    for i in range(arr.shape[1]):
        vif_val = variance_inflation_factor(arr, i)
        vif_list.append(vif_val)
    return pd.DataFrame({"feature": x.columns, "VIF": vif_list}).sort_values("VIF", ascending=False)


def stepwise_vif_selection(x: pd.DataFrame, threshold: float = 10.0, max_steps: int = 100):
    """
    逐步剔除法：
      - 计算各特征 VIF
      - 若最高 VIF > 阈值，则剔除该特征，继续
      - 直到全部 <= 阈值 或 达到 max_steps
    """
    kept_cols = list(x.columns)
    history = []
    dropped_order = []

    for step in range(max_steps):
        vif_df = compute_vif_table(x[kept_cols])
        history.append(vif_df.copy())

        max_row = vif_df.iloc[0]
        if max_row["VIF"] > threshold and len(kept_cols) > 1:
            drop_col = max_row["feature"]
            dropped_order.append(drop_col)
            kept_cols.remove(drop_col)
            # 下一轮继续
        else:
            break

    final_vif = compute_vif_table(x[kept_cols])
    return x[kept_cols].copy(), final_vif, history, dropped_order


def main():
    # 0) 读取工作簿并定位工作表
    xls = pd.ExcelFile(INPUT_FILE)
    required_any = [TARGET_COL] + PREDICTOR_COLS
    sheet = find_sheet_with_columns(xls, required_any)
    if sheet is None:
        raise RuntimeError("未能在工作簿中找到包含所需列的工作表，请检查列名。")

    df = pd.read_excel(xls, sheet_name=sheet)
    print(f"使用工作表：{sheet}")

    # 1) 检查并提取列
    missing = [c for c in [TARGET_COL] + PREDICTOR_COLS if c not in df.columns]
    if missing:
        print("警告：下列列在此工作表中未找到，将忽略：", missing)

    cols_in = [c for c in PREDICTOR_COLS if c in df.columns]
    if len(cols_in) < 2:
        raise RuntimeError("可用的自变量少于 2 个，无法计算 VIF。请检查列名。")

    # 2) 复制出自变量矩阵 X，并做清洗/数值化
    X = df[cols_in].copy()

    # 2.1) 特殊处理 “检测孕周”
    if "检测孕周" in X.columns:
        X["检测孕周"] = X["检测孕周"].apply(to_weeks)

    # 2.2) 将可疑的百分号字符串/一般数值字符串 -> 数值
    percent_like_cols = [
        "在参考基因组上比对的比例", "重复读段的比例", "GC含量",
        "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
        "被过滤掉读段数的比例"
    ]
    for c in X.columns:
        if c in percent_like_cols or X[c].dtype == "object":
            X[c] = X[c].apply(to_numeric_maybe_percent)

    # 2.3) 全部转为数值
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 3) 缺失值填补
    X = X.apply(lambda s: s.fillna(s.median()), axis=0)

    # 3.1) 若仍存在全为 NaN 的列，剔除该列
    all_na_cols = [c for c in X.columns if X[c].isna().all()]
    if all_na_cols:
        print("警告：以下列全部为缺失，已剔除：", all_na_cols)
        X = X.drop(columns=all_na_cols)

    # 4) 逐步 VIF 筛选
    kept_X, final_vif, history, dropped = stepwise_vif_selection(X, threshold=VIF_THRESHOLD, max_steps=200)

    # 5) 打印日志
    print("\n=== VIF 逐步筛选日志（每轮最高值在首行）===")
    for i, vif_df in enumerate(history, 1):
        top = vif_df.iloc[0]
        print(f"第 {i:02d} 轮：最高 VIF = {top['VIF']:.3f}（{top['feature']}）; 特征数={len(vif_df)}")

    if dropped:
        print("\n被剔除特征顺序：", " -> ".join(dropped))
    else:
        print("\n未剔除任何特征（均 <= 阈值）。")

    print("\n=== 最终保留特征及其 VIF ===")
    print(final_vif.to_string(index=False))

    # 6) 导出结果
    out_path = Path(OUTPUT_FILE).resolve()
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 每一轮 VIF
        for i, vif_df in enumerate(history, 1):
            vif_df.to_excel(writer, sheet_name=f"round_{i:02d}_vif", index=False)
        # 最终 VIF
        final_vif.to_excel(writer, sheet_name="final_vif", index=False)
        # 剔除顺序
        pd.DataFrame({"dropped_order": dropped}).to_excel(writer, sheet_name="dropped_order", index=False)
    print(f"\n结果已导出：{out_path}")


if __name__ == "__main__":
    main()
