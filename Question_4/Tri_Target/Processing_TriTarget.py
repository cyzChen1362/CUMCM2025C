import re
import numpy as np
import pandas as pd
from pathlib import Path

# ========== 可调路径 ==========
in_path  = Path("附件.xlsx")
sheet    = "女胎检测数据"
out_path = Path("附件_女胎检测数据_处理后.xlsx")

# ========== 工具函数 ==========
def to_weeks(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # 13w+5 或 13W+5
    m = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m:
        w = int(m.group(1)); d = int(m.group(2))
        return round(w + d/7.0, 2)
    # 13w 或 13W
    m = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m:
        return float(m.group(1))
    # 纯数字
    m = re.match(r"^\s*\d+(\.\d+)?\s*$", s)
    if m:
        return float(s)
    return np.nan

def yn01(x):
    return "0" if str(x) == "否" else "1"

# ========== 读取 ==========
df = pd.read_excel(in_path, sheet_name=sheet)

# ========== 1) 删除 U、V 两列 ==========
try:
    cols_to_drop = []
    if len(df.columns) >= 21:
        cols_to_drop.append(df.columns[20])
    if len(df.columns) >= 22:
        cols_to_drop.append(df.columns[21])
    df = df.drop(columns=cols_to_drop, errors="ignore")
except Exception:
    pass

# ========== 2) 新增 “13/18/21号染色体是否正常” ==========
col_abn = "染色体的非整倍体"
s_abn = df.get(col_abn, pd.Series([np.nan]*len(df))).fillna("").astype(str)

is_T13 = s_abn.str.contains("T13", case=False, regex=False)
is_T18 = s_abn.str.contains("T18", case=False, regex=False)
is_T21 = s_abn.str.contains("T21", case=False, regex=False)

df["13号染色体是否正常"] = np.where(is_T13, "否", "是")
df["18号染色体是否正常"] = np.where(is_T18, "否", "是")
df["21号染色体是否正常"] = np.where(is_T21, "否", "是")

# ========== 3) 新增 对应 0-1 变量 ==========
df["13号染色体是否正常0-1变量"] = df["13号染色体是否正常"].map(yn01)
df["18号染色体是否正常0-1变量"] = df["18号染色体是否正常"].map(yn01)
df["21号染色体是否正常0-1变量"] = df["21号染色体是否正常"].map(yn01)

# ========== 4) 规范 “检测孕周” ==========
col_w = "检测孕周"
if col_w in df.columns:
    df[col_w] = df[col_w].apply(to_weeks)

# ========== 5) “孕妇BMI” 缺失均值填充 ==========
col_bmi = "孕妇BMI"
if col_bmi in df.columns:
    mean_bmi = pd.to_numeric(df[col_bmi], errors="coerce").mean()
    df[col_bmi] = pd.to_numeric(df[col_bmi], errors="coerce").fillna(mean_bmi)

# ========== 6) 复制 “原GC含量” ==========
col_gc = "GC含量"
if col_gc in df.columns and "原GC含量" not in df.columns:
    df["原GC含量"] = df[col_gc]

# ========== 7) “末次月经” 转标准日期（无时间） ==========
col_lmp = "末次月经"
if col_lmp in df.columns:
    dt = pd.to_datetime(df[col_lmp], errors="coerce")
    df[col_lmp] = dt.dt.date

# ========== 写出 ==========
out_path.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name=sheet)

print(f"处理完成：{out_path}")
