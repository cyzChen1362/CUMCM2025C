import re
import numpy as np
import pandas as pd

# ======= 路径与目标工作表 =======
in_path  = "..\Data\附件.xlsx"
sheet    = "女胎检测数据"
out_path = "..\Data\附件_女胎检测数据_处理后.xlsx"

# ======= 工具=======
def to_weeks(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 2)

    s = str(value).strip()
    m_plus = re.match(r"^\s*(\d+)\s*[wW]\s*\+\s*(\d+)\s*$", s)
    if m_plus:
        w = int(m_plus.group(1)); d = int(m_plus.group(2))
        return round(w + d/7.0, 2)

    m_w = re.match(r"^\s*(\d+)\s*[wW]\s*$", s)
    if m_w:
        return round(float(m_w.group(1)), 2)

    m_num = re.match(r"^\s*(\d+(?:\.\d+)?)\s*$", s)
    if m_num:
        return round(float(m_num.group(1)), 2)

    return np.nan

# ======= 读取目标表 =======
xl = pd.ExcelFile(in_path)
if sheet not in xl.sheet_names:
    raise ValueError(f"未找到工作表：{sheet}；现有工作表：{xl.sheet_names}")
df = pd.read_excel(in_path, sheet_name=sheet)

# ======= 1) 删除 U、V 两列 =======
if df.shape[1] >= 22:
    cols_to_drop = []
    try:
        cols_to_drop.append(df.columns[20])  # U
        cols_to_drop.append(df.columns[21])  # V
    except Exception:
        cols_to_drop = []
    if cols_to_drop:
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 2) 新增“染色体是否正常”
col_aneu = "染色体的非整倍体"
if col_aneu not in df.columns:
    alt = [c for c in df.columns if "非整倍体" in str(c)]
    if alt:
        col_aneu = alt[0]
    else:
        df[col_aneu] = np.nan

df["染色体是否正常"] = np.where(
    df[col_aneu].astype(str).str.strip().replace({"nan": ""}) == "",
    "是", "否"
)

df["染色体是否正常0-1变量"] = np.where(df["染色体是否正常"] == "是", 1, 0)

# ======= 3) 规范“检测孕周” =======
col_weeks = "检测孕周"
if col_weeks not in df.columns:
    alt = [c for c in df.columns if "孕周" in str(c)]
    if alt:
        col_weeks = alt[0]
    else:
        raise ValueError("未找到“检测孕周”列，也未识别到包含“孕周”的别名。")
df[col_weeks] = df[col_weeks].apply(to_weeks)

# ======= 4) “孕妇BMI”缺失均值填充=======
bmi_candidates = ["孕妇BMI", "BMI", "孕妇BM"]
bmi_col = None
for c in bmi_candidates:
    if c in df.columns:
        bmi_col = c
        break
if bmi_col is None:
    if {"身高", "体重"}.issubset(df.columns):
        df["孕妇BMI"] = df["体重"] / (df["身高"] / 100.0) ** 2
        bmi_col = "孕妇BMI"
    else:
        raise ValueError("未找到“孕妇BMI/BMI/孕妇BM”，且无法由身高/体重计算。")
df[bmi_col] = pd.to_numeric(df[bmi_col], errors="coerce")
df[bmi_col] = df[bmi_col].fillna(df[bmi_col].mean(skipna=True))

# ======= 5) 复制“GC含量”为“原GC含量” =======
gc_col = "GC含量"
if gc_col not in df.columns:
    alt = [c for c in df.columns if "GC" in str(c) and "含量" in str(c) and "染色体" not in str(c)]
    if alt:
        gc_col = alt[0]
    else:
        raise ValueError("未找到“GC含量”列。")
df["原GC含量"] = df[gc_col]

# ======= 6) 复制“检测孕周”“孕妇BMI”为备份列 =======
df["原检测孕周"] = df[col_weeks]
df["原孕妇BMI"] = df[bmi_col]
if "末次月经" in df.columns:
    df["末次月经"] = pd.to_datetime(df["末次月经"], errors="coerce").dt.date

# ======= 保存 =======
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name=sheet, index=False)

print(f"处理完成！已保存：{out_path}")