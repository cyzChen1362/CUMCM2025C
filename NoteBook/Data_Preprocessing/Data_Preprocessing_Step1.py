# =================================================
# Conversion of gestational weeks units
# =================================================

import pandas as pd
import re

def convert_gestational_age(value):
    """
    将检测孕周转换为小数
    """
    if isinstance(value, str):
        value = value.strip()
        # 匹配 "11w+6" 或 "11W+6"
        match_plus = re.match(r"(\d+)[wW]\+(\d+)", value)
        if match_plus:
            weeks = int(match_plus.group(1))
            days = int(match_plus.group(2))
            return round(weeks + days / 7, 2)

        # 匹配 "13w" 或 "13W"
        match_week = re.match(r"(\d+)[wW]", value)
        if match_week:
            return int(match_week.group(1))

    return value

def clean_last_menstruation(value):
    """
    清理“末次月经”列:
        datetime → 'YYYY/MM/DD'
        '2023/2/1 0:00:00' → '2023/2/1'
    """
    if pd.isna(value):
        return value  # 保留空值

    # 如果是 datetime 类型
    if hasattr(value, "strftime"):
        return value.strftime("%Y/%m/%d")

    # 如果是字符串
    if isinstance(value, str):
        return value.replace(" 0:00:00", "").strip()

    return str(value)  # 其它情况强制转为字符串

def process_excel(input_file, output_file):
    # 读取“男胎检测数据”工作表
    df = pd.read_excel(input_file, sheet_name="男胎检测数据")

    # 生成“NIPT准确性”列
    df["NIPT准确性"] = df["Y染色体浓度"].apply(lambda x: "是" if x > 0.04 else "否")

    # 转换“检测孕周”列
    if "检测孕周" in df.columns:
        df["检测孕周"] = df["检测孕周"].apply(convert_gestational_age)

    # 清理“末次月经”列
    if "末次月经" in df.columns:
        df["末次月经"] = df["末次月经"].apply(clean_last_menstruation)

    # 保存到新的 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"处理完成，保存为: {output_file}")

if __name__ == "__main__":
    input_path = "../Data/附件.xlsx"  # 输入文件路径
    output_path = "../Data/男胎检测数据_处理后.xlsx"  # 输出文件路径
    process_excel(input_path, output_path)