import pandas as pd

def filter_gc_content(input_file, output_file):
    """
    读取Excel文件，筛选P列“GC含量”在[0.4, 0.6]之间的行，并保存到新文件。
    """
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 筛选条件：GC含量 >= 0.4 且 <= 0.6
    df_filtered = df[(df["GC含量"] >= 0.4) & (df["GC含量"] <= 0.6)]

    # 保存到新文件
    df_filtered.to_excel(output_file, index=False)
    print(f"筛选完成，结果已保存到: {output_file}")


if __name__ == "__main__":
    input_path = "男胎检测数据_处理后.xlsx"  # 输入文件路径
    output_path = "男胎检测数据_处理后_filtered.xlsx"  # 输出文件路径

    filter_gc_content(input_path, output_path)
