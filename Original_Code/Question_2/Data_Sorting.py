import pandas as pd

# 读取表格
file_path = r"/Original_Code/Data_Processing\男胎检测数据_处理后_filtered.xlsx"  # 输入文件路径
df = pd.read_excel(file_path)

# 拆分“是”和“否”
yes_df = df[df["NIPT准确性"] == "是"].copy()
no_df = df[df["NIPT准确性"] == "否"].copy()

# 重新排列
result_rows = []
yes_index, no_index = 0, 0

while yes_index < len(yes_df) and no_index < len(no_df):
    # 添加6行“是”
    result_rows.append(yes_df.iloc[yes_index:yes_index + 6])
    yes_index += 6

    # 添加1行“否”
    result_rows.append(no_df.iloc[no_index:no_index + 1])
    no_index += 1

# 不再添加剩余的“是”，因为要求丢弃

# 合并结果
result_df = pd.concat(result_rows, ignore_index=True)

# 保存结果
output_path = "NIPT_重新排序结果.xlsx"  # 输出文件路径
result_df.to_excel(output_path, index=False)

print(f"处理完成！结果已保存到 {output_path}")
