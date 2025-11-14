import pandas as pd

# 读取文件
file_path = "男胎检测数据_处理后_filtered_pairwise.xlsx"
df = pd.read_excel(file_path)

# 新增两列：检测孕周^2 和 孕妇BMI^2
df["检测孕周^2"] = df["检测孕周"] ** 2
df["孕妇BMI^2"] = df["孕妇BMI"] ** 2

# 保存为新文件
output_path = "男胎检测数据_处理后_filtered_pairwise_with_squares.xlsx"
df.to_excel(output_path, index=False)

print("处理完成，已保存到：", output_path)
