import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 基本设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
file_path = '../Data/附件.xlsx'

# GC 含量正常范围
GC_LOW = 0.4
GC_HIGH = 0.6

# 读取数据
male_df = pd.read_excel(file_path, sheet_name='男胎检测数据')
female_df = pd.read_excel(file_path, sheet_name='女胎检测数据')

# 确保 GC 含量是数值类型，并去掉缺失
male_gc = pd.to_numeric(male_df['GC含量'], errors='coerce').dropna()
female_gc = pd.to_numeric(female_df['GC含量'], errors='coerce').dropna()

# 定义一个通用绘图函数
def plot_gc_qc(ax_row, gc_series, label, bins_size1 = None, bins_size2 = None):
    """
    ax_row: axes[行索引, :]，长度为3的一行Axes
    gc_series: 该表的GC含量Series
    label: '男胎' 或 '女胎'
    """
    total = len(gc_series)
    mask_ok = (gc_series >= GC_LOW) & (gc_series <= GC_HIGH)
    keep_gc = gc_series[mask_ok]
    keep_count = mask_ok.sum()
    drop_count = total - keep_count
    keep_ratio = keep_count / total
    drop_ratio = drop_count / total

    # 过滤前直方图
    ax1 = ax_row[0]
    ax1.hist(gc_series, bins=bins_size1, color='#d9d9d9', edgecolor='black')
    ax1.axvline(GC_LOW, color='red', linestyle='--', linewidth=1.5, label='40%下限')
    ax1.axvline(GC_HIGH, color='red', linestyle='--', linewidth=1.5, label='60%上限')
    ax1.set_title(f'{label}GC含量分布（过滤前）')
    ax1.set_xlabel('GC含量')
    ax1.set_ylabel('频数')
    ax1.set_xlim(0.35, 0.65)
    ax1.legend()

    # 过滤后直方图
    ax2 = ax_row[1]
    ax2.hist(keep_gc, bins=bins_size2, color='#99d98c', edgecolor='black')
    ax2.set_title(f'{label}GC含量分布（过滤后）')
    ax2.set_xlabel('GC含量')
    ax2.set_ylabel('频数')
    # 过滤后 x 轴稍微缩小一点范围
    if not keep_gc.empty:
        ax2.set_xlim(max(keep_gc.min() - 0.002, 0), keep_gc.max() + 0.002)

    # 饼图
    ax3 = ax_row[2]
    sizes = [keep_count, drop_count]
    labels = ['保留', '过滤']
    colors = ['#8fd175', '#ff6f69']
    explode = (0.03, 0)

    wedges, texts, autotexts = ax3.pie(
        sizes,
        labels=labels,
        autopct='%.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={'fontsize': 10}
    )
    ax3.set_title(f'{label}样本过滤结果\n({keep_count}/{total})')
    ax3.axis('equal')  # 让饼图为正圆

    # 在控制台打印统计信息
    print(f'[{label}] 总样本数: {total}')
    print(f'[{label}] 保留样本数: {keep_count} ({keep_ratio:.2%})')
    print(f'[{label}] 过滤样本数: {drop_count} ({drop_ratio:.2%})')
    print('-' * 40)

# 生成六合一图
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 男胎
plot_gc_qc(axes[0, :], male_gc, '男胎', 46, 30)
# 女胎
plot_gc_qc(axes[1, :], female_gc, '女胎', 30, 20)

fig.suptitle('GC含量质量控制分析 - 男胎vs女胎', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存图片
plt.savefig('./Images/picture10.png', dpi=300, bbox_inches='tight')
plt.show()
