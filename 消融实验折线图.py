import matplotlib.pyplot as plt

# 1. 准备数据 (来自第一张图)
categories = ['移除静态词向量', '移除特征拼接', '移除对抗训练', '仅保留原始bert', '本文模型']
acc_values = [75.33, 74.86, 75.28, 75.33, 76.17]
f1_values = [74.02, 73.95, 74.40, 74.53, 74.82]

# 2. 设置画布
plt.figure(figsize=(10, 6), dpi=150) # 高分辨率

# 3. 字体设置 (关键步骤：防止中文乱码)
# 请确保您的环境中有 SimHei (黑体) 或其它中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 4. 绘制 ACC 折线 (黑色实线)
plt.plot(categories, acc_values, 
         color='black',       # 线条颜色
         linestyle='-',       # 实线
         linewidth=1.5,
         marker='^',          # 三角形标记
         markersize=8, 
         markerfacecolor='gray', # 标记填充色 (灰色，显得更有层次)
         markeredgecolor='black',
         label='ACC')

# 5. 绘制 F1 折线 (红色虚线)
plt.plot(categories, f1_values, 
         color='#d62728',     # 红色
         linestyle=':',       # 虚线 (对应图2风格)
         linewidth=1.5,
         marker='^',          # 三角形标记
         markersize=8, 
         markerfacecolor='white', # 标记内部白色 (模仿空心效果，也可改为红色)
         markeredgecolor='#d62728',
         markeredgewidth=1.5,
         label='F1 (%)')

# 6. 添加数值标签 (模仿图2的位置)
# ACC 数值显示在点上方
for i, val in enumerate(acc_values):
    plt.text(i, val + 0.15, f'{val}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')

# F1 数值显示在点下方
for i, val in enumerate(f1_values):
    plt.text(i, val - 0.25, f'{val}', 
             ha='center', va='top', fontsize=9, fontweight='bold', color='#d62728')

# 7. 调整坐标轴与图例
plt.ylim(73, 77) # 设置Y轴范围，让差异更明显
plt.legend(loc='upper center', frameon=False, ncol=2) # 图例置顶并横向排列

# 添加淡网格线
plt.grid(axis='y', linestyle='-', alpha=0.3)
plt.grid(axis='x', linestyle='-', alpha=0.1)

plt.tight_layout()
plt.show()