import matplotlib.pyplot as plt
import numpy as np

# --- 1. 准备数据 (仅保留两组) ---
models = ['BERT-base (FP32)', 'SAFE (Ours, AMP)']
# 数值参考: BERT约8450MB, Ours约4280MB
memory_vals = [8450, 4857]

# --- 2. 设置画布 ---
plt.figure(figsize=(6, 5), dpi=150) # 调小一点尺寸，适合两根柱子
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 3. 绘制柱状图 ---
colors = ['#8E97A3', '#4EC3D0'] # 灰色 vs 青色
bars = plt.bar(models, memory_vals, color=colors, width=0.4, edgecolor='white', linewidth=0.5)

# --- 4. 添加数值标签 ---
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{int(height)} MB',
             ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')

# --- 5. 添加"降低约50%"的标注箭头 ---
# 计算箭头起止点
x_start = bars[0].get_x() + bars[0].get_width()/2
y_start = bars[0].get_height()
x_end = bars[1].get_x() + bars[1].get_width()/2
y_end = bars[1].get_height()

# 绘制弯曲箭头
plt.annotate('显存降低 ~49.3%',
             xy=(x_end, y_end), xycoords='data',
             xytext=(x_end + 0.5, (y_start+y_end)/2), textcoords='data', # 文字放中间偏右
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color='#E15759', lw=2),
             ha='left', va='center', fontsize=11, color='#E15759', fontweight='bold')

# --- 6. 美化 ---
plt.ylabel('Peak GPU Memory Usage (MB)', fontsize=11)
plt.title("", fontsize=13, fontweight='bold', pad=15)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.ylim(0, 10000)

plt.tight_layout()
plt.show()