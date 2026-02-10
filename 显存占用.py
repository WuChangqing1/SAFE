import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
models = [
    "Base-BERT\n(FP32)",   # 未优化基线
    "TextCNN", 
    "TextRCNN", 
    "Hard-Attn", 
    "Cross-Attn", 
    "Soft-Attn", 
    "ERNIE", 
    "SAFE\n(Ours)"         # 您的模型 (Log中的BERT)
]

# 对应的显存占用 (MB)
memory_usage = [
    6758.06, # Base-BERT
    4295.76, # TextCNN
    4433.44, # TextRCNN
    4317.65, # Hard-Attn
    4279.17, # Cross-Attn
    4426.44, # Soft-Attn
    5088.87, # ERNIE
    3896.10  # SAFE (原Log中的BERT)
]

# 2. 设置配色
colors = ['#333333'] + ['#A9A9A9'] * 5 + ['#4682B4', '#D9534F']

# 3. 创建画布 (缩小整体尺寸：从9,5.5调整为7,4)
plt.figure(figsize=(7, 4), dpi=300)

# 4. 绘制柱状图
bars = plt.bar(models, memory_usage, color=colors, width=0.65, zorder=3)

# 5. 设置Y轴范围 (稍微留出头部空间)
plt.ylim(0, 7800)

# 6. 添加网格线 (仅Y轴)
plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# 7. 添加数值标签 (字体也适当缩小：从9调整为8)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 150,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

# 8. 设置标题和坐标轴标签 (字体缩小：从11调整为10)
plt.ylabel('Peak Memory Usage (MB)', fontsize=10, fontweight='bold')

# 9. 优化刻度显示 
plt.xticks(fontsize=8, rotation=0)  # 模型名字字体缩小：从10调整为8
plt.yticks(fontsize=8)              # Y轴刻度字体同步缩小：从10调整为8

# 10. 添加优势标注 (核心修改：字体从9调整为7)
save_percent = (6713.43 - 3896.10) / 6713.43 * 100
plt.text(7, 4400, f"↓ {save_percent:.1f}% vs Base", 
         ha='center', va='bottom', fontsize=6, color='#D9534F', fontweight='bold')

# 11. 调整布局防止遮挡
plt.tight_layout()

# 12. 保存并显示
plt.savefig('memory_benchmark_small.png')
plt.show()