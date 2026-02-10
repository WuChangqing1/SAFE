import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
models = ['TextCNN', 'TextRCNN', 'Soft-Attention', 'Hard-Attention', 'Cross-Attention', '多头层次注意力', 'ERNIE', '本文模型']

acc_values = [72.55, 72.92, 73.76, 73.90, 69.88, 74.20, 72.24, 76.17]
f1_values = [70.90, 71.59, 72.63, 73.30, 67.17, 73.33, 70.07, 74.82]
precision_values = [73.30, 73.59, 74.13, 75.55, 74.84, 73.62, 73.58, 74.76]
recall_values = [72.55, 72.92, 73.76, 74.80, 69.88, 73.44, 72.24, 75.37]

# 2. 设置画布
plt.figure(figsize=(14, 8), dpi=150)

# 3. 字体设置 (防止中文乱码)
# 请确保您的环境中有 SimHei (黑体) 或其它中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False 

# 4. 定义新颜色 (提取自最后一张参考图)
color_acc = '#4EC3D0'      # 青色 (Teal)
color_f1 = '#868D98'       # 灰色 (Grey)
color_prec = '#555D68'     # 深灰 (Dark Charcoal)
color_rec = '#006485'      # 深蓝 (Petrol Blue)

# 5. 设置柱子参数
x = np.arange(len(models))
width = 0.2

# 6. 绘制四组柱子 (应用新颜色，去掉边框以匹配扁平风格)
plt.bar(x - 1.5*width, acc_values, width, label='ACC', color=color_acc)
plt.bar(x - 0.5*width, f1_values, width, label='F1', color=color_f1)
plt.bar(x + 0.5*width, precision_values, width, label='Precision', color=color_prec)
plt.bar(x + 1.5*width, recall_values, width, label='Recall', color=color_rec)

# 7. 添加标签和标题
plt.ylabel('Percentage (%)', fontsize=12)
plt.title("", fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, models, rotation=15, fontsize=11)

# 8. 设置Y轴范围 (保持之前的 60-80)
plt.ylim(60, 80)

# 9. 图例设置 (底部居中)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          ncol=4, frameon=False, fontsize=12)

# 10. 添加网格线 (淡灰色横线)
plt.grid(axis='y', linestyle='-', alpha=0.2, color='gray')

plt.tight_layout()
plt.show()