import matplotlib.pyplot as plt
import numpy as np

# 1. 从你的日志文件中提取的真实数据
epsilons = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

# 原始数据是 0.7590... 这种格式，我们需要乘以 100 变成百分比
# acc_scores_raw = [0.7590361445783133, 0.7527501309586171, 0.7527501309586171, 0.7579884756416972, 0.7564169722367732, 0.7564169722367732, 0.7585123101100052]
# f1_scores_raw = [0.7437795190600207, 0.7385275234079298, 0.738504736111474, 0.7444290777354203, 0.7426469493120907, 0.7460865935406814, 0.7467197523231343]
acc_scores = [0.7527501309586171, 0.7585123101100052, 0.7553693033001572, 0.7517024620220011, 0.7590361445783133, 0.7574646411733892, 0.752226296490309]       
f1_scores = [0.7378327115873794, 0.7444704807365731, 0.7412521103967189, 0.7371338621018906, 0.7469442417847453, 0.747237658246336, 0.7363190776701323]
# 转换为百分比并保留两位小数
acc_scores = [x * 100 for x in acc_scores]
f1_scores = [x * 100 for x in f1_scores]

# 2. 设置绘图风格 (学术风)
# 如果系统没有安装 Times New Roman，可以注释掉下面这行，或者换成 'serif'
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots(figsize=(8, 6))

# 3. 绘制 Accuracy (左轴)
color = 'tab:blue'
ax1.set_xlabel(r'Perturbation Magnitude ($\epsilon$)', fontsize=14)
ax1.set_ylabel('Accuracy (%)', color=color, fontsize=14)
line1 = ax1.plot(epsilons, acc_scores, marker='o', linestyle='-', linewidth=2, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# 设置左轴范围，让波动看起来明显一点 (可选)
# ax1.set_ylim(74.0, 76.5)

# 4. 绘制 F1-Score (右轴)
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('F1-Score (%)', color=color, fontsize=14)
line2 = ax2.plot(epsilons, f1_scores, marker='s', linestyle='--', linewidth=2, color=color, label='F1-Score')
ax2.tick_params(axis='y', labelcolor=color)

# 设置右轴范围 (可选)
# ax2.set_ylim(73.0, 75.0)

# 5. 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', fontsize=12)

# 6. 设置标题和布局
plt.title(r"", fontsize=16, pad=15)
plt.tight_layout()

# 7. 保存图片
plt.savefig('Figure8_Epsilon_Sensitivity.png', dpi=300)
print("图表已生成: Figure8_Epsilon_Sensitivity.png")
plt.show()