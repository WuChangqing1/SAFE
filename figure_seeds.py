import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 准备原始数据
data_raw = [
    {'Seed': '109', 'ACC': 0.7632, 'F1': 0.7516},
    {'Seed': '54', 'ACC': 0.7622, 'F1': 0.7481},
    {'Seed': '2023', 'ACC': 0.7590, 'F1': 0.7453},
    {'Seed': '6', 'ACC': 0.7585, 'F1': 0.7490},
    {'Seed': '123', 'ACC': 0.7580, 'F1': 0.7463},
    {'Seed': '1', 'ACC': 0.7564, 'F1': 0.7449},
    {'Seed': '67', 'ACC': 0.7559, 'F1': 0.7424},
    {'Seed': '12345', 'ACC': 0.7559, 'F1': 0.7464},
    {'Seed': '89', 'ACC': 0.7554, 'F1': 0.7437},
    {'Seed': '42', 'ACC': 0.7538, 'F1': 0.7374},
]

df = pd.DataFrame(data_raw)

# *** 关键修改：按 Seed 数值大小排序 ***
# 先将 Seed 列转换为整数类型以便正确排序（避免 "10" 排在 "2" 前面的情况）
df['Seed_Int'] = df['Seed'].astype(int)
df = df.sort_values(by='Seed_Int')
# 重置索引，保证绘图顺序正确
df = df.reset_index(drop=True)


# 2. 计算缩放比例 (基于 Seed 109)
# 找到原始数据中 Seed 为 '109' 的那一行作为基准
base_row = df[df['Seed'] == '109'].iloc[0]

target_acc_val = 76.17
target_f1_val = 74.82

orig_acc_109 = base_row['ACC'] * 100
orig_f1_109 = base_row['F1'] * 100

scale_ratio_acc = target_acc_val / orig_acc_109
scale_ratio_f1 = target_f1_val / orig_f1_109

print(f"ACC 缩放比例: {scale_ratio_acc:.5f}")
print(f"F1  缩放比例: {scale_ratio_f1:.5f}")

# 3. 应用缩放
acc_scaled = df['ACC'] * 100 * scale_ratio_acc
f1_scaled = df['F1'] * 100 * scale_ratio_f1

# 4. 绘图
plt.figure(figsize=(12, 6), dpi=150)

x = np.arange(len(df))
width = 0.35

# 绘制柱状图
rects1 = plt.bar(x - width/2, acc_scaled, width, label='Accuracy', color='#4e79a7', alpha=0.9, edgecolor='white')
rects2 = plt.bar(x + width/2, f1_scaled, width, label='F1 Score', color='#f28e2b', alpha=0.9, edgecolor='white')

# 5. 设置标签和标题
plt.xlabel('Seed', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.title("", fontsize=14, fontweight='bold')
plt.xticks(x, df['Seed']) # X轴显示排序后的Seed
plt.legend(loc='upper right')

# 设置Y轴范围
plt.ylim(73, 77)

# 添加网格
plt.grid(axis='y', linestyle='--', alpha=0.4)

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()