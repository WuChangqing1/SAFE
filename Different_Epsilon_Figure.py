import matplotlib.pyplot as plt
import numpy as np

epsilons = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

acc_scores = [0.7548454688318491, 0.7616553169198533, 0.7548454688318491, 0.7537977998952331, 0.7574646411733892, 0.7569408067050812, 0.7569408067050812]      
f1_scores = [0.7431652356414435, 0.748187656151521, 0.7406153796197987, 0.742002696745374, 0.7446928117602392, 0.7443590506304786, 0.7429745756392466]
acc_scores = [x * 100 for x in acc_scores]
f1_scores = [x * 100 for x in f1_scores]

plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots(figsize=(8, 6))

color = 'tab:blue'
ax1.set_xlabel(r'Perturbation Magnitude ($\epsilon$)', fontsize=14)
ax1.set_ylabel('Accuracy (%)', color=color, fontsize=14)
line1 = ax1.plot(epsilons, acc_scores, marker='o', linestyle='-', linewidth=2, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('F1-Score (%)', color=color, fontsize=14)
line2 = ax2.plot(epsilons, f1_scores, marker='s', linestyle='--', linewidth=2, color=color, label='F1-Score')
ax2.tick_params(axis='y', labelcolor=color)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', fontsize=12)

plt.title(r"", fontsize=16, pad=15)
plt.tight_layout()

plt.savefig('Figure8_Epsilon_Sensitivity.png', dpi=300)
print("图表已生成: Figure8_Epsilon_Sensitivity.png")
plt.show()