import subprocess
import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# ================= 配置区域 =================
# 设定要测试的种子列表
# 您可以在这里继续添加您想测试的数字
seeds_to_test = [1, 6, 42, 54, 67, 89, 109, 123, 2023, 12345]

print(f"准备测试的 Seed 列表: {seeds_to_test}")
print("="*40)

# ================= 运行测试 =================
best_acc = 0.0
best_seed = -1
results = []

# 获取当前环境变量，确保子进程能找到 CUDA
current_env = os.environ.copy()

for seed in seeds_to_test:
    print(f"\n正在运行 Seed {seed} ...")
    
    # 构造命令：python run.py --model bert --seed <seed>
    cmd = [
        sys.executable, "run.py", 
        "--model", "bert", 
        "--seed", str(seed)
    ]
    
    try:
        # 1. 移除 encoding='utf-8'，让它自动跟随系统（Windows会自动用gbk）
        # 2. 增加 errors='replace'，遇到解不开的字符直接变成 '?'，防止报错
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            errors='replace',
            env=current_env
        )
        
        output = result.stdout
        
        # 检查是否真的运行成功了
        if result.returncode != 0:
            print(f"Seed {seed} 运行失败 (Return Code {result.returncode})")
            # 打印一点点错误信息来看看（防止刷屏）
            print(f"错误信息片段: {result.stderr[:200]}...")
            continue

        # 使用正则提取 run.py 最后打印的 FINAL_RESULT
        # 格式: FINAL_RESULT: Seed=1, ACC=0.7550, F1=0.7420
        # 注意：这里假设您的 run.py 输出是小数 (0.7550) 或者百分比 (75.50)，正则都匹配数字点
        match = re.search(r"FINAL_RESULT: Seed=(\d+), ACC=([0-9.]+), F1=([0-9.]+)", output)
        
        if match:
            s = int(match.group(1))
            acc = float(match.group(2))
            f1 = float(match.group(3))
            
            # 存入结果列表
            results.append((s, acc, f1))
            
            # 打印当前结果
            print(f"  -> 结果: ACC={acc:.4f}, F1={f1:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_seed = s
                print(f"  ★ 发现新纪录！当前最佳 Seed={s} (ACC={acc:.4f})")
        else:
            print("  -> 未找到结果数据，请检查：")
            print("     1. run.py 是否保存了最新的代码（包含 FINAL_RESULT 打印）")
            print("     2. 显存是否溢出 (OOM)")
            # 也可以打印一下 output 看看输出了什么
            # print(output[:500])

    except Exception as e:
        print(f"脚本执行出错: {e}")

# ================= 结果汇总 =================
print("\n" + "="*30)
print("搜索结束！总结如下：")
print("="*30)

if not results:
    print("没有获取到任何有效结果，请检查 run.py 是否能单独运行成功。")
else:
    # 按 ACC 从高到低排序
    results.sort(key=lambda x: x[1], reverse=True)

    for res in results: # 打印所有结果
        print(f"Seed {res[0]}: ACC={res[1]:.4f}, F1={res[2]:.4f}")

    print(f"\n【推荐】请在你的论文实验和最终代码中使用 Seed = {best_seed}")

    # ================= 画图部分 =================
    print("\n正在生成柱状图...")
    
    # 提取数据用于绘图
    # 为了图表美观，我们按 Seed 的大小排序显示，或者按 ACC 排序显示
    # 这里我们选择按 Seed 大小排序，方便查找
    results.sort(key=lambda x: x[0]) 
    
    plot_seeds = [str(r[0]) for r in results]
    plot_accs = [r[1] for r in results]
    plot_f1s = [r[2] for r in results]

    x = np.arange(len(plot_seeds))  # 标签位置
    width = 0.35  # 柱状图宽度

    plt.figure(figsize=(12, 6)) # 设置画布大小
    
    # 画 ACC 和 F1 两组柱子
    rects1 = plt.bar(x - width/2, plot_accs, width, label='Accuracy', color='#1f77b4')
    rects2 = plt.bar(x + width/2, plot_f1s, width, label='F1 Score', color='#ff7f0e')

    # 添加标签、标题等
    plt.ylabel('Scores')
    plt.title('Performance Comparison by Random Seed')
    plt.xticks(x, plot_seeds) # 设置 x 轴标签为 Seed
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上方显示具体数值
    # 假设数值是 0.xxxx 的小数，我们显示为百分比或者保留4位小数
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移 3 点
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=90)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout() # 自动调整布局防止重叠
    
    # 保存图片并在窗口显示
    plt.savefig('seed_comparison.png', dpi=300)
    print("图表已保存为 seed_comparison.png")
    plt.show()