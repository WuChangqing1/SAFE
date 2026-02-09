import subprocess
import re
import matplotlib.pyplot as plt
import sys
import os
import time

def run_experiment_and_plot():
    # ================= 配置区域 =================
    # 1. 确保这个路径是正确的！必须指向你的 300维 sgns 文件
    static_emb_path = './pretrained/bert_pretrained/sgns.merge.char' 
    
    # 2. 测试维度列表
    # 注意：不能超过源文件的最大维度（通常是300）。
    # 如果你尝试读取 400，utils.py 会返回空矩阵，导致效果极差或报错。
    dims_to_test = [50, 100, 150, 200, 300] 
    
    # ===========================================

    if not os.path.exists(static_emb_path):
        print(f"❌ 错误: 找不到词向量文件: {static_emb_path}")
        print("请修改代码中的 static_emb_path 变量。")
        return

    results_acc = {}
    results_f1 = {}

    print(f"🚀 开始运行维度验证实验")
    print(f"📂 词向量源文件: {static_emb_path}")
    print(f"📊 测试维度: {dims_to_test}")
    print("="*60)

    for dim in dims_to_test:
        print(f"\n[Running] 正在训练维度: {dim} ...")
        start_t = time.time()
        
        # 构造运行命令
        cmd = [
            sys.executable, 'run.py',
            '--model', 'bert',
            '--static-emb-path', static_emb_path,
            '--emb_dim', str(dim),
            '--seed', '109'  # 固定种子，保证不同维度的比较是公平的
        ]
        
        try:
            # 运行命令，并捕获输出
            # 使用 Popen 实时流式输出，防止程序看起来像死机了
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore', # 忽略编码错误
                bufsize=1
            )
            
            stdout_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(f"    | {line.strip()}") # 缩进打印子进程输出
                    stdout_lines.append(line)
            
            # 等待结束
            stdout_full = "".join(stdout_lines)
            _, stderr_full = process.communicate()

            if process.returncode != 0:
                print(f"❌ 实验 {dim} 维运行失败！")
                print("错误详情:\n", stderr_full)
                continue
            
            # 正则提取结果
            # 匹配 run.py 中的: FINAL_RESULT: Seed=xx, ACC=0.xxxx, F1=0.xxxx
            match_f1 = re.search(r"F1=(\d+\.\d+)", stdout_full)
            match_acc = re.search(r"ACC=(\d+\.\d+)", stdout_full)
            
            if match_f1 and match_acc:
                acc_val = float(match_acc.group(1)) * 100
                f1_val = float(match_f1.group(1)) * 100
                
                results_acc[dim] = acc_val
                results_f1[dim] = f1_val
                
                duration = time.time() - start_t
                print(f"✅ 完成 {dim}维: F1={f1_val:.2f}%, ACC={acc_val:.2f}% (耗时 {duration:.0f}s)")
            else:
                print("⚠️ 警告: 未能在输出中找到 FINAL_RESULT。请检查 run.py 是否运行完整。")
                
        except Exception as e:
            print(f"❌ 运行过程发生异常: {str(e)}")
            return

    if not results_f1:
        print("\n❌ 没有获得任何有效结果，终止绘图。")
        return

    # ================= 绘图逻辑 =================
    print("\n🎨 正在绘制图表...")
    dims = sorted(results_f1.keys())
    f1_scores = [results_f1[d] for d in dims]
    acc_scores = [results_acc[d] for d in dims]

    plt.figure(figsize=(10, 6), dpi=120)
    plt.style.use('seaborn-v0_8-whitegrid') # 如果报错，可改为 'ggplot'
    
    # 绘制 F1 曲线
    plt.plot(dims, f1_scores, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='F1-Score')
    # 绘制 Accuracy 曲线 (虚线)
    plt.plot(dims, acc_scores, 's--', color='#1f77b4', linewidth=2, markersize=8, label='Accuracy', alpha=0.7)
    
    # 标注数值
    for x, y in zip(dims, f1_scores):
        plt.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10), 
                     textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#d62728')

    plt.title('Impact of Static Embedding Dimension on Model Performance', fontsize=14, pad=20)
    plt.xlabel('Embedding Dimension ($d_{static}$)', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xticks(dims)
    
    # 自动调整Y轴范围，使其美观
    all_scores = f1_scores + acc_scores
    min_s, max_s = min(all_scores), max(all_scores)
    plt.ylim(min_s - 0.5, max_s + 0.8)
    
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = 'dim_experiment_result.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"🎉 验证完成！图片已保存为: {save_path}")
    
    # 如果是在本地环境，取消下面这行的注释可以弹窗显示
    # plt.show()

if __name__ == "__main__":
    run_experiment_and_plot()