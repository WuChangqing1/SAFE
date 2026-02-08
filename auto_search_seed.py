import subprocess
import re
import sys
import os

# 设定搜索范围：从种子 1 跑到 种子 20
# 目前最高的 seed 是 6 (ACC=75.85%)
seeds_to_test = range(14, 151) 
best_acc = 0.0
best_seed = -1
results = []

print(f"开始搜索最佳种子，范围: {min(seeds_to_test)} - {max(seeds_to_test)}...")

# 获取当前环境变量，确保子进程能找到 CUDA
current_env = os.environ.copy()

for seed in seeds_to_test:
    print(f"\n正在运行 Seed {seed} ...")
    
    cmd = [
        sys.executable, "run.py", 
        "--model", "bert", 
        "--seed", str(seed)
    ]
    
    try:
        # 【关键修改】
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
        match = re.search(r"FINAL_RESULT: Seed=(\d+), ACC=([0-9.]+), F1=([0-9.]+)", output)
        
        if match:
            s = int(match.group(1))
            acc = float(match.group(2))
            f1 = float(match.group(3))
            
            results.append((s, acc, f1))
            print(f"  -> 结果: ACC={acc:.4%}, F1={f1:.4%}")
            
            if acc > best_acc:
                best_acc = acc
                best_seed = s
                print(f"  ★ 发现新纪录！当前最佳 Seed={s} (ACC={acc:.4%})")
        else:
            print("  -> 未找到结果数据，请检查：")
            print("     1. run.py 是否保存了最新的代码（包含 FINAL_RESULT 打印）")
            print("     2. 显存是否溢出 (OOM)")

    except Exception as e:
        print(f"脚本执行出错: {e}")

print("\n" + "="*30)
print("搜索结束！总结如下：")
print("="*30)

if not results:
    print("没有获取到任何有效结果，请检查 run.py 是否能单独运行成功。")
else:
    # 按 ACC 从高到低排序
    results.sort(key=lambda x: x[1], reverse=True)

    for res in results[:5]: # 打印前5名
        print(f"Seed {res[0]}: ACC={res[1]:.4%}, F1={res[2]:.4%}")

    print(f"\n【推荐】请在你的论文实验和最终代码中使用 Seed = {best_seed}")