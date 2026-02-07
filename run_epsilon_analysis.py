# run_epsilon_analysis.py
import torch
import numpy as np
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif, load_static_embedding
import os
# 【修复点1】导入 GradScaler
from torch.amp import GradScaler 

# 设定要测试的 epsilon 列表
epsilon_list = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
results_acc = []
results_f1 = []

# 初始化配置
dataset = 'CSTA-Corpus'  # 请确保这里是你的数据集目录名
model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config(dataset)

# 强制设置设备
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据（只加载一次以节省时间）
print("Loading data...")
train_data, dev_data, test_data = build_dataset(config)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config, return_contents=True)

# 预加载静态词向量 (路径需要根据你的实际情况调整，这里使用了 run.py 中的默认路径逻辑)
static_path = './pretrained/bert_pretrained/sgns.merge.char'
if not os.path.exists(static_path):
    # 尝试另一个常见路径
    static_path = config.bert_path + '/sgns.merge.char'

print(f"Loading static embeddings from {static_path}...")
static_emb_matrix = load_static_embedding(static_path, config.tokenizer)

for eps in epsilon_list:
    print(f"\n===== 开始训练: epsilon = {eps} =====")
    
    # 1. 动态设置 epsilon
    config.fgm_epsilon = eps
    
    # 【修复点2】每次训练前初始化混合精度缩放器，并挂载到 config 上
    # 必须在 device='cuda' 时才初始化，否则会报错（虽然你的代码基本都是跑在 GPU 上的）
    if torch.cuda.is_available():
        config.scaler = GradScaler('cuda')
    else:
        # 如果是 CPU 环境（通常不跑 AMP），给个假的 scaler 避免报错，或者直接报错提示
        raise RuntimeError("此脚本需要 CUDA 环境支持混合精度训练")
    
    # 初始化模型 (每次都重新初始化，保证参数重置)
    model = x.Model(config, static_emb_matrix=static_emb_matrix).to(config.device)
    
    # 训练并获取结果
    from train_eval import train
    acc, f1 = train(config, model, train_iter, dev_iter, test_iter, train_data)
    
    results_acc.append(acc)
    results_f1.append(f1)
    print(f"当前轮结果: eps={eps}, ACC={acc:.4f}, F1={f1:.4f}")

print("\n===== 所有训练结束 =====")
print("请将以下数据填入 plot_figure8.py 进行绘图：")
print(f"epsilons = {epsilon_list}")
print(f"acc_scores = {results_acc}")
print(f"f1_scores = {results_f1}")