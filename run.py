# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif, load_static_embedding
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.amp import GradScaler

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--static-emb-path', type=str, default='./pretrained/bert_pretrained/sgns.merge.char', help='静态词向量文件路径')
parser.add_argument('--seed', type=int, default=109, help='random seed')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of static embeddings')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'CSTA-Corpus' 
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证可复现性
    
    start_time = time.time()
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    test_iter = build_iterator(test_data, config, return_contents=True)
    
    # 初始化混合精度缩放器
    scaler = GradScaler('cuda')
    config.scaler = scaler
    
    if model_name == 'bert':
        # print(f"检测到模型为 {model_name}，正在加载静态词向量...")
        print(f"检测到模型为 {model_name}，正在加载静态词向量 (Dim={args.emb_dim})...")
        # static_emb_matrix = load_static_embedding(args.static_emb_path, config.tokenizer)
        static_emb_matrix = load_static_embedding(args.static_emb_path, config.tokenizer, embed_dim=args.emb_dim)
        model = x.Model(config, static_emb_matrix=static_emb_matrix).to(config.device)
    else:
        model = x.Model(config).to(config.device)

    # 训练
    acc, f1 = train(config, model, train_iter, dev_iter, test_iter, train_data)
    
    # 【关键】最后打印一行特定的格式，方便我们用脚本抓取结果
    print(f"FINAL_RESULT: Seed={seed}, ACC={acc:.4f}, F1={f1:.4f}")