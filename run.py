# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif, load_static_embedding  # 导入加载函数
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.amp import GradScaler

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--static-emb-path', type=str, default='./pretrained/bert_pretrained/sgns.merge.char', help='静态词向量文件路径')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'CSTA-Corpus'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    
    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    test_iter = build_iterator(test_data, config, return_contents=True)
    
    # 加载静态词向量（float32，兼容混合精度）
    print("Loading static embedding...")
    static_emb_matrix = load_static_embedding(args.static_emb_path, config.tokenizer)
    
    # 初始化混合精度缩放器
    scaler = GradScaler('cuda')
    config.scaler = scaler
    
    # 初始化模型（传入静态词向量矩阵）
    model = x.Model(config, static_emb_matrix=static_emb_matrix).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter, train_data)