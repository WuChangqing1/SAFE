# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import os
# 【重点】不导入 autocast，确保全程 FP32

class Config(object):
    """配置参数 - 高显存消耗基线版"""
    def __init__(self, dataset):
        self.model_name = 'base_bert_fp32'
        self.dataset = dataset
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', 'r', encoding='utf-8').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        
        save_dir = os.path.dirname(self.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 1
        
        # 【关键修改1】Batch Size 从 64 降为 8
        # 原因：FP32模式下，512长度的序列极其吃显存。BS=8 大概会占用 6GB-7.5GB 显存。
        # 对比：你的 SAFE 模型可以用 BS=64，吞吐量是这个基线的 8 倍！
        self.batch_size = 8  
        
        # 【关键修改2】保持 512 长度
        # 即使文本很短，也强制填充到 512，导致计算量和显存最大化
        self.pad_size = 512  
        
        self.learning_rate = 2e-5
        self.bert_path = './pretrained/bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.1

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 【关键修改3】添加 attn_implementation="eager"
        # 这消除了 "BertSdpaSelfAttention" 的警告，强制使用标准的 PyTorch Attention
        self.bert = BertModel.from_pretrained(
            config.bert_path, 
            attn_implementation="eager"
        )
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        # 兼容性变量 (供 train_eval.py 分析使用)
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def forward(self, x):
        # x: [batch(8), seq_len(512)]
        context = x[0]  
        mask = x[2]     
        
        # 【关键修改4】无 autocast 上下文
        # 强制使用 Float32 (4 Bytes) 运行，显存占用翻倍
        
        # output_attentions=True 会让模型额外存储注意力权重，进一步增加显存
        outputs = self.bert(context, attention_mask=mask, output_attentions=True)
        
        # 使用 Pooler Output (CLS + Tanh)
        pooler_output = outputs.pooler_output
        
        out = self.fc(pooler_output)
        
        # === 保存中间状态 ===
        self.probabilities = F.softmax(out, dim=1).detach()
        self.hidden_states = pooler_output.detach()
        
        # 提取最后一层的注意力权重用于分析
        # outputs.attentions 是 tuple，取最后一个 [-1]
        # 形状: [batch, num_heads, seq_len, seq_len] -> [8, 12, 512, 512]
        # 这个矩阵非常大，也是导致 OOM 的元凶之一
        last_layer_attn = outputs.attentions[-1]
        
        # 为了防止这里再次 OOM，我们尽快 detach 并取平均
        self.attention_weights = torch.mean(last_layer_attn, dim=1)[:, 0, :].detach()

        return out