# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import os
from torch.amp import autocast

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'text_rcnn'
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
        self.num_epochs = 10
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 2e-5
        self.bert_path = './pretrained/bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.rnn_hidden = 256
        self.num_layers = 1
        self.dropout = 0.1

class Model(nn.Module):
    def __init__(self, config, static_emb_matrix=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
            
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        # RCNN max-pooling 后的投影层
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)
        
        # === 兼容性初始化 ===
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0]  # [batch, seq_len]
        mask = x[2]     # [batch, seq_len]
        
        with autocast('cuda'):
            # [batch, seq, hidden]
            embed = self.bert(context, attention_mask=mask)[0]
            
            # [batch, seq, rnn_hidden*2]
            out, _ = self.lstm(embed)
            
            # RCNN的核心：将 Embedding 和 LSTM输出 拼接
            # [batch, seq, hidden + rnn_hidden*2]
            out = torch.cat((embed, out), 2)
            
            # 激活函数
            out = F.relu(out)
            
            # [batch, hidden + rnn_hidden*2, seq]
            out = out.permute(0, 2, 1)
            
            # Max Pooling over time
            # [batch, hidden + rnn_hidden*2, 1] -> squeeze -> [batch, hidden + rnn_hidden*2]
            pooled = F.max_pool1d(out, out.size(2)).squeeze(2)
            
            # === 保存中间结果 ===
            self.hidden_states = pooled.detach()
            
            # [关键修改] 生成伪造的注意力权重以适配 train_eval.py
            # TextRCNN 使用 Max Pooling，没有显式的 Token 级注意力
            # 我们基于 mask 生成一个均匀分布的权重（所有非 Padding 词权重相等）
            dummy_attn = mask.float()
            # 归一化使得和为1，防止除以0
            dummy_attn = dummy_attn / (dummy_attn.sum(dim=1, keepdim=True) + 1e-9)
            self.attention_weights = dummy_attn.detach()
            
            logits = self.fc(pooled)
            
            # === 保存概率分布 ===
            self.probabilities = F.softmax(logits, dim=1).detach()
            
        return logits