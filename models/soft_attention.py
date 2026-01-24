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
        self.model_name = 'soft_attention'
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
        self.num_epochs = 20
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
        
        # Attention Layers
        self.w_omega = nn.Parameter(torch.Tensor(config.rnn_hidden * 2, config.rnn_hidden * 2))
        self.u_omega = nn.Parameter(torch.Tensor(config.rnn_hidden * 2, 1))
        
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        
        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

        # === 兼容性初始化 ===
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def attention_net(self, x, mask):
        # x: [batch, seq_len, hidden*2]
        # mask: [batch, seq_len]
        
        # u = tanh(x * w)
        u = torch.tanh(torch.matmul(x, self.w_omega)) # [batch, seq, hidden*2]
        
        # att = u * u_omega
        att = torch.matmul(u, self.u_omega) # [batch, seq, 1]
        
        # Masking padding (使用 -1e4 防止 fp16 溢出)
        mask_expanded = mask.unsqueeze(-1).float() # [batch, seq, 1]
        att = att.masked_fill(mask_expanded == 0, -1e4)
        
        att_score = F.softmax(att, dim=1) # [batch, seq, 1]
        
        # === 保存注意力权重 [batch, seq] ===
        self.attention_weights = att_score.squeeze(-1).detach()
        
        # Weighted sum
        scored_x = x * att_score
        context = torch.sum(scored_x, dim=1) # [batch, hidden*2]
        return context

    def forward(self, x):
        context = x[0]
        mask = x[2]
        
        with autocast('cuda'):
            bert_out = self.bert(context, attention_mask=mask)[0] # [batch, seq, 768]
            lstm_out, _ = self.lstm(bert_out) # [batch, seq, 2*rnn_hidden]
            
            # 计算 Attention 并聚合
            attn_out = self.attention_net(lstm_out, mask)
            
            # === 保存隐层状态 ===
            self.hidden_states = attn_out.detach()
            
            logits = self.fc(attn_out)
            
            # === 保存概率分布 ===
            self.probabilities = F.softmax(logits, dim=1).detach()
            
        return logits