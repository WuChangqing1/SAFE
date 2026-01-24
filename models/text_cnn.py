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
        self.model_name = 'text_cnn'
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
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256         # 卷积核数量
        self.dropout = 0.1

class Model(nn.Module):
    def __init__(self, config, static_emb_matrix=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
            
        # TextCNN 结构
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        
        # === 兼容性初始化 ===
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def conv_and_pool(self, x, conv):
        # x: [batch_size, 1, seq_len, hidden_size]
        # conv(x): [batch_size, num_filters, seq_len-k+1, 1]
        x = F.relu(conv(x)).squeeze(3)
        # pool: [batch_size, num_filters, 1] -> squeeze: [batch_size, num_filters]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # [batch, seq_len]
        mask = x[2]     # [batch, seq_len]
        
        with autocast('cuda'):
            encoder_out = self.bert(context, attention_mask=mask)[0] # [batch, seq_len, 768]
            out = encoder_out.unsqueeze(1) # [batch, 1, seq_len, 768]
            
            # 多尺度卷积和池化
            out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # [batch, num_filters * 3]
            
            out = self.dropout(out)
            
            # === 保存中间结果 ===
            self.hidden_states = out.detach()
            
            # [关键修改] 生成伪造的注意力权重以适配 train_eval.py
            # TextCNN 无真实注意力，我们给所有非 Padding 的词分配相等的权重
            # mask: [batch, seq] -> float
            # 归一化使得和为1，模仿 softmax 的行为
            dummy_attn = mask.float()
            dummy_attn = dummy_attn / (dummy_attn.sum(dim=1, keepdim=True) + 1e-9)
            self.attention_weights = dummy_attn.detach()
            
            logits = self.fc(out)
            
            # === 保存概率分布 ===
            self.probabilities = F.softmax(logits, dim=1).detach()
            
        return logits