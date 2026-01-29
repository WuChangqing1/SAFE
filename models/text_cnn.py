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
        self.num_epochs = 10
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 2e-5 
        self.bert_path = './pretrained/bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.dropout = 0.1

class Model(nn.Module):
    # 修改这里：删除了 static_emb_matrix=None
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
            
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]
        mask = x[2]
        
        with autocast('cuda'):
            encoder_out = self.bert(context, attention_mask=mask)[0]
            out = encoder_out.unsqueeze(1)
            out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
            out = self.dropout(out)
            
            self.hidden_states = out.detach()
            
            dummy_attn = mask.float()
            dummy_attn = dummy_attn / (dummy_attn.sum(dim=1, keepdim=True) + 1e-9)
            self.attention_weights = dummy_attn.detach()
            
            logits = self.fc(out)
            self.probabilities = F.softmax(logits, dim=1).detach()
            
        return logits