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
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
            
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)
        
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0]
        mask = x[2]
        
        with autocast('cuda'):
            embed = self.bert(context, attention_mask=mask)[0]
            out, _ = self.lstm(embed)
            out = torch.cat((embed, out), 2)
            out = F.relu(out)
            out = out.permute(0, 2, 1)
            pooled = F.max_pool1d(out, out.size(2)).squeeze(2)
            
            self.hidden_states = pooled.detach()
            
            dummy_attn = mask.float()
            dummy_attn = dummy_attn / (dummy_attn.sum(dim=1, keepdim=True) + 1e-9)
            self.attention_weights = dummy_attn.detach()
            
            logits = self.fc(pooled)
            self.probabilities = F.softmax(logits, dim=1).detach()
            
        return logits