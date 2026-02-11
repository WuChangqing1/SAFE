import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import os

class Config(object):
    def __init__(self, dataset):
        self.model_name = 'base_bert'
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
        
        self.batch_size = 8  
        
        self.pad_size = 512  
        
        self.learning_rate = 2e-5
        self.bert_path = './pretrained/bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.1

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(
            config.bert_path, 
            attn_implementation="eager"
        )
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0]  
        mask = x[2]     
        
        outputs = self.bert(context, attention_mask=mask, output_attentions=True)
        
        pooler_output = outputs.pooler_output
        
        out = self.fc(pooler_output)
        
        self.probabilities = F.softmax(out, dim=1).detach()
        self.hidden_states = pooler_output.detach()
        
        last_layer_attn = outputs.attentions[-1]
        
        self.attention_weights = torch.mean(last_layer_attn, dim=1)[:, 0, :].detach()

        return out