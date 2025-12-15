# coding: UTF-8
# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
import time
from datetime import timedelta
from collections import defaultdict
import numpy as np

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask, content))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, return_contents=False):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.return_contents = return_contents  # 新增标

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        contents = [_[4] for _ in datas] if self.return_contents else None
        if self.return_contents:
            return (x, seq_len, mask), y, contents
        else:
            return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config, return_contents=False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, return_contents)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, weight=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, logits, targets):
#         ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss
#         return focal_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        batch_size = embeddings.size(0) // 2
        if batch_size == 0:
            return torch.tensor(0.0).to(embeddings.device)
        
        # 1. 创建主对角线矩阵（自身样本掩码）
        mask_self = torch.eye(batch_size * 2, dtype=torch.bool).to(embeddings.device)
        
        # 2. 创建移位batch_size的对角线矩阵（正样本对掩码）
        # 方法：创建单位矩阵后，上下移位batch_size行
        mask_positive = torch.eye(batch_size * 2, dtype=torch.bool).to(embeddings.device)
        mask_positive = torch.roll(mask_positive, shifts=batch_size, dims=0)  # 关键修正：用roll实现移位
        
        # 3. 合并掩码（排除自身和正样本对，只保留负样本）
        mask = mask_self | mask_positive
        mask = ~mask
        
        # 后续逻辑保持不变
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / self.temperature
        exp_sim = torch.exp(sim_matrix) * (~mask)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        pos_pairs = torch.cat([torch.arange(batch_size, device=embeddings.device), 
                              torch.arange(batch_size, device=embeddings.device) + batch_size])
        pos_sim = log_prob[pos_pairs, pos_pairs - batch_size if batch_size > 0 else pos_pairs]
        return -pos_sim.mean()
    
def load_static_embedding(emb_path, tokenizer, embed_dim=300):
    """加载静态词向量（兼容混合精度，返回float32张量）"""
    emb_dict = defaultdict(lambda: np.random.normal(0, 0.01, embed_dim).astype(np.float32))
    with open(emb_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip().split()
            if len(line) != embed_dim + 1:
                continue
            word = line[0]
            vec = np.array([float(x) for x in line[1:]], dtype=np.float32)
            emb_dict[word] = vec
    
    vocab_size = tokenizer.vocab_size
    emb_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for token, idx in tokenizer.vocab.items():
        emb_matrix[idx] = emb_dict[token]
    
    return torch.tensor(emb_matrix, dtype=torch.float32)  # 保持float32，兼容混合精度