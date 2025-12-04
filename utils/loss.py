import torch
import torch.nn as nn
import torch.nn.functional as F

class LGPNLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, sample_representations, label_representations, labels):
        """
        计算LGPN损失
        
        Args:
            sample_representations: 样本表示 [batch_size, hidden_size]
            label_representations: 标签表示 [num_classes, hidden_size] 
            labels: 样本标签 [batch_size]
        """
        batch_size = sample_representations.size(0)
        num_classes = label_representations.size(0)
        
        # 计算样本与所有标签的相似度
        similarities = torch.matmul(sample_representations, label_representations.t()) / self.temperature
        
        # 创建目标矩阵
        targets = F.one_hot(labels, num_classes).float()
        
        # 计算交叉熵损失
        loss = F.cross_entropy(similarities, labels)
        
        return loss

class LabelSeparationLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, label_representations):
        """
        确保标签表示之间的区分性
        """
        num_classes = label_representations.size(0)
        
        # 计算标签之间的相似度
        label_similarities = torch.matmul(label_representations, label_representations.t()) / self.temperature
        
        # 创建目标：对角线为1，其他为0
        targets = torch.eye(num_classes, device=label_representations.device)
        
        # 使用交叉熵让每个标签与其他标签区分开
        loss = F.cross_entropy(label_similarities, torch.arange(num_classes, device=label_representations.device))
        
        return loss