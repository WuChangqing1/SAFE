import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
# from pytorch_pretrained.optimization import BertAdam
from utils import build_dataset, build_iterator, get_time_dif
import csv
import os
import matplotlib.pyplot as plt
# from pytorch_pretrained import BertTokenizer  # 确保使用兼容版本的tokenizer
from transformers import BertTokenizer
# from transformers import AdamW
from utils import FocalLoss

from torch.amp import autocast

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


# ==========================================
# R-Drop 核心辅助函数: 计算KL散度
# ==========================================
def compute_kl_loss(p, q):
    """
    计算双向KL散度
    p, q: 模型两次前向传播的 logits
    """
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # 维度处理：[batch, classes] -> [batch] -> scalar
    p_loss = p_loss.sum(dim=1)
    q_loss = q_loss.sum(dim=1)
    
    # 取平均
    loss = (p_loss.mean() + q_loss.mean()) / 2
    return loss


# 2222222222222222222222222222222222222222222222
from transformers import get_linear_schedule_with_warmup
# from transformers.optimization import AdamW
from torch.optim import AdamW

def train(config, model, train_iter, dev_iter, test_iter, train_data):
    # 原有初始化代码不变
    class_counts = np.array([175, 486, 201, 259, 37, 402, 346])
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.sum() * len(class_counts)
    weights = torch.tensor(weights, dtype=torch.float).to(config.device)
    config.weights = weights
    
    # R-Drop 超参数
    rdrop_alpha = 4.0  # 控制KL散度损失的权重，通常取 4 或 5

    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=0.01, eps=1e-8)
    total_steps = len(train_iter) * config.num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    scaler = config.scaler  # 获取梯度缩放器
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            with autocast('cuda'):  # 开启混合精度
                # ================= R-Drop 修改部分开始 =================
                # 1. 第一次前向传播
                outputs1 = model(trains)
                # 2. 第二次前向传播 (由于Dropout的存在，结果会略有不同)
                outputs2 = model(trains)
                
                # 3. 计算两次的交叉熵损失 (Cross Entropy)
                ce_loss1 = F.cross_entropy(outputs1, labels, weight=weights, label_smoothing=0.1)
                ce_loss2 = F.cross_entropy(outputs2, labels, weight=weights, label_smoothing=0.1)
                ce_loss = 0.5 * (ce_loss1 + ce_loss2)
                
                # 4. 计算KL散度损失 (KL Divergence)
                kl_loss = compute_kl_loss(outputs1, outputs2)
                
                # 5. 原有的L2正则化
                l2_lambda = 0.01
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                
                # 6. 总损失 = CE + alpha * KL + L2
                base_loss = ce_loss + rdrop_alpha * kl_loss + l2_lambda * l2_norm
                
                # 为了日志打印方便，output仍使用第一次的
                outputs = outputs1 
                # ================= R-Drop 修改部分结束 =================

            model.zero_grad()
            scaler.scale(base_loss).backward()  # 缩放损失
            
            # 若集成FGM，需适配混合精度：
            # fgm.attack()
            # with autocast():
            #     # FGM对抗训练通常不需要做两次Forward，只做一次即可
            #     outputs_adv = model(trains)
            #     adv_loss = F.cross_entropy(outputs_adv, labels, weight=weights, label_smoothing=0.1)
            # scaler.scale(adv_loss).backward()
            # fgm.restore()
            
            scaler.unscale_(optimizer)  # 恢复梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)  # 缩放优化器步骤
            scaler.update()  # 更新缩放器
            scheduler.step()
            
            # 保留原日志打印、早停逻辑
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    save_dir = os.path.dirname(config.save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                # 打印 Loss 时，base_loss 包含了 KL Loss，这有助于观察 R-Drop 是否生效
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, base_loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)
    
    
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    wrong_samples = [] 
    correct_samples = [] 

    with torch.no_grad():
        for batch in data_iter:
            if test:
                (inputs, seq_len, mask), labels, contents = batch
            else:
                (inputs, seq_len, mask), labels = batch
            
            outputs = model((inputs, seq_len, mask))
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            
            # --- 修改开始：兼容性处理 ---
            # 1. 安全获取 Attention 权重 (TextCNN/RCNN 可能没有)
            attn_weights = getattr(model, 'attention_weights', None)
            if attn_weights is not None:
                attn_weights = attn_weights.cpu().numpy()
            
            # 2. 安全获取概率分布 (如果没有保存，则现场计算)
            if hasattr(model, 'probabilities') and model.probabilities is not None:
                probabilities = model.probabilities.cpu().numpy()
            else:
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            
            # 3. 安全获取 Hidden States
            hidden_states = getattr(model, 'hidden_states', None)
            if hidden_states is not None:
                hidden_states = hidden_states.cpu().numpy()
            # --- 修改结束 ---
            
            labels_np = labels.cpu().numpy()
            predic = torch.max(outputs, 1)[1].cpu().numpy()

            labels_all = np.append(labels_all, labels_np)
            predict_all = np.append(predict_all, predic)

            if test:
                batch_size = inputs.size(0)
                for i in range(batch_size):
                    sample_data = {
                        'text': contents[i],
                        'true': labels_np[i],
                        'pred': predic[i],
                        'prob': probabilities[i],
                        'input_ids': inputs[i].cpu().numpy(),
                        'mask': mask[i].cpu().numpy(),
                        'seq_len': seq_len[i].item(),
                        # 只有存在时才保存
                        'attention': attn_weights[i] if attn_weights is not None else None,
                        'hidden': hidden_states[i] if hidden_states is not None else None
                    }
                    if predic[i] != labels_np[i]:
                        wrong_samples.append(sample_data)
                    else:
                        correct_samples.append(sample_data)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        return (acc, 
                loss_total/len(data_iter), 
                metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4),
                metrics.confusion_matrix(labels_all, predict_all),
                wrong_samples,
                correct_samples,
                labels_all, 
                predict_all
               )
    else:
        return acc, loss_total/len(data_iter)

def analyze_correctly_classified(config, correct_samples, tokenizer, max_samples=0):
    print("\n分析正确分类样本的关键特征...")
    
    for idx, sample in enumerate(correct_samples[:max_samples]):
        print(f"\n样本 {idx+1}/{len(correct_samples)}")
        print(f"文本：{sample['text'][:100]}...")
        print(f"真实标签：{config.class_list[sample['true']]}")
        
        try:
            valid_length = sample.get('seq_len', sum(sample['mask']))
            input_ids = sample['input_ids'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
            
            if not tokens:
                print("有效token序列为空")
                continue

            if sample['attention'] is None:
                # print("该模型无注意力权重，跳过注意力分析") # 可选打印
                pass
            else:
                attn_weights = sample['attention'][:len(tokens)]
                sorted_indices = np.argsort(-attn_weights)[:5]
            
            print("\n关键注意力位置：")
            for i, pos in enumerate(sorted_indices):
                if pos < len(tokens):
                    print(f"Top{i+1}: [{tokens[pos]}] ({pos}位) 权重：{attn_weights[pos]:.4f}")

        except Exception as e:
            print(f"分析出错：{str(e)}")
        print("="*80)    
    
def analyze_misclassified(config, wrong_samples, tokenizer, max_samples=0):
    """
    增强安全性的误分类样本分析函数
    """
    print("\n正在分析误分类样本的中间结果...")
    
    for idx, sample in enumerate(wrong_samples[:max_samples]):
        # 基础信息展示
        print(f"\n样本 {idx+1}/{len(wrong_samples)}")
        print(f"文本：{sample['text'][:100]}...")  # 截断长文本
        print(f"真实标签：{config.class_list[sample['true']]}")
        print(f"预测标签：{config.class_list[sample['pred']]}")
        
        # ========== 安全获取token序列 ==========
        try:
            # 使用预存的有效序列长度
            valid_length = sample.get('seq_len', sum(sample['mask']))
            input_ids = sample['input_ids'][:valid_length]  # 截断有效部分
            
            # 转换token并过滤特殊符号
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
            
            # 空序列保护
            if not tokens:
                print("警告：有效token序列为空，跳过分析")
                continue
        except Exception as e:
            print(f"Token解析失败：{str(e)}")
            continue

        try:
            if sample['attention'] is None:
                # print("该模型无注意力权重，跳过注意力分析") # 可选打印
                pass
            else:
                # 对齐注意力权重与token序列
                attn_weights = sample['attention'][:len(tokens)]
                
                # 获取top注意力位置
                sorted_indices = np.argsort(-attn_weights)[:5]  # 取前5个
        except Exception as e:
            print(f"注意力处理失败：{str(e)}")
            continue

        # ========== 概率分布展示 ==========
        print("\n预测概率分布：")
        for cls_idx, prob in enumerate(sample['prob']):
            print(f"{config.class_list[cls_idx]}: {prob:.4f}")

        # ========== 注意力分析 ==========
        print("\n注意力权重分析：")
        for i, pos in enumerate(sorted_indices):
            # 边界检查双重保险
            if pos >= len(tokens):
                print(f"位置{pos}超出token序列范围（总长度{len(tokens)}）")
                continue
                
            token = tokens[pos]
            weight = attn_weights[pos]
            print(f"Top{i+1}: [{token}] ({pos}位) 权重：{weight:.4f}")

        # ========== 隐藏状态分析 ==========
        try:
            hidden = sample['hidden'][0]  # CLS向量
            print("\n隐藏状态分析：")
            print(f"均值：{np.mean(hidden):.4f}")
            print(f"标准差：{np.std(hidden):.4f}")
            print(f"最大值：{np.max(hidden):.4f}")
        except Exception as e:
            print(f"隐藏状态分析失败：{str(e)}")

        print("="*80)


def save_correct_csv(samples, config, tokenizer, save_dir):
    csv_path = os.path.join(save_dir, "correct_details.csv")
    fieldnames = ['text', 'true_label'] + [f'prob_{cls}' for cls in config.class_list] + \
                 [f'top{i}_token' for i in range(1, 6)] + [f'top{i}_weight' for i in range(1, 6)]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            row = {
                'text': sample['text'],
                'true_label': config.class_list[sample['true']],
            }
            # 概率分布
            for i, prob in enumerate(sample['prob']):
                row[f'prob_{config.class_list[i]}'] = f"{prob:.4f}"

            # Top注意力词汇处理
            valid_length = sample.get('seq_len', sum(sample['mask']))
            input_ids = sample['input_ids'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]

            # === 修改开始：安全处理注意力权重 ===
            sorted_indices = [] # 预先初始化为空列表
            attn_weights = None
            
            # 只有当 attention 存在且不为 None 时才进行计算
            if sample.get('attention') is not None:
                attn_weights = sample['attention'][:len(tokens)]
                if len(attn_weights) > 0:
                    sorted_indices = np.argsort(-attn_weights)[:5]

            for i in range(5):
                # 增加了对 attn_weights is not None 的判断
                if i < len(sorted_indices) and sorted_indices[i] < len(tokens) and attn_weights is not None:
                    row[f'top{i+1}_token'] = tokens[sorted_indices[i]]
                    row[f'top{i+1}_weight'] = f"{attn_weights[sorted_indices[i]]:.4f}"
                else:
                    row[f'top{i+1}_token'] = ''
                    row[f'top{i+1}_weight'] = ''
            # === 修改结束 ===

            writer.writerow(row)

    print(f"正确分类样本细节已保存到 {csv_path}")

def visualize_samples(samples, tokenizer, save_dir, prefix="correct", max_samples=20):
    """
    可视化样本的注意力权重并保存为图片
    """
    vis_dir = os.path.join(save_dir, f"{prefix}_vis")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)

    print(f"\n正在生成 {prefix} 样本的注意力可视化图 (前 {max_samples} 个)...")
    
    for idx, sample in enumerate(samples[:max_samples]):
        try:
            # 数据准备
            valid_length = sample['seq_len']
            # 截取有效长度的 input_ids
            input_ids = sample['input_ids'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # 过滤特殊 token 用于展示
            display_tokens = []
            display_weights = []
            
            # 获取注意力权重 (如果存在)
            if sample.get('attention') is not None:
                attn = sample['attention'][:valid_length]
                
                # 对齐 token 和权重
                for t, w in zip(tokens, attn):
                    if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                        display_tokens.append(t)
                        display_weights.append(w)
            else:
                continue # 无注意力权重则跳过

            if not display_tokens:
                continue

            # 绘图
            plt.figure(figsize=(12, 6))
            x_pos = np.arange(len(display_tokens))
            plt.bar(x_pos, display_weights, align='center', alpha=0.7)
            plt.xticks(x_pos, display_tokens, rotation=45, ha='right', fontproperties='SimHei') # 需确保系统有中文字体，否则显示方框
            plt.ylabel('Attention Weight')
            plt.title(f"True: {sample['true_label']} | Pred: {sample.get('pred_label', sample['true_label'])}")
            
            # 保存
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"sample_{idx}.png"))
            plt.close()

        except Exception as e:
            # 忽略绘图错误，避免中断流程
            # print(f"可视化错误 {prefix}样本{idx}: {str(e)}")
            plt.close()
                                      
def test(config, model, test_iter):
    # 加载最优模型
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    
    # 评估
    test_acc, test_loss, test_report, test_confusion, wrong_samples, correct_samples, labels_all, predict_all = evaluate(config, model, test_iter, test=True)
    
    tokenizer = config.tokenizer
    
    # --- 1. 控制台输出分析 ---
    if wrong_samples:
        analyze_misclassified(config, wrong_samples, tokenizer, max_samples=5)
    if correct_samples:
        analyze_correctly_classified(config, correct_samples, tokenizer, max_samples=3)

    # --- 2. 保存详细数据到 CSV ---
    save_dir = os.path.dirname(config.save_path)
    
    # 保存正确样本详情
    if correct_samples:
        save_correct_csv(correct_samples, config, tokenizer, save_dir)
        # 可视化部分正确样本
        visualize_samples(correct_samples, tokenizer, save_dir, "correct", max_samples=10)

    # 保存错误样本详情 (包含所有中间层信息)
    if wrong_samples:
        csv_path = os.path.join(save_dir, "misclassified_details.csv")
        fieldnames = ['text', 'true_label', 'pred_label'] + \
                     [f'prob_{cls}' for cls in config.class_list] + \
                     ['top1_token', 'top1_weight', 'top2_token', 'top2_weight', 'top3_token', 'top3_weight'] + \
                     ['hidden_mean', 'hidden_std']

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for sample in wrong_samples:
                # 基础信息
                row = {
                    'text': sample['text'],
                    'true_label': config.class_list[sample['true']],
                    'pred_label': config.class_list[sample['pred']]
                }
                
                # 概率分布
                for i, prob in enumerate(sample['prob']):
                    row[f'prob_{config.class_list[i]}'] = f"{prob:.4f}"
                
                # Token 和 Attention 处理
                try:
                    tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
                    # 过滤特殊字符
                    valid_tokens = []
                    valid_weights = []
                    
                    if sample.get('attention') is not None:
                        raw_weights = sample['attention'][:sample['seq_len']]
                        for idx, t in enumerate(tokens[:sample['seq_len']]):
                            if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                                valid_tokens.append(t)
                                valid_weights.append(raw_weights[idx])
                        
                        # 排序取 Top 3
                        if valid_weights:
                            sorted_indices = np.argsort(-np.array(valid_weights))
                            for i in range(min(3, len(sorted_indices))):
                                pos = sorted_indices[i]
                                row[f'top{i+1}_token'] = valid_tokens[pos]
                                row[f'top{i+1}_weight'] = f"{valid_weights[pos]:.4f}"
                except Exception:
                    pass
                
                # 隐藏层统计
                if sample.get('hidden') is not None:
                    hidden = sample['hidden'][0] # CLS
                    row['hidden_mean'] = f"{np.mean(hidden):.4f}"
                    row['hidden_std'] = f"{np.std(hidden):.4f}"

                writer.writerow(row)
        
        print(f"误分类样本详情已保存到 {csv_path}")
        # 可视化部分错误样本
        # 修正：传递正确的 pred_label 供 visualize_samples 使用
        for s in wrong_samples:
            s['true_label'] = config.class_list[s['true']]
            s['pred_label'] = config.class_list[s['pred']]
        visualize_samples(wrong_samples, tokenizer, save_dir, "wrong", max_samples=10)

    # --- 3. 打印最终测试指标 ---
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)