import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import FGM, get_time_dif
from utils import build_dataset, build_iterator, get_time_dif
import csv
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils import FocalLoss
from torch.amp import autocast

# 权重初始化
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

def train(config, model, train_iter, dev_iter, test_iter, train_data):
    # 1. 初始化显存统计 (用于论文显存分析)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    class_counts = np.array([175, 486, 201, 259, 37, 402, 346])
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / weights.sum() * len(class_counts)
    weights = torch.tensor(weights, dtype=torch.float).to(config.device)
    config.weights = weights
    
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

    # FGM对抗训练初始化
    fgm = FGM(model)
    # 【新增】获取动态 epsilon，默认为 1.0 (兼容普通训练和敏感性分析)
    epsilon_val = getattr(config, 'fgm_epsilon', 1.0)
    print(f"当前 FGM 对抗扰动幅度 epsilon: {epsilon_val}")

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            with autocast('cuda'):  # 开启混合精度
                outputs = model(trains)
                cls_loss = F.cross_entropy(outputs, labels, weight=weights, label_smoothing=0.1)
                l2_lambda = 0.01
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                base_loss = cls_loss + l2_lambda * l2_norm
            
            model.zero_grad()
            scaler.scale(base_loss).backward()  # 缩放损失
            
            # 2. FGM 对抗训练逻辑
            # 【修改】使用动态 epsilon
            fgm.attack(epsilon=epsilon_val, emb_name='word_embeddings') 
            
            with autocast('cuda'): # 对抗样本的前向传播也要用混合精度
                outputs_adv = model(trains)
                loss_adv = F.cross_entropy(outputs_adv, labels, weight=weights, label_smoothing=0.1)
            
            scaler.scale(loss_adv).backward() # 累加对抗梯度
            
            fgm.restore(emb_name='word_embeddings') # 恢复原始 Embedding
            
            scaler.unscale_(optimizer)  # 恢复梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)  # 缩放优化器步骤
            scaler.update()  # 更新缩放器
            scheduler.step()
            
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
                
                # 【新增】显存监控逻辑
                mem_info = ""
                if torch.cuda.is_available():
                    # 获取真实的峰值显存 (MB)
                    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                    mem_info = f" | Max Mem: {max_mem:.0f}MB"

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6} {7}'
                print(msg.format(total_batch, base_loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve, mem_info))
                model.train()
            
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    
    # 【修改】train 函数现在返回 test 的结果，方便外部脚本收集数据
    return test(config, model, test_iter)


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
            
            # --- 兼容性处理 ---
            attn_weights = getattr(model, 'attention_weights', None)
            if attn_weights is not None:
                attn_weights = attn_weights.cpu().numpy()
            
            if hasattr(model, 'probabilities') and model.probabilities is not None:
                probabilities = model.probabilities.cpu().numpy()
            else:
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            
            hidden_states = getattr(model, 'hidden_states', None)
            if hidden_states is not None:
                hidden_states = hidden_states.cpu().numpy()
            # -----------------
            
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
            if sample['attention'] is not None:
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
    print("\n正在分析误分类样本的中间结果...")
    for idx, sample in enumerate(wrong_samples[:max_samples]):
        print(f"\n样本 {idx+1}/{len(wrong_samples)}")
        print(f"文本：{sample['text'][:100]}...")
        print(f"真实标签：{config.class_list[sample['true']]}")
        print(f"预测标签：{config.class_list[sample['pred']]}")
        try:
            valid_length = sample.get('seq_len', sum(sample['mask']))
            input_ids = sample['input_ids'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
            if not tokens: continue
        except Exception as e: continue
        
        # 概率分布
        print("\n预测概率分布：")
        for cls_idx, prob in enumerate(sample['prob']):
            print(f"{config.class_list[cls_idx]}: {prob:.4f}")

        # 注意力分析
        if sample['attention'] is not None:
            try:
                attn_weights = sample['attention'][:len(tokens)]
                sorted_indices = np.argsort(-attn_weights)[:5]
                print("\n注意力权重分析：")
                for i, pos in enumerate(sorted_indices):
                    if pos >= len(tokens): continue
                    print(f"Top{i+1}: [{tokens[pos]}] ({pos}位) 权重：{attn_weights[pos]:.4f}")
            except Exception as e: pass
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
            for i, prob in enumerate(sample['prob']):
                row[f'prob_{config.class_list[i]}'] = f"{prob:.4f}"
            
            valid_length = sample.get('seq_len', sum(sample['mask']))
            input_ids = sample['input_ids'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]

            sorted_indices = []
            attn_weights = None
            if sample.get('attention') is not None:
                attn_weights = sample['attention'][:len(tokens)]
                if len(attn_weights) > 0:
                    sorted_indices = np.argsort(-attn_weights)[:5]

            for i in range(5):
                if i < len(sorted_indices) and sorted_indices[i] < len(tokens) and attn_weights is not None:
                    row[f'top{i+1}_token'] = tokens[sorted_indices[i]]
                    row[f'top{i+1}_weight'] = f"{attn_weights[sorted_indices[i]]:.4f}"
                else:
                    row[f'top{i+1}_token'] = ''
                    row[f'top{i+1}_weight'] = ''
            writer.writerow(row)
    print(f"正确分类样本细节已保存到 {csv_path}")

def visualize_samples(samples, tokenizer, save_dir, prefix="correct", max_samples=20):
    # 这里保留你的可视化逻辑占位，或者你可以完善它
    pass

def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, wrong_samples, correct_samples, labels_all, predict_all = evaluate(config, model, test_iter, test=True)
    
    tokenizer = config.tokenizer
    if wrong_samples is not None:
        analyze_misclassified(config, wrong_samples, tokenizer, max_samples=5) # 限制打印数量防止刷屏
    if correct_samples is not None:
        # analyze_correctly_classified(config, correct_samples, tokenizer, max_samples=5)
        pass

    # 保存正确样本分析结果
    save_dir = os.path.join(config.dataset, "saved_dict")
    save_correct_csv(correct_samples, config, tokenizer, save_dir)
    # visualize_samples(correct_samples, tokenizer, save_dir, "correct")

    # 保存误分类到CSV
    csv_path = os.path.join(save_dir, "misclassified_details.csv")
    fieldnames = ['text', 'true_label', 'pred_label'] + [f'prob_{cls}' for cls in config.class_list]
    for i in range(1, 6): fieldnames += [f'top{i}_token', f'top{i}_weight']
    fieldnames += ['hidden_mean', 'hidden_std', 'hidden_max']

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample in wrong_samples:
            row = {
                'text': sample['text'],
                'true_label': config.class_list[sample['true']],
                'pred_label': config.class_list[sample['pred']]
            }
            for i, prob in enumerate(sample['prob']):
                row[f'prob_{config.class_list[i]}'] = f"{prob:.4f}"
            
            tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
            valid_indices = [i for i, t in enumerate(tokens) if i < sample['seq_len']]
            
            if sample['attention'] is not None and len(valid_indices) > 0:
                valid_attention = sample['attention'][:sample['seq_len']][valid_indices]
                if len(valid_attention) > 0:
                    sorted_indices = np.argsort(-valid_attention)
                    for i in range(5):
                        if i < len(sorted_indices) and sorted_indices[i] < len(tokens):
                            pos = sorted_indices[i]
                            row[f'top{i+1}_token'] = tokens[pos]
                            row[f'top{i+1}_weight'] = f"{valid_attention[pos]:.4f}"
                        else:
                            row[f'top{i+1}_token'] = ''; row[f'top{i+1}_weight'] = ''
            
            if sample['hidden'] is not None:
                hidden = sample['hidden'][0]
                row['hidden_mean'] = f"{np.mean(hidden):.4f}"
                row['hidden_std'] = f"{np.std(hidden):.4f}"
                row['hidden_max'] = f"{np.max(hidden):.4f}"
            writer.writerow(row)

    precision_macro = metrics.precision_score(labels_all, predict_all, average='macro')
    recall_macro = metrics.recall_score(labels_all, predict_all, average='macro')
    f1_macro = metrics.f1_score(labels_all, predict_all, average='macro')
    precision_weighted = metrics.precision_score(labels_all, predict_all, average='weighted')
    recall_weighted = metrics.recall_score(labels_all, predict_all, average='weighted')
    f1_weighted = metrics.f1_score(labels_all, predict_all, average='weighted')

    batch_size = config.batch_size
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    dropout = getattr(config, 'dropout', 0.1)

    print("\nFinal Result Matrix:")
    print("+------------------+----------------+")
    print("| Parameter        | Value          |")
    print("+------------------+----------------+")
    print(f"| Batch Size       | {batch_size:<14} |")
    print(f"| Learning Rate    | {learning_rate:<14.5f} |")
    print(f"| Epochs           | {num_epochs:<14} |")
    print(f"| Dropout          | {dropout:<14.4f} |")
    print("+------------------+----------------+")
    print("| Metric           | Value          |")
    print("+------------------+----------------+")
    print(f"| Accuracy         | {test_acc:<14.4f} |")
    print(f"| Precision (Macro)| {precision_macro:<14.4f} |")
    print(f"| Recall (Macro)   | {recall_macro:<14.4f} |")
    print(f"| F1 (Macro)       | {f1_macro:<14.4f} |")
    print(f"| Precision (Weight)| {precision_weighted:<14.4f} |")
    print(f"| Recall (Weight)  | {recall_weighted:<14.4f} |")
    print(f"| F1 (Weight)      | {f1_weighted:<14.4f} |")
    print("+------------------+----------------+")

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Time usage:", get_time_dif(start_time))
    
    # 【修改】返回结果以便外部脚本画图
    return test_acc, f1_macro