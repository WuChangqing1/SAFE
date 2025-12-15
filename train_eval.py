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
                outputs = model(trains)
                cls_loss = F.cross_entropy(outputs, labels, weight=weights, label_smoothing=0.1)
                l2_lambda = 0.01
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                base_loss = cls_loss + l2_lambda * l2_norm
            
            model.zero_grad()
            scaler.scale(base_loss).backward()  # 缩放损失
            
            # 若集成FGM，需适配混合精度：
            # fgm.attack()
            # with autocast():
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
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, base_loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
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
    correct_samples = []  # 新增正确样本收集

    with torch.no_grad():
        for batch in data_iter:
            if test:
                (inputs, seq_len, mask), labels, contents = batch
            else:
                (inputs, seq_len, mask), labels = batch
            
            outputs = model((inputs, seq_len, mask))
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            
            # 获取中间结果
            attn_weights = model.attention_weights.cpu().numpy()
            probabilities = model.probabilities.cpu().numpy()
            hidden_states = model.hidden_states.cpu().numpy()
            
            labels_np = labels.cpu().numpy()
            predic = torch.max(outputs, 1)[1].cpu().numpy()

            labels_all = np.append(labels_all, labels_np)
            predict_all = np.append(predict_all, predic)

            if test:
                # 保存错误样本和正确样本的详细信息
                batch_size = inputs.size(0)
                for i in range(batch_size):
                    sample_data = {
                        'text': contents[i],
                        'true': labels_np[i],
                        'pred': predic[i],
                        'attention': attn_weights[i],
                        'prob': probabilities[i],
                        'hidden': hidden_states[i],
                        'input_ids': inputs[i].cpu().numpy(),
                        'mask': mask[i].cpu().numpy(),
                        'seq_len': seq_len[i].item()
                    }
                    if predic[i] != labels_np[i]:
                        wrong_samples.append(sample_data)
                    else:  # 新增正确样本保存
                        correct_samples.append(sample_data)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        return (acc, 
                loss_total/len(data_iter), 
                metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4),
                metrics.confusion_matrix(labels_all, predict_all),
                wrong_samples,
                correct_samples,
                labels_all,  # 新增返回
                predict_all  # 新增返回
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

            # Top注意力词汇
            valid_length = sample.get('seq_len', sum(sample['mask']))
            input_ids = sample['input_ids'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]

            attn_weights = sample['attention'][:len(tokens)]
            sorted_indices = np.argsort(-attn_weights)[:5]

            for i in range(5):
                if i < len(sorted_indices) and sorted_indices[i] < len(tokens):
                    row[f'top{i+1}_token'] = tokens[sorted_indices[i]]
                    row[f'top{i+1}_weight'] = f"{attn_weights[sorted_indices[i]]:.4f}"
                else:
                    row[f'top{i+1}_token'] = ''
                    row[f'top{i+1}_weight'] = ''

            writer.writerow(row)

    print(f"正确分类样本细节已保存到 {csv_path}")

def visualize_samples(samples, tokenizer, save_dir, prefix="correct", max_samples=20):
    
    for idx, sample in enumerate(samples[:max_samples]):
        try:
            # 注意力可视化
            valid_length = sample['seq_len']
            input_ids = sample['input_ids'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
            attn = sample['attention'][:len(tokens)]
        except Exception as e:
            print(f"可视化错误 {prefix}样本{idx}: {str(e)}")

                                    
                                      
def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # 修正解包变量数量，增加correct_samples
    test_acc, test_loss, test_report, test_confusion, wrong_samples, correct_samples, labels_all, predict_all = evaluate(config, model, test_iter, test=True)
    # test_acc, test_loss, test_report, test_confusion, wrong_samples, correct_samples, labels_all, predict_all = evaluate(
    #     config, model, test_iter, test=True, num_predictions=100)  # 测试时用100次预测
    
    
    # test_acc, test_loss, test_report, test_confusion, wrong_samples, correct_samples = evaluate(config, model, test_iter, test=True)
    
    tokenizer = config.tokenizer
    # analyze_misclassified(config, wrong_samples, tokenizer)
    # analyze_correctly_classified(config, correct_samples, tokenizer)  # 确保使用correct_samples变量

    # 确保这些变量在后续代码中可用
    if wrong_samples is not None:
        analyze_misclassified(config, wrong_samples, tokenizer)
    if correct_samples is not None:
        analyze_correctly_classified(config, correct_samples, tokenizer)

    # 保存正确样本分析结果
    save_dir = os.path.join(config.dataset, "saved_dict")
    save_correct_csv(correct_samples, config, tokenizer, save_dir)
    visualize_samples(correct_samples, tokenizer, save_dir, "correct")

    # 保存到CSV（新增详细中间结果）
    save_dir = os.path.join(config.dataset, "saved_dict")
    csv_path = os.path.join(save_dir, "misclassified_details.csv")
    
    # 定义CSV列头
    fieldnames = ['text', 'true_label', 'pred_label']
    fieldnames += [f'prob_{cls}' for cls in config.class_list]
    for i in range(1, 6):
        fieldnames += [f'top{i}_token', f'top{i}_weight']
    fieldnames += ['hidden_mean', 'hidden_std', 'hidden_max']

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
            
            # 新代码（修复索引越界）
            tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]

            # 获取实际有效的注意力权重（与过滤后的tokens对应）
            valid_indices = [i for i, t in enumerate(tokens) if i < sample['seq_len']]  # 有效位置
            valid_attention = sample['attention'][:sample['seq_len']][valid_indices]  # 对应权重

            if len(valid_attention) > 0:
                sorted_indices = np.argsort(-valid_attention)
                for i in range(5):
                    if i < len(sorted_indices) and sorted_indices[i] < len(tokens):
                        pos = sorted_indices[i]
                        row[f'top{i+1}_token'] = tokens[pos]
                        row[f'top{i+1}_weight'] = f"{valid_attention[pos]:.4f}"
                    else:
                        row[f'top{i+1}_token'] = ''
                        row[f'top{i+1}_weight'] = ''

            
            # 隐藏状态统计
            hidden = sample['hidden'][0]  # CLS向量
            row['hidden_mean'] = f"{np.mean(hidden):.4f}"
            row['hidden_std'] = f"{np.std(hidden):.4f}"
            row['hidden_max'] = f"{np.max(hidden):.4f}"

            writer.writerow(row)
    
    for idx, sample in enumerate(wrong_samples[:20]):  # 只可视化前20个样本
        try:
            # 注意力可视化

            valid_length = sample['seq_len']
            attn = sample['attention'][:valid_length]
            tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
            valid_indices = [i for i in range(min(len(tokens), valid_length))]
            tokens = [tokens[i] for i in valid_indices]
            attn = sample['attention'][:valid_length][valid_indices]
        except Exception as e:
            print(f"Error visualizing sample {idx}: {str(e)}")

    precision_macro = metrics.precision_score(labels_all, predict_all, average='macro')
    recall_macro = metrics.recall_score(labels_all, predict_all, average='macro')
    f1_macro = metrics.f1_score(labels_all, predict_all, average='macro')
    precision_weighted = metrics.precision_score(labels_all, predict_all, average='weighted')
    recall_weighted = metrics.recall_score(labels_all, predict_all, average='weighted')
    f1_weighted = metrics.f1_score(labels_all, predict_all, average='weighted')

    # 获取模型参数
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    num_epochs = config.num_epochs
    dropout = getattr(config, 'dropout', 0.1)  # 确保config有dropout参数

    # 打印结果矩阵
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

    # 原始输出保持精简
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Time usage:", get_time_dif(start_time)) 
    
    
    
# def train(config, model, train_iter, dev_iter, test_iter, train_data):
    
#     class_counts = np.array([175, 486, 201, 259, 37, 402, 346])  # 你实际的类别样本数替换
#     # class_counts = np.array([175, 486, 201, 259, 37, 402, 346])  # 你实际的类别样本数替换
#     weights = 1.0 / class_counts
#     weights = weights / weights.sum() * len(class_counts)
#     weights = torch.tensor(weights, dtype=torch.float).to(config.device)
    
#     start_time = time.time()
#     model.train()
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
#     optimizer = BertAdam(optimizer_grouped_parameters,
#                          lr=config.learning_rate,
#                          warmup=0.05,
#                          t_total=len(train_iter) * config.num_epochs)
    

#     total_batch = 0  # 记录进行到多少batch
#     dev_best_loss = float('inf')
#     last_improve = 0  # 记录上次验证集loss下降的batch数
#     flag = False  # 记录是否很久没有效果提升
#     model.train()
#     for epoch in range(config.num_epochs):
#         print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#         for i, (trains, labels) in enumerate(train_iter):
#             outputs = model(trains)
#             model.zero_grad()
#             # loss = F.cross_entropy(outputs, labels)
#             loss = F.cross_entropy(outputs, labels, weight=weights)
#             loss.backward()
#             optimizer.step()
#             if total_batch % 100 == 0:
#                 # 每多少轮输出在训练集和验证集上的效果
#                 true = labels.data.cpu()
#                 predic = torch.max(outputs.data, 1)[1].cpu()
#                 train_acc = metrics.accuracy_score(true, predic)
#                 dev_acc, dev_loss = evaluate(config, model, dev_iter)
#                 if dev_loss < dev_best_loss:
#                     dev_best_loss = dev_loss
#                     torch.save(model.state_dict(), config.save_path)
#                     improve = '*'
#                     last_improve = total_batch
#                 else:
#                     improve = ''
#                 time_dif = get_time_dif(start_time)
#                 msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
#                 print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
#                 model.train()
#             total_batch += 1
#             if total_batch - last_improve > config.require_improvement:
#                 # 验证集loss超过1000batch没下降，结束训练
#                 print("No optimization for a long time, auto-stopping...")
#                 flag = True
#                 break
#         if flag:
#             break
#     test(config, model, test_iter)

#     # 打印基础测试结果
#     msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
#     print(msg.format(test_loss, test_acc))
#     print("Precision, Recall and F1-Score...")
#     print(test_report)
#     print("Confusion Matrix...")
#     print(test_confusion)
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)
    
#     # 打印部分错误样本
#     print(f"\n误分类样本数: {len(wrong_samples)}")
#     for i, sample in enumerate(wrong_samples[:10]):  # 打印前10个错误
#         print(f"样本 {i+1}:")
#         print(f"文本: {sample['text']}")
#         print(f"真实标签: {config.class_list[sample['true']]}")
#         print(f"预测标签: {config.class_list[sample['pred']]}\n")

#     txt_path = os.path.join(save_dir, "misclassified.txt")
#     with open(txt_path, 'w', encoding='utf-8') as f:
#         for sample in wrong_samples:
#             f.write(f"文本: {sample['text']}\n真实: {config.class_list[sample['true']]}\n预测: {config.class_list[sample['pred']]}\n\n")
            
