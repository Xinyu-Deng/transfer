import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score
from model_node import Specformer,SpecLayer
from torch_geometric.nn import GCNConv
from utils import count_parameters, init_params, seed_everything, get_split
from torch_geometric.utils import add_self_loops, remove_self_loops, degree

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

def few_shot_validate(model, e,u,x,y, label_embs,n_way, k_shot=1, device='cuda'):
    """
    对图模型进行 1-shot 验证，只训练 SpecLayer。
    Args:
        model: 图模型，包含 .speclayer 属性
        data: 包含 x, y, train_mask, val_mask 的数据对象
        n_way: 类别数
        k_shot: 每类样本数
    """
    model.eval()
    nclass = n_way

    # ====== Step 1. 构造 1-shot 样本 ======
    train_idx = []
    for c in range(n_way):
        cls_idx = (y == c).nonzero(as_tuple=True)[0]
        if len(cls_idx) >= k_shot:
            sampled = cls_idx[torch.randperm(len(cls_idx))[:k_shot]]
            train_idx.append(sampled)
    train_idx = torch.cat(train_idx)

    # ====== Step 2. 冻结其他模块 ======
    for name, param in model.named_parameters():
        # if 'layers' or 'linear_encoder' in name.lower() or 'classify' in name.lower() in name.lower():
        #     param.requires_grad = True
        if 'layers' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    #         print("🔁 Reinitializing SpecLayer parameters ...")
    #     for name, module in model.named_modules():
    #         if isinstance(module, SpecLayer):
    #             module.weight.data.fill_(1.0)
    #     print("✅ SpecLayer re-initialized.")

    # ====== Step 3. 仅优化 SpecLayer ======
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0)

    # ====== Step 4. few-shot 训练 ======
    model.train()
    for epoch in range(50):  # 通常 few-shot 不需要太多 epoch
        optimizer.zero_grad()
        logits = model(e,u,x)
        similarity = torch.mm(logits, label_embs.t())/0.07
        loss = F.cross_entropy(similarity[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"[Few-Shot Epoch {epoch}] Loss: {loss.item():.4f}")

    # ====== Step 5. 验证 ======
    model.eval()
    logits = model(e,u,x)
    similarity = torch.mm(logits, label_embs.t())/0.07
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
    acc = evaluation(similarity.cpu(), y.cpu()).item()

    print(f"✅ {k_shot}-shot Validation Accuracy: {acc:.4f}")
    return acc

class MLP5Shot(nn.Module):
    def __init__(self, in_dim, hidden_dim, nclass):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass)
        )

    def forward(self, x):
        return self.mlp(x)

class GCN5Shot(nn.Module):
    """
    简单 2-layer GCN，用于 few-shot baseline。
    forward 输入: x [N, F], edge_index [2, E]
    返回: logits 或 embedding （取决于最后一层是否输出 nclass）
    """
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5, use_bn=False):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.use_bn:
            self.bn1.reset_parameters()

    def forward(self, x, edge_index):
        # ensure self-loops (GCNConv usually benefits from self-loops)
        # (torch_geometric GCNConv handles self-loops internally in many versions,
        #  but to be safe we'll add them)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # 如果 out_dim == nclass 则为 logits；否则可以认为是 embedding
def normalize_adj(edge_index, num_nodes):
    """
    计算标准化邻接矩阵 A_hat = D^{-1/2} (A + I) D^{-1/2}
    返回: edge_index, norm
    """
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return edge_index, norm

def propagate_once(x, edge_index, norm):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.index_add_(0, row, norm.unsqueeze(1) * x[col])
    return out

# ========== 线性 SGC/HGC 模型 ==========
class LinearSGC1(nn.Module):
    def forward(self, x, edge_index):
        edge_index, norm = normalize_adj(edge_index, x.size(0))
        return propagate_once(x, edge_index, norm)

class LinearSGC2(nn.Module):
    def forward(self, x, edge_index):
        edge_index, norm = normalize_adj(edge_index, x.size(0))
        x1 = propagate_once(x, edge_index, norm)
        x2 = propagate_once(x1, edge_index, norm)
        return x2

class LinearHGC1(nn.Module):
    def forward(self, x, edge_index):
        edge_index, norm = normalize_adj(edge_index, x.size(0))
        low = propagate_once(x, edge_index, norm)
        return x - low  # 高通

class LinearHGC2(nn.Module):
    def forward(self, x, edge_index):
        edge_index, norm = normalize_adj(edge_index, x.size(0))
        low = propagate_once(x, edge_index, norm)
        low2 = propagate_once(low, edge_index, norm)
        return x - low2  # 二阶高通
def evaluate_linear_model(x, edge_index, y, label_embs, nclass):
    """
    通用评估函数：使用线性层对 encoder 生成的嵌入做分类。
    """

    models = {
        'SGC1': LinearSGC1(),
        'SGC2': LinearSGC2(),
        'HGC1': LinearHGC1(),
        'HGC2': LinearHGC2(),
    }

    results = {}
    for name, model in models.items():
        evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
        logits = torch.mm(x, label_embs.t())/0.07
        logits = model(logits, edge_index)  # [N, F']
        # similarity = torch.mm(logits, label_embs.t())/0.07
        acc = evaluation(logits.cpu(), y.cpu()).item()
        results[name] = acc

    print("\n📊 Summary:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return 


# ========== few-shot 训练/验证函数 ==========
def few_shot_gcn_baseline(x, label_embs,edge_index, y, train_mask, val_mask, nclass,
                          k_shot=5, epochs=200, lr=1e-3,
                          weight_decay=0.0, device='cuda'):
    """
    用 5-shot（或 k-shot）训练 GCN 的 baseline。
    参数:
        x: [N, F] 特征 tensor (CPU or CUDA)
        edge_index: [2, E] long tensor
        y: [N] long tensor (labels 0..C-1)
        train_mask, val_mask: boolean or index tensors indicating train/val sets (仅用于选择验证集)
        nclass: 类别数
        k_shot: 每类采样 k 个样本做 few-shot 训练
        epochs, lr, hidden_dim, weight_decay, device: 训练超参
    返回:
        val_acc 在 val_mask 上的准确率
    """
    

    # 初始化模型（输出维度为 nclass -> 直接输出 logits）
    logits = torch.mm(x, label_embs.t())/0.07  # 预计算相似度矩阵作为输入
    model = GCN5Shot(in_dim=logits.size(1), hidden_dim=logits.size(1), out_dim=logits.size(1), dropout=0.5, use_bn=True).cuda()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ====== few-shot 采样 ======
    train_idx = []
    for c in range(nclass):
        cls_idx = (y == c).nonzero(as_tuple=True)[0]
        if len(cls_idx) >= k_shot:
            sampled = cls_idx[torch.randperm(len(cls_idx))[:k_shot]]
            train_idx.append(sampled)
    train_idx = torch.cat(train_idx)
    val_idx = val_mask.nonzero(as_tuple=True)[0].cuda()

    # ========== 训练 ==========
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(logits, edge_index)  # [N, nclass]
        loss = criterion(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % max(1, epochs // 10) == 0:
            print(f"[GCN Epoch {epoch}/{epochs}] Loss: {loss.item():.4f}")

    # ========== 验证 ==========
    model.eval()
    with torch.no_grad():
        out = model(logits, edge_index)
        pred = out[val_idx].argmax(dim=1)
        acc = (pred == y[val_idx]).float().mean().item()
    print(f"🏁 GCN {k_shot}-shot Validation Acc: {acc:.4f}")
    return acc, model



def few_shot_mlp_baseline(x, y, train_mask, val_mask, nclass, k_shot=5, epochs=200, lr=1e-3):
    model = MLP5Shot(x.size(1), 128, nclass).cuda()

    # ====== few-shot 采样 ======
    train_idx = []
    for c in range(nclass):
        cls_idx = (y == c).nonzero(as_tuple=True)[0]
        if len(cls_idx) >= k_shot:
            sampled = cls_idx[torch.randperm(len(cls_idx))[:k_shot]]
            train_idx.append(sampled)
    train_idx = torch.cat(train_idx)
    val_idx = val_mask.nonzero(as_tuple=True)[0].cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"[MLP Epoch {epoch}] Loss: {loss.item():.4f}")

    # ====== Validation ======
    model.eval()
    with torch.no_grad():
        pred = model(x[val_idx]).argmax(dim=1)
        acc = (pred == y[val_idx]).float().mean().item()
    print(f"🏁 MLP 5-shot Validation Acc: {acc:.4f}")
    return acc


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    device = 'cuda:{}'.format(args.cuda)
    torch.cuda.set_device(args.seed)

    epoch = config['epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    nclass = config['nclass']
    nlayer = config['nlayer']
    hidden_dim = config['hidden_dim']
    num_heads = config['num_heads']
    tran_dropout = config['tran_dropout']
    feat_dropout = config['feat_dropout']
    prop_dropout = config['prop_dropout']
    norm = config['norm']
    print('device:',device)
    if 'signal' in args.dataset:
        e, u, x, y, m = torch.load('data/{}.pt'.format(args.dataset))
        e, u, x, y, m = e.cuda(), u.cuda(), x.cuda(), y.cuda(), m.cuda()
        mask = torch.where(m == 1)
        x = x[:, args.image].unsqueeze(1)
        y = y[:, args.image]
    else:
        e, u, x, y,edge_index = torch.load('TAG_data/{}.pt'.format(args.dataset))
        label_embs = torch.load('label/{}/llm_y.pt'.format(args.dataset))
        # e = (e - e.mean()) / (e.std() + 1e-5)
        # # 找到低频和高频索引
        # low_idx = (e < 0.9).nonzero(as_tuple=True)[0]
        # high_idx = (e >= 1.1).nonzero(as_tuple=True)[0]
        # mid_idx = ((e >= 0.5) & (e < 1.5)).nonzero(as_tuple=True)[0]
        # print(f'Low freq count: {low_idx.shape[0]}, High freq count: {high_idx.shape[0]}')
        # # 取对应的特征值和特征向量
        # e_low, u_low = e[low_idx], u[:, low_idx]
        # e_high, u_high = e[high_idx], u[:, high_idx]
        # e_mid, u_mid = e[mid_idx], u[:, mid_idx]
        # e = e_mid
        # u = u_mid
        # # e = (e-e.min())/(e.max()-e.min())*2.0
        # e = torch.cat((e_low, e_high), dim=0)
        # u = torch.cat((u_low, u_high), dim=1)
        x= x.float()
        e, u, x, y,edge_index = e.cuda(), u.cuda(), x.cuda(), y.cuda(),edge_index.cuda()
        label_embs = label_embs.float()
        label_embs = label_embs.cuda()
        F.normalize(label_embs, p=2, dim=1)


        if len(y.size()) > 1:
            if y.size(1) > 1:
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1)

        train, valid, test = get_split(args.dataset, y, nclass, args.seed) 
        train, valid, test = map(torch.LongTensor, (train, valid, test))
        train, valid, test = train.cuda(), valid.cuda(), test.cuda()

    nfeat = x.size(1)
    net = Specformer(nfeat, nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, norm).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    print(count_parameters(net))

    res = []
    min_loss = 100.0
    max_acc = 0
    counter = 0
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
    if args.mode == 'visualize':
        net.load_state_dict(torch.load('checkpoints/{}.pt'.format(args.checkpoint)))
        net.eval()
        logits,attn = net(e, u, x)
        print(e.min(), e.max())
        print(attn.shape)
        attn = attn.detach().cpu().numpy()

        # --- Original Attention Visualization ---
        save_dir = './image'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # --- New Aggregated Attention Visualization ---
        e_cpu = e.cpu().numpy()
        # Define eigenvalue intervals from 0 to 1.5 with a step of 0.3
        boundaries = np.arange(0, 1.5 + 0.3, 0.3)
        num_intervals = len(boundaries) - 1
        
        # Find which interval each eigenvalue belongs to
        indices_by_interval = []
        for i in range(num_intervals):
            lower, upper = boundaries[i], boundaries[i+1]
            # Find indices of eigenvalues in the current interval [lower, upper)
            # The last interval includes the upper bound.
            if i == num_intervals - 1:
                mask = (e_cpu >= lower) & (e_cpu <= upper)
            else:
                mask = (e_cpu >= lower) & (e_cpu < upper)
            indices_by_interval.append(np.where(mask)[0])

        # Create and populate the aggregated attention matrix
        # We will average the attention for each block
        agg_attn = np.zeros((num_intervals, num_intervals))
        
        # We use the attention from the first head for this visualization
        attn_head_0 = attn

        for i in range(num_intervals):
            for j in range(num_intervals):
                row_indices = indices_by_interval[i]
                col_indices = indices_by_interval[j]
                
                # If either group is empty, attention is 0
                if len(row_indices) == 0 or len(col_indices) == 0:
                    agg_attn[i, j] = 0
                    continue
                
                # Extract the sub-matrix (block) and calculate the mean
                attn_block = attn_head_0[np.ix_(row_indices, col_indices)]
                agg_attn[i, j] = attn_block.mean()

        # Visualize the new aggregated attention matrix
        plt.figure(figsize=(6, 5))
        plt.imshow(agg_attn, cmap='magma', aspect='auto', origin='lower')
        
        # Set ticks and labels to correspond to interval boundaries
        tick_labels = [f'{b:.1f}' for b in boundaries]
        plt.xticks(ticks=np.arange(num_intervals + 1) - 0.5, labels=tick_labels)
        plt.yticks(ticks=np.arange(num_intervals + 1) - 0.5, labels=tick_labels)

        plt.title(f'Aggregated Attention Map ({args.checkpoint})')
        plt.xlabel('Eigenvalue Interval j')
        plt.ylabel('Eigenvalue Interval i')
        plt.colorbar(label='Average Attention Weight')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'attn_aggregated_{args.checkpoint}.png'), dpi=300)
        plt.show()
        plt.close()
        return

    if args.mode == 'test':
        net.load_state_dict(torch.load('checkpoints/{}.pt'.format(args.checkpoint)))
        net.eval()
        logits,_ = net(e, u, x)
        logits = F.normalize(logits, p=2, dim=1)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            r2 = r2_score(y[mask].data.cpu().numpy(), logits[mask].data.cpu().numpy())
            sse = torch.square(logits[mask] - y[mask]).sum().item()
            print('R2:', r2, 'SSE:', sse)
        else:
            similarity = torch.mm(logits, label_embs.t())/0.07
            test_acc = evaluation(similarity.cpu(), y.cpu()).item()
            ref_sim = torch.mm(x, label_embs.t())/0.07
            ref_acc = evaluation(ref_sim.cpu(), y.cpu()).item()
            print('Ref Acc:', ref_acc)
            print('Test Acc:', test_acc)
        few_shot_validate(net, e,u,x,y, label_embs,nclass, k_shot=3, device=device)
        few_shot_mlp_baseline(x, y, train, valid, nclass, k_shot=5, epochs=200, lr=1e-3)
        few_shot_gcn_baseline(x, label_embs,edge_index, y, train_mask=train, val_mask=valid,
                                             nclass=nclass, k_shot=5, epochs=200, lr=1e-3,
                                              device=device)
        evaluate_linear_model(x, edge_index, y, label_embs, nclass)
        return

    for idx in range(epoch):

        net.train()
        optimizer.zero_grad()
        logits,_ = net(e, u, x)
        F.normalize(logits, p=2, dim=1)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            loss = torch.square((logits[mask] - y[mask])).sum()
        else:
            similarity = torch.mm(logits[train], label_embs.t())/0.07
            loss = F.cross_entropy(similarity, y[train])


        loss.backward()
        optimizer.step()

        net.eval()
        logits,_ = net(e, u, x)

        if 'signal' in args.dataset:
            logits = logits.view(y.size())
            r2 = r2_score(y[mask].data.cpu().numpy(), logits[mask].data.cpu().numpy())
            sse = torch.square(logits[mask] - y[mask]).sum().item()
            print(r2, sse)
        else:
            # val_loss = F.cross_entropy(logits[valid], y[valid]).item()
            similarity = torch.mm(logits, label_embs.t())/0.07
            val_loss = F.cross_entropy(similarity[valid], y[valid]).item()

            val_acc = evaluation(similarity[valid].cpu(), y[valid].cpu()).item()
            test_acc = evaluation(similarity[test].cpu(), y[test].cpu()).item()
            res.append([val_loss, val_acc, test_acc])

            print(idx, val_loss, val_acc, test_acc)

            if val_loss < min_loss and val_acc > max_acc:
                max_acc = val_acc
                min_loss = val_loss
                counter = 0
            else:
                counter += 1

        if counter == 200:
            max_acc1 = sorted(res, key=lambda x: x[0], reverse=False)[0][-1]
            max_acc2 = sorted(res, key=lambda x: x[1], reverse=True)[0][-1]
            print(max_acc1, max_acc2)
            print(e.shape[0])
            
            break
    torch.save(net.state_dict(), 'checkpoints/{}.pt'.format(args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--image', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint',type=str, default='cornell')

    args = parser.parse_args()
    
    if 'signal' in args.dataset:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)['signal']
    else:
        config = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)[args.dataset]

    main_worker(args, config)

