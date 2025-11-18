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
# from model_node import Specformer,SpecLayer
from model import Specformer, LinearSGC1, LinearHGC1,LinearSGC2,LinearHGC2
from torch_geometric.nn import GCNConv
from utils import count_parameters, init_params, seed_everything, get_split
from torch_geometric.utils import add_self_loops, remove_self_loops, degree

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

def compute_linear_logits(features, y, sample_idx, nclass):
    """
    基于最小二乘法计算线性分类器的logits。
    Args:
        features: torch.Tensor [N, D]
        y: torch.Tensor [N]
        sample_idx: torch.Tensor [n_train]
        nclass: int
    Returns:
        logits: torch.Tensor [N, nclass]
    """
    device = features.device
    F_cpu = features[sample_idx].cpu()
    Y_train = F.one_hot(y[sample_idx], num_classes=nclass).float().cpu()
    
    # 使用最小二乘法求解 W
    W = torch.linalg.lstsq(F_cpu, Y_train, driver="gelss")[0]  # [D, nclass]
    
    # 得到所有节点的 logits
    logits = features @ W.to(device)
    return logits

def few_shot_validate(net, dataset_key, e,u,x,y, label_embs,n_way, k_shot=1):
    nclass = n_way
    seed_everything(0)
    sample_idx = []
    for c in range(n_way):
        cls_idx = (y == c).nonzero(as_tuple=True)[0]
        if len(cls_idx) >= k_shot:
            sampled = cls_idx[torch.randperm(len(cls_idx))[:k_shot]]
            sample_idx.append(sampled)
    sample_idx = torch.cat(sample_idx)
    net.eval()
    with torch.no_grad():
        new_e,attn = net.encoder(e)
        logits = net.conv(new_e, u, x)
        logits = compute_linear_logits(logits, y, sample_idx, nclass)
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
    acc = evaluation(logits.cpu(), y.cpu()).item()
    print(f"✅ [{dataset_key}] {k_shot}-shot Validation Accuracy: {acc:.4f}")
    sgc1 = LinearSGC1()
    sgc2 = LinearSGC2()
    hgc1 = LinearHGC1()
    hgc2 = LinearHGC2()
    for model,name in zip([sgc1,sgc2,hgc1,hgc2],['SGC1','SGC2','HGC1','HGC2']):
        logits_ref = model(e,u,x)
        logits_ref = compute_linear_logits(logits_ref, y, sample_idx, nclass)
        acc_ref = evaluation(logits_ref.cpu(), y.cpu()).item()
        print(f"✅ [{dataset_key}] {name} {k_shot}-shot Validation Accuracy: {acc_ref:.4f}")
    logits_ref = compute_linear_logits(x, y, sample_idx, nclass)
    acc_ref = evaluation(logits_ref.cpu(), y.cpu()).item()
    print(f"✅ [{dataset_key}] Linear {k_shot}-shot Validation Accuracy: {acc_ref:.4f}")
    return acc

# class MLP5Shot(nn.Module):
#     def __init__(self, in_dim, hidden_dim, nclass):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, nclass)
#         )

#     def forward(self, x):
#         return self.mlp(x)

# class GCN5Shot(nn.Module):
#     """
#     简单 2-layer GCN，用于 few-shot baseline。
#     forward 输入: x [N, F], edge_index [2, E]
#     返回: logits 或 embedding （取决于最后一层是否输出 nclass）
#     """
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5, use_bn=False):
#         super().__init__()
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, out_dim)
#         self.dropout = dropout
#         self.use_bn = use_bn
#         if use_bn:
#             self.bn1 = nn.BatchNorm1d(hidden_dim)

#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()
#         if self.use_bn:
#             self.bn1.reset_parameters()

#     def forward(self, x, edge_index):
#         # ensure self-loops (GCNConv usually benefits from self-loops)
#         # (torch_geometric GCNConv handles self-loops internally in many versions,
#         #  but to be safe we'll add them)
#         edge_index, _ = remove_self_loops(edge_index)
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#         x = self.conv1(x, edge_index)
#         if self.use_bn:
#             x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x  # 如果 out_dim == nclass 则为 logits；否则可以认为是 embedding
# def normalize_adj(edge_index, num_nodes):
#     """
#     计算标准化邻接矩阵 A_hat = D^{-1/2} (A + I) D^{-1/2}
#     返回: edge_index, norm
#     """
#     edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
#     row, col = edge_index
#     deg = degree(col, num_nodes, dtype=torch.float32)
#     deg_inv_sqrt = deg.pow(-0.5)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#     return edge_index, norm

# def propagate_once(x, edge_index, norm):
#     row, col = edge_index
#     out = torch.zeros_like(x)
#     out.index_add_(0, row, norm.unsqueeze(1) * x[col])
#     return out

# # ========== 线性 SGC/HGC 模型 ==========
# class LinearSGC1(nn.Module):
#     def forward(self, x, edge_index):
#         edge_index, norm = normalize_adj(edge_index, x.size(0))
#         return propagate_once(x, edge_index, norm)

# class LinearSGC2(nn.Module):
#     def forward(self, x, edge_index):
#         edge_index, norm = normalize_adj(edge_index, x.size(0))
#         x1 = propagate_once(x, edge_index, norm)
#         x2 = propagate_once(x1, edge_index, norm)
#         return x2

# class LinearHGC1(nn.Module):
#     def forward(self, x, edge_index):
#         edge_index, norm = normalize_adj(edge_index, x.size(0))
#         low = propagate_once(x, edge_index, norm)
#         return x - low  # 高通

# class LinearHGC2(nn.Module):
#     def forward(self, x, edge_index):
#         edge_index, norm = normalize_adj(edge_index, x.size(0))
#         low = propagate_once(x, edge_index, norm)
#         low2 = propagate_once(low, edge_index, norm)
#         return x - low2  # 二阶高通
# def evaluate_linear_model(x, edge_index, y, label_embs, nclass):
#     """
#     通用评估函数：使用线性层对 encoder 生成的嵌入做分类。
#     """

#     models = {
#         'SGC1': LinearSGC1(),
#         'SGC2': LinearSGC2(),
#         'HGC1': LinearHGC1(),
#         'HGC2': LinearHGC2(),
#     }

#     results = {}
#     for name, model in models.items():
#         evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)
#         logits = torch.mm(x, label_embs.t())/0.07
#         logits = model(logits, edge_index)  # [N, F']
#         # similarity = torch.mm(logits, label_embs.t())/0.07
#         acc = evaluation(logits.cpu(), y.cpu()).item()
#         results[name] = acc

#     print("\n📊 Summary:")
#     for k, v in results.items():
#         print(f"{k}: {v:.4f}")

#     return 


# # ========== few-shot 训练/验证函数 ==========
# def few_shot_gcn_baseline(x, label_embs,edge_index, y, train_mask, val_mask, nclass,
#                           k_shot=5, epochs=200, lr=1e-3,
#                           weight_decay=0.0, device='cuda'):
#     """
#     用 5-shot（或 k-shot）训练 GCN 的 baseline。
#     参数:
#         x: [N, F] 特征 tensor (CPU or CUDA)
#         edge_index: [2, E] long tensor
#         y: [N] long tensor (labels 0..C-1)
#         train_mask, val_mask: boolean or index tensors indicating train/val sets (仅用于选择验证集)
#         nclass: 类别数
#         k_shot: 每类采样 k 个样本做 few-shot 训练
#         epochs, lr, hidden_dim, weight_decay, device: 训练超参
#     返回:
#         val_acc 在 val_mask 上的准确率
#     """
    

#     # 初始化模型（输出维度为 nclass -> 直接输出 logits）
#     logits = torch.mm(x, label_embs.t())/0.07  # 预计算相似度矩阵作为输入
#     model = GCN5Shot(in_dim=logits.size(1), hidden_dim=logits.size(1), out_dim=logits.size(1), dropout=0.5, use_bn=True).cuda()
#     model.reset_parameters()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     criterion = nn.CrossEntropyLoss()

#     # ====== few-shot 采样 ======
#     sample_idx = []
#     for c in range(nclass):
#         cls_idx = (y == c).nonzero(as_tuple=True)[0]
#         if len(cls_idx) >= k_shot:
#             sampled = cls_idx[torch.randperm(len(cls_idx))[:k_shot]]
#             sample_idx.append(sampled)
#     sample_idx = torch.cat(sample_idx)
#     val_idx = val_mask.nonzero(as_tuple=True)[0].cuda()

#     # ========== 训练 ==========
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(logits, edge_index)  # [N, nclass]
#         loss = criterion(out[sample_idx], y[sample_idx])
#         loss.backward()
#         optimizer.step()
#         if epoch % max(1, epochs // 10) == 0:
#             print(f"[GCN Epoch {epoch}/{epochs}] Loss: {loss.item():.4f}")

#     # ========== 验证 ==========
#     model.eval()
#     with torch.no_grad():
#         out = model(logits, edge_index)
#         pred = out[val_idx].argmax(dim=1)
#         acc = (pred == y[val_idx]).float().mean().item()
#     print(f"🏁 GCN {k_shot}-shot Validation Acc: {acc:.4f}")
#     return acc, model



# def few_shot_mlp_baseline(x, y, train_mask, val_mask, nclass, k_shot=5, epochs=200, lr=1e-3):
#     model = MLP5Shot(x.size(1), 128, nclass).cuda()

#     # ====== few-shot 采样 ======
#     sample_idx = []
#     for c in range(nclass):
#         cls_idx = (y == c).nonzero(as_tuple=True)[0]
#         if len(cls_idx) >= k_shot:
#             sampled = cls_idx[torch.randperm(len(cls_idx))[:k_shot]]
#             sample_idx.append(sampled)
#     sample_idx = torch.cat(sample_idx)
#     val_idx = val_mask.nonzero(as_tuple=True)[0].cuda()

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         out = model(x)
#         loss = F.cross_entropy(out[sample_idx], y[sample_idx])
#         loss.backward()
#         optimizer.step()
#         if epoch % 50 == 0:
#             print(f"[MLP Epoch {epoch}] Loss: {loss.item():.4f}")

#     # ====== Validation ======
#     model.eval()
#     with torch.no_grad():
#         pred = model(x[val_idx]).argmax(dim=1)
#         acc = (pred == y[val_idx]).float().mean().item()
#     print(f"🏁 MLP 5-shot Validation Acc: {acc:.4f}")
#     return acc
def load_single_dataset(name, device):
    print(f"Loading dataset: {name}")
    e,u,x,y = torch.load(f"data/{name}.pt")
    x = x.float()

    if len(y.size()) > 1:
        y = torch.argmax(y, dim=1) if y.size(1) > 1 else y.view(-1)

    e,u,x,y = e.to(device), u.to(device), x.to(device), y.to(device)
    nfeat = x.size(1)
    nclass = int(y.max().item() + 1)

    return {
        "name": name,
        "e": e,
        "u": u,
        "x": x,
        "y": y,
        "nfeat": nfeat,
        "nclass": nclass
    }

def prepare_dataset(args, dataset_name, device):
    data = load_single_dataset(dataset_name, device)
    train, valid, test = get_split(dataset_name, data["y"], data["nclass"], args.seed)
    def _as_index(split):
        tensor = torch.as_tensor(split, device=device)
        if tensor.dtype == torch.bool:
            tensor = tensor.nonzero(as_tuple=False).view(-1)
        else:
            tensor = tensor.to(torch.long).view(-1)
        return tensor
    data["train_idx"] = _as_index(train)
    data["valid_idx"] = _as_index(valid)
    data["test_idx"] = _as_index(test)
    return data

def main_worker(args, config):
    print(args, config)
    seed_everything(0)
    device = 'cuda:{}'.format(args.cuda)
    torch.cuda.set_device(args.cuda)

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
    dataset_names = getattr(args, 'dataset_names', None) or [args.dataset]
    if len(dataset_names) == 1 and 'signal' in dataset_names[0]:
        e, u, x, y, m = torch.load('data/{}.pt'.format(args.dataset))
        e, u, x, y, m = e.cuda(), u.cuda(), x.cuda(), y.cuda(), m.cuda()
        mask = torch.where(m == 1)
        x = x[:, args.image].unsqueeze(1)
        y = y[:, args.image]
    else:
        datasets = {name: prepare_dataset(args, name, device) for name in dataset_names}
        dataset_meta = {name: {"nfeat": data["nfeat"], "nclass": data["nclass"]} for name, data in datasets.items()}
        print(num_heads)
        net = Specformer(
            dataset_meta,
            None,
            nlayer,
            hidden_dim,
            num_heads,
            tran_dropout,
            feat_dropout,
            prop_dropout,
            norm
        ).cuda()
        net.apply(init_params)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        print(count_parameters(net))
        if args.mode == 'visualize':
            target = dataset_names[0]
            data = datasets[target]
            net.encoder.load_state_dict(torch.load(f'encoders/{args.checkpoint}.pt'))
            net.eval()
            with torch.no_grad():
                encoded_eig, attn = net.encoder(data["e"])
            attn = attn.detach().cpu().numpy()
            save_dir = './image'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            e_cpu = data["e"].detach().cpu().numpy()
            # --- Original Attention Visualization ---
            # save_dir = './image'
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)

            # Define eigenvalue intervals from 0 to 1.5 with a step of 0.3
            boundaries = np.arange(0, 1.5 + 0.3, 0.3)
            num_intervals = len(boundaries) - 1
            
            # Find which interval each eigenvalue belongs to
            indices_by_interval = []
            for i in range(num_intervals):
                lower, upper = boundaries[i], boundaries[i+1]
                # Find indices of eigenvalues in the current interval [lower, upper)
                # The last interval includes the upper bound.
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
            print(agg_attn.sum(), agg_attn.shape,agg_attn.mean())
            agg_attn = agg_attn / agg_attn.sum()
            print(agg_attn.sum(), agg_attn.shape,agg_attn.mean())
            print("Aggregated Attention Matrix:")
            print(agg_attn)
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
            orig_vals = e_cpu.reshape(-1)
            filt_vals = encoded_eig.detach().cpu().numpy()
            if filt_vals.ndim == 1:
                filt_vals = filt_vals[:, np.newaxis]
            order = np.argsort(orig_vals)
            orig_sorted = orig_vals[order]
            filt_sorted = filt_vals[order]
            plt.figure(figsize=(6, 4))
            for head in range(filt_sorted.shape[1]):
                label = f'Head {head}' if filt_sorted.shape[1] > 1 else 'Filter'
                plt.plot(orig_sorted, filt_sorted[:, head], label=label)
            plt.title(f'Spectral Filter ({args.checkpoint})')
            plt.xlabel('Original Eigenvalue')
            plt.ylabel('Encoded Value')
            if filt_sorted.shape[1] > 1:
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'filter_{target}.png'), dpi=300)
            plt.show()
            plt.close()
            return
        if args.mode == 'test':
            net.encoder.load_state_dict(torch.load(f'encoders/{args.checkpoint}.pt'))
            net.eval()
            for name, data in datasets.items():
                logits,_ = net(data["e"], data["u"], data["x"], dataset_key=name)
                evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=data["nclass"])
                test_acc = evaluation(logits.cpu(), data["y"].cpu()).item()
                print(f'[{name}] Test Accuracy: {test_acc:.4f}')
                few_shot_validate(net, name, data["e"], data["u"], data["x"], data["y"], None, data["nclass"], k_shot=5)
            return
        if args.mode == 'train':
            res = {name: [] for name in dataset_names}
            min_loss = float('inf')
            counter = 0
            for idx in range(epoch):
                net.train()
                optimizer.zero_grad()
                losses = []
                train_loss_items = {}
                for name, data in datasets.items():
                    logits,_ = net(data["e"], data["u"], data["x"], dataset_key=name)
                    loss = F.cross_entropy(logits[data["train_idx"]], data["y"][data["train_idx"]])
                    losses.append(loss)
                    train_loss_items[name] = loss.detach().item()
                total_loss = torch.stack(losses).mean()
                total_loss.backward()
                optimizer.step()
                net.eval()
                val_loss_map, val_acc_map, test_acc_map = {}, {}, {}
                with torch.no_grad():
                    for name, data in datasets.items():
                        logits,_ = net(data["e"], data["u"], data["x"], dataset_key=name)
                        val_loss = F.cross_entropy(logits[data["valid_idx"]], data["y"][data["valid_idx"]]).item()
                        val_metric = torchmetrics.Accuracy(task='multiclass', num_classes=data["nclass"])
                        test_metric = torchmetrics.Accuracy(task='multiclass', num_classes=data["nclass"])
                        val_acc = val_metric(logits[data["valid_idx"]].cpu(), data["y"][data["valid_idx"]].cpu()).item()
                        test_acc = test_metric(logits[data["test_idx"]].cpu(), data["y"][data["test_idx"]].cpu()).item()
                        res[name].append((train_loss_items[name], val_loss, val_acc, test_acc))
                        val_loss_map[name] = val_loss
                        val_acc_map[name] = val_acc
                        test_acc_map[name] = test_acc
                metrics_msg = ' | '.join(
                    f"{name} train {train_loss_items[name]:.4f} val {val_loss_map[name]:.4f} "
                    f"val_acc {val_acc_map[name]:.4f} test_acc {test_acc_map[name]:.4f}"
                    for name in dataset_names
                )
                print(idx, total_loss.item(), metrics_msg)
                val_loss_sum = sum(val_loss_map.values())
                if val_loss_sum < min_loss:
                    min_loss = val_loss_sum
                    counter = 0
                else:
                    counter += 1
                if counter == 200:
                    break
            for name, logs in res.items():
                if logs:
                    best_test_at_min_val = min(logs, key=lambda x: x[1])[-1]
                    best_test_at_best_val = max(logs, key=lambda x: x[2])[-1]
                    print(f"{name}: {best_test_at_min_val} {best_test_at_best_val}")
            save_tag = dataset_names[0] if len(dataset_names) == 1 else '_'.join(dataset_names)
            torch.save(net.encoder.state_dict(), f'encoders/{save_tag}.pt')
            return
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--image', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--checkpoint',type=str, default='cornell')
    parser.add_argument('--datasets', type=str, default=None, help='Comma separated dataset names for joint encoder training')

    args = parser.parse_args()
    dataset_names = [d.strip() for d in args.datasets.split(',') if d.strip()] if args.datasets else [args.dataset]
    args.dataset_names = dataset_names
    args.dataset = dataset_names[0]
    config_root = yaml.load(open('config.yaml'), Loader=yaml.SafeLoader)
    if len(dataset_names) == 1 and 'signal' in dataset_names[0]:
        config = config_root['signal']
    else:
        config = config_root[dataset_names[0]]
    main_worker(args, config)

