import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.extra_feat_dim = 4
        self.eig_w = nn.Linear(hidden_dim + 1 + self.extra_feat_dim, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        norm_e = (e - e.min()) / (e.max() - e.min() + 1e-6)
        idx = torch.arange(e.size(0), device=e.device, dtype=e.dtype)
        pos = idx / max(e.size(0) - 1, 1)
        inv_pos = 1 - pos
        high_flag = (norm_e > 0.5).to(e.dtype)
        context = torch.stack((e, norm_e, pos, inv_pos, high_flag), dim=1)
        eeig = torch.cat((context, torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
    
# class SpecLayer(nn.Module):

#     def __init__(self, ):
#         super(SpecLayer, self).__init__()
#         # self.linear = nn.Linear(nbases, 1, bias=False)

#         # if norm == 'none': 
#         #     self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
#         # else:
#         #     self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
#         #     nn.init.normal_(self.weight, mean=0.0, std=0.01)

#         # if norm == 'layer':    # Arxiv
#         #     self.norm = nn.LayerNorm(ncombines)
#         # elif norm == 'batch':  # Penn
#         #     self.norm = nn.BatchNorm1d(ncombines)
#         # else:                  # Others
#         #     self.norm = None 

#     def forward(self, x):
#         x = self.prop_dropout(x)     # [N, m, d] * [1, m, d]
#         x = torch.mean(x, dim=1)

#         # if self.norm is not None:
#         #     x = self.norm(x)
#         #     x = F.relu(x)
#         x = F.relu(x)
#         return x


class SpecEncoder(nn.Module):
    def __init__(self, hidden_dim=128, nheads=1, tran_dropout=0.0):
        super().__init__()
        self.eig_encoder = SineEncoding(hidden_dim)
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, 1, tran_dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
        self.decoders = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(nheads)])
        self.head_emb = nn.Parameter(torch.randn(nheads, hidden_dim))
        self.hidden_dim = hidden_dim
        self.nheads = nheads

    def forward(self, e):
        eig = self.eig_encoder(e)
        head_base = eig.unsqueeze(0).expand(self.nheads, -1, -1) + self.head_emb.unsqueeze(1)
        mha_in = self.mha_norm(head_base)
        mha_out, attn = self.mha(mha_in, mha_in, mha_in, need_weights=True, average_attn_weights=False)
        head_post = head_base + self.mha_dropout(mha_out)
        ffn_in = self.ffn_norm(head_post)
        ffn_out = self.ffn(ffn_in.reshape(-1, self.hidden_dim)).reshape(self.nheads, -1, self.hidden_dim)
        eig_head = head_post + self.ffn_dropout(ffn_out)
        eig_head = eig_head.permute(1, 0, 2)
        new_e = torch.cat([decoder(eig_head[:, idx, :]) for idx, decoder in enumerate(self.decoders)], dim=1)
        attn = attn.squeeze(1).mean(dim=0)
        return new_e, attn



class SpecConv(nn.Module):
    # def __init__(self, nclass, nfeat, nheads=1, hidden_dim=128,feat_dropout=0.0,prop_dropout=0.0, nlayer=1, norm='none'):
    def __init__(self, nheads=1):
        super().__init__()
        self.nheads = nheads
        self.head_logits = nn.Parameter(torch.zeros(self.nheads + 1))
        self.sim_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, e, u, x, dataset_vec=None):
        ut = u.permute(1, 0)
        # h = self.feat_dp1(x)
        # h = self.feat_encoder(h)
        # h = self.linear_encoder(h)
        # h = self.feat_dp2(h)
        h = x.clone()
        # for conv in self.layers:
        basic_feats = [h]
        utx = ut @ h
        for i in range(self.nheads):
            # print(e.shape,u.shape,x.shape)
            basic_feats.append(u @ (e[:, i].unsqueeze(1) * utx))
        stacked_feats = torch.stack(basic_feats, dim=1)
        if dataset_vec is not None:
            norm_dataset = F.normalize(dataset_vec, dim=0, eps=1e-6)
            norm_heads = F.normalize(stacked_feats, dim=2, eps=1e-6)
            cos_scores = torch.matmul(norm_heads, norm_dataset)
            weights = F.softmax(cos_scores / torch.clamp(self.sim_temperature, min=1e-3), dim=1)
        else:
            base_weights = F.softmax(self.head_logits, dim=0)
            weights = base_weights.unsqueeze(0).expand(stacked_feats.size(0), -1)
        basic_feats = (stacked_feats * weights.unsqueeze(-1)).sum(dim=1)
        # if self.norm is not None:
        #     x = self.norm(x)
        #     x = F.relu(x)
        basic_feats = F.relu(basic_feats)
        h = basic_feats
        # h = self.feat_dp2(h)
        # h = self.classify(h)
        return h
    
class Specformer(nn.Module):
    def __init__(self, dataset_meta_or_nclass, nfeat=None, nlayer=1, hidden_dim=128, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super().__init__()
        if isinstance(dataset_meta_or_nclass, dict):
            dataset_meta = dataset_meta_or_nclass
        else:
            if nfeat is None:
                raise ValueError("nfeat must be provided when dataset metadata is not supplied.")
            dataset_meta = {"default": {"nclass": dataset_meta_or_nclass, "nfeat": nfeat}}
        if not dataset_meta:
            raise ValueError("dataset_meta must contain at least one dataset.")
        self.dataset_meta = dataset_meta
        print(nheads)
        self.encoder = SpecEncoder(
            hidden_dim=hidden_dim,
            nheads=nheads,
            tran_dropout=tran_dropout
        )
        self.conv = SpecConv(
            nheads=nheads,
        )
        self.dataset_classifiers = nn.ModuleDict({
            name: nn.Linear(meta["nfeat"], meta["nclass"])
            for name, meta in dataset_meta.items()
        })
        self.dataset_gates = nn.ParameterDict({
            name: nn.Parameter(torch.randn(meta["nfeat"]))
            for name, meta in dataset_meta.items()
        })
        self._default_dataset_key = next(iter(self.dataset_classifiers.keys()))
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.feat_dropout = feat_dropout
        self.prop_dropout = prop_dropout
        self.nlayer = nlayer
        self.norm = norm
        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

    def register_dataset_head(self, name, nfeat, nclass):
        if name in self.dataset_classifiers:
            return
        device = next(self.parameters()).device
        self.dataset_classifiers[name] = nn.Linear(nfeat, nclass).to(device)
        self.dataset_meta[name] = {"nfeat": nfeat, "nclass": nclass}
        self.dataset_gates[name] = nn.Parameter(torch.randn(nfeat, device=device))

    def forward(self, e,u,x, dataset_key=None):
        if dataset_key is None:
            dataset_key = self._default_dataset_key
        if dataset_key not in self.dataset_classifiers:
            raise KeyError(f"Unknown dataset head: {dataset_key}")
        new_e,attn = self.encoder(e)
        h = self.feat_dp1(x)
        dataset_vec = self.dataset_gates[dataset_key]
        out = self.conv(new_e, u, h, dataset_vec)
        out = self.feat_dp2(out)
        logits = self.dataset_classifiers[dataset_key](out)
        return logits, attn

class LinearSGC2(nn.Module):
    """Compute X' = U diag(e) U^T X
    - u: [N, k]   (eigenvectors, columns are eigenvectors)
    - e: [k]      (eigenvalues or filter coeff for each eigen)
    - x: [N, d]   (node features)
    """
    def __init__(self):
        super().__init__()

    def forward(self, e, u, x):
        # e: [k] or [k, 1]
        # u: [N, k]
        # x: [N, d]
        # utx = U^T X -> [k, d]
        utx = u.t() @ x                # [k, d]
        e = (1-e)*(1-e)
        # apply spectral filter coefficients (broadcast over feature dim)
        filtered = (e.unsqueeze(1) * utx)   # [k, d]
        out = u @ filtered             # [N, d]
        out = F.relu(out)
        return out

class LinearSGC1(nn.Module):
    """SGC1: h(λ) = 1 - λ"""
    def __init__(self):
        super().__init__()

    def forward(self, e, u, x):
        # e: eigenvalues [k]
        # u: eigenvectors [N, k]
        # x: node features [N, d]
        utx = u.t() @ x            # [k, d]
        filt = (1 - e).unsqueeze(1) * utx
        out = u @ filt
        out = F.relu(out)
        return out

class LinearHGC1(nn.Module):
    """HGC1: h(λ) = λ"""
    def __init__(self):
        super().__init__()

    def forward(self, e, u, x):
        utx = u.t() @ x            # [k, d]
        filt = e.unsqueeze(1) * utx
        out = u @ filt
        out = F.relu(out)
        return out

class LinearHGC2(nn.Module):
    """HGC2: h(λ) = λ^2"""
    def __init__(self):
        super().__init__()

    def forward(self, e, u, x):
        utx = u.t() @ x            # [k, d]
        filt = (e * e).unsqueeze(1) * utx
        out = u @ filt
        out = F.relu(out)
        return out





