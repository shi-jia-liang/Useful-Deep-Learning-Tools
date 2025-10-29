# 交叉聚合器
import torch
from torch import nn
import torch.nn.functional as F
from .attention_blocks import AttentionBlock


class CrossAggregator(nn.Module):
    """Aggregate a list of same-scale feature maps (B, C, H, W) across images using cross-attention.

    Returns a list of aggregated tensors matched to input order.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.projs = nn.ModuleDict()
        self.cross = AttentionBlock(d_model=d_model, nhead=nhead)

    def _ensure_proj(self, in_ch: int, key: str, device):
        if key not in self.projs:
            conv = nn.Conv2d(in_ch, self.d_model, kernel_size=1)
            conv = conv.to(device)
            self.projs[key] = conv
        return self.projs[key]

    def forward(self, feat_list):
        # feat_list: list of (B, C, H, W)
        if not isinstance(feat_list, (list, tuple)):
            raise ValueError('feat_list must be list or tuple')
        n = len(feat_list)
        if n == 0:
            return []

        projs = []
        for f in feat_list:
            p = self._ensure_proj(f.shape[1], f'p{f.shape[1]}', f.device)(f)
            projs.append(p)

        if n == 1:
            return projs

        seqs = [p.flatten(2).permute(0, 2, 1) for p in projs]
        outs = []
        for i in range(n):
            q = seqs[i]
            kv = torch.cat([seqs[j] for j in range(n) if j != i], dim=1)
            out_seq = self.cross(q, kv, kv)
            B, S, C = out_seq.shape
            out = out_seq.permute(0, 2, 1).reshape(B, C, feat_list[0].shape[2], feat_list[0].shape[3])
            outs.append(out)
        return outs
