# 注意力机制
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
import warnings


class AttentionBlock(nn.Module):
    """通用注意力原语（可做 self-attn 或 cross-attn）。

    Contract:
      - Inputs are sequences in shape (B, S, C) when batch_first=True.
      - forward(q, k=None, v=None, need_weights=False) returns either out or (out, attn_weights).
      - If need_weights=True, a CPU copy of latest attention weights is stored in
        `self.last_attn_weights_cpu` (or None if not computed).

    Notes:
      - For long sequence lengths (S large) this module may allocate significant memory.
      - Uses torch.nn.MultiheadAttention with batch_first=True.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        mha_dropout: float = 0.3,
        ffn_dropout: float = 0.3,
        activation: str = 'relu',
    ):
        super().__init__()
        # 多头注意力，支持内部 dropout
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=mha_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # activation function
        if activation == 'gelu':
            act_layer = nn.GELU()
        else:
            act_layer = nn.ReLU(inplace=True)

        # 前馈网络，带 dropout
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            act_layer,
            nn.Dropout(ffn_dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(ffn_dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # store last attention weights (CPU tensor) when need_weights=True
        self.last_attn_weights_cpu: Optional[torch.Tensor] = None

    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """执行注意力：如果未提供 k/v，则为 self-attention。

        当 need_weights=True 时返回 (out, attn_weights)，否则只返回 out。
        """
        k = q if k is None else k
        v = q if v is None else v

        # 警告：当序列长度很大时，全局 attention 可能会耗尽内存
        S = q.shape[1]
        if S > 4096:
            warnings.warn(f'Large sequence length S={S} for AttentionBlock - may be slow or OOM.')

        if need_weights:
            attn_out, attn_weights = self.mha(q, k, v, need_weights=True)
            # store a CPU copy for debugging/visualization (detach)
            try:
                self.last_attn_weights_cpu = attn_weights.detach().cpu()
            except Exception:
                self.last_attn_weights_cpu = None
        else:
            attn_out, _ = self.mha(q, k, v, need_weights=False)
            attn_weights = None
            self.last_attn_weights_cpu = None

        x = self.norm1(q + attn_out)
        ff = self.ff(x)
        x = self.norm2(x + ff)

        if need_weights:
            return x, attn_weights
        return x

    def save_last_attn(self, path: str) -> None:
        """Save last attention weights (CPU tensor) to disk.

        If last_attn_weights_cpu is a Tensor, saves as .npy for numpy-friendly formats and
        .pt if path ends with .pt.
        """
        if self.last_attn_weights_cpu is None:
            raise RuntimeError('No last attention weights to save. Call forward(..., need_weights=True) first.')
        if path.endswith('.npy'):
            np.save(path, self.last_attn_weights_cpu.numpy())
        else:
            torch.save(self.last_attn_weights_cpu, path)
