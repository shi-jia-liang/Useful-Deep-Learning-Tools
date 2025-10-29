# 2D 位置编码，支持两种模式：sinusoidal（无参数）和 learnable（小型 conv）
import math
from typing import Literal

import torch
from torch import nn


class PositionEncoding2D(nn.Module):
    """2D position encoding for feature maps.

    Args:
      mode: 'sin' or 'learnable'. 'sin' returns fixed 8-channel sin/cos map (same as previous impl).
      pe_ch: number of channels produced by encoding (for 'sin' mode default=8). For 'learnable' mode
             this is the output channels of a small conv.

    Forward:
      - input x: (B, C, H, W) -> returns concatenated tensor (B, C+pe_ch, H, W)
    """

    def __init__(self, mode: Literal['sin', 'learnable'] = 'sin', pe_ch: int = 8):
        super().__init__()
        self.mode = mode
        self.pe_ch = pe_ch
        if mode == 'learnable':
            # small conv to produce pe_ch channels from 2-channel coord input
            self.net = nn.Sequential(
                nn.Conv2d(2, pe_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(pe_ch, pe_ch, kernel_size=1),
            )
        else:
            self.net = None

    def _sin_pe(self, device, dtype, h: int, w: int) -> torch.Tensor:
        pe = torch.zeros(self.pe_ch, h, w, device=device, dtype=dtype)
        y_position = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).unsqueeze(1).expand(h, w)
        x_position = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).unsqueeze(0).expand(h, w)
        pi = math.pi
        # support pe_ch up to 8 in the simple preset; if pe_ch differs, tile or truncate
        # base 8-channel template
        base = [x_position, y_position,
                torch.sin(x_position * pi * 2), torch.cos(y_position * pi * 2),
                torch.sin(x_position * pi * 8), torch.cos(y_position * pi * 8),
                torch.sin(x_position * pi * 32), torch.cos(y_position * pi * 32)]
        base_ch = len(base)
        if self.pe_ch <= base_ch:
            for i in range(self.pe_ch):
                pe[i] = base[i]
        else:
            # repeat base channels if pe_ch larger
            for i in range(self.pe_ch):
                pe[i] = base[i % base_ch]
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            raise ValueError('Input tensor x is None')
        if x.dim() != 4:
            raise ValueError(f'Expected input with 4 dims (B,C,H,W), got {x.dim()} dims')
        bs, _, h, w = x.shape
        if self.mode == 'sin':
            with torch.no_grad():
                pe = self._sin_pe(x.device, x.dtype, h, w).unsqueeze(0).expand(bs, -1, -1, -1)
            out = torch.cat([x, pe], dim=1)
            return out
        else:
            # learnable mode: build coordinate grid and feed through small conv
            assert self.net is not None, 'Internal error: learnable net not initialized'
            y_position = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype).unsqueeze(1).expand(h, w)
            x_position = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype).unsqueeze(0).expand(h, w)
            coords = torch.stack([x_position, y_position], dim=0).unsqueeze(0).expand(bs, -1, -1, -1)
            pe = self.net(coords)
            out = torch.cat([x, pe], dim=1)
            return out

    def visualize_pe(self, h: int, w: int, out_path: str):
        """Save a small visualization of the positional encoding (for 'sin' mode)."""
        import matplotlib.pyplot as plt
        pe = self._sin_pe('cpu', torch.float32, h, w)
        # show first 3 channels
        fig, axes = plt.subplots(1, min(3, pe.shape[0]), figsize=(3 * min(3, pe.shape[0]), 3))
        for i in range(min(3, pe.shape[0])):
            ax = axes[i] if hasattr(axes, '__iter__') else axes
            im = pe[i].cpu().numpy()
            ax.imshow(im, cmap='RdBu')
            ax.axis('off')
        fig.savefig(out_path, bbox_inches='tight')
