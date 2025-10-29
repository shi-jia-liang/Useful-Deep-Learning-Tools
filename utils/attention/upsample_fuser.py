# 上采样并融合低分辨率和高分辨率特征
import torch
from torch import nn
import torch.nn.functional as F
from .attention_blocks import AttentionBlock


class UpsampleFuser(nn.Module):
    """Upsample low-res features and fuse with higher-res features.

    Modes:
      - 'concat': concatenate upsampled + proj(high) then 1x1 conv
      - 'cross_attn': use cross-attention (q=proj(high), kv=upsampled)
    """
    def __init__(self, d_model: int = 256, fusion: str = 'concat', nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.fusion = fusion
        self.nhead = nhead
        self.projs = nn.ModuleDict()
        if fusion == 'concat':
            # dynamic conv will be created at forward if needed
            self._dummy = nn.Identity()
        else:
            self.cross = AttentionBlock(d_model=d_model, nhead=nhead)

    def _ensure_proj(self, in_ch: int, key: str, device):
        if key not in self.projs:
            conv = nn.Conv2d(in_ch, self.d_model, kernel_size=1)
            conv = conv.to(device)
            self.projs[key] = conv
        return self.projs[key]

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor, align_corners: bool = False):
        # low_feat: (B, C_low, Hl, Wl)
        # high_feat: (B, C_high, Hh, Wh)
        up = F.interpolate(low_feat, size=high_feat.shape[2:], mode='bilinear', align_corners=align_corners)
        up = self._ensure_proj(up.shape[1], f'p_up{up.shape[1]}', up.device)(up)
        proj_high = self._ensure_proj(high_feat.shape[1], f'p{high_feat.shape[1]}', high_feat.device)(high_feat)

        if self.fusion == 'concat':
            cat = torch.cat([up, proj_high], dim=1)
            conv_fuse = nn.Conv2d(cat.shape[1], self.d_model, kernel_size=1).to(cat.device)
            out = F.relu(conv_fuse(cat))
            return out
        else:
            # cross-attn: q=proj_high, kv=up
            q = proj_high.flatten(2).permute(0, 2, 1)
            kv = up.flatten(2).permute(0, 2, 1)
            out_seq = self.cross(q, kv, kv)
            B, S, C = out_seq.shape
            out = out_seq.permute(0, 2, 1).reshape(B, C, high_feat.shape[2], high_feat.shape[3])
            return out
