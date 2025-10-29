import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .attention_blocks import AttentionBlock
from typing import List


class MultiScalePyramidFusion(nn.Module):
    """Pyramid-local fusion:
    - global self-attn on deep scales (feat_32, feat_16)
    - upsample+fuse into feat_8 using provided UpsampleFuser (external)
    - local windowed self-attn on fused /8 to inject local context

    This module expects an UpsampleFuser-like object passed as `fuser` to reuse its projection convs.
    """

    def __init__(self, fuser, d_model: int = 256, nhead: int = 8, window_size: int = 8):
        super().__init__()
        self.fuser = fuser
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size

        # attention blocks
        self.global_attn = AttentionBlock(d_model=d_model, nhead=nhead)
        self.local_attn = AttentionBlock(d_model=d_model, nhead=nhead)

    def _proj(self, x: torch.Tensor):
        # reuse fuser projection convs to project arbitrary in_ch -> d_model
        conv = self.fuser._ensure_proj(x.shape[1], f'p{x.shape[1]}', x.device)
        return conv(x)

    # Note: pooling-based key-length limiting was intentionally removed per user request
    # because input image sizes are constrained upstream (<=640x480). Keep logic simple.

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """feats: dict with keys 'feat_32','feat_16','feat_8' (some may be None)
        returns: fused feature at /8 resolution (B, d_model, H8, W8)
        """
        feat32 = feats.get('feat_32', None)
        feat16 = feats.get('feat_16', None)
        feat8 = feats.get('feat_8', None)
        # 1) project and global-attend on deep scales (if present)
        seqs: List[torch.Tensor] = []
        shapes = {}
        if feat32 is not None:
            p32 = self._proj(feat32)
            B, C, H32, W32 = p32.shape
            seq32 = p32.flatten(2).permute(0, 2, 1)
            seqs.append(seq32)
            shapes['32'] = (H32, W32)
        if feat16 is not None:
            p16 = self._proj(feat16)
            B, C, H16, W16 = p16.shape
            seq16 = p16.flatten(2).permute(0, 2, 1)
            seqs.append(seq16)
            shapes['16'] = (H16, W16)

        if len(seqs) == 0:
            raise ValueError('No input features provided to MultiScalePyramidFusion')

        out_seq32 = None
        out_seq16 = None
        if len(seqs) > 1:
            cat = torch.cat(seqs, dim=1)  # (B, S_total, C)
            out_cat = self.global_attn(cat)
            # split back (if we pooled we keep simple split by original sizes up to available tokens)
            ptr = 0
            if '32' in shapes:
                S32 = shapes['32'][0] * shapes['32'][1]
                out_seq32 = out_cat[:, ptr:ptr + S32, :]
                ptr += S32
            if '16' in shapes:
                S16 = shapes['16'][0] * shapes['16'][1]
                out_seq16 = out_cat[:, ptr:ptr + S16, :]
                ptr += S16
        else:
            # only single scale present
            if '32' in shapes:
                out_seq32 = self.global_attn(seqs[0])
            elif '16' in shapes:
                out_seq16 = self.global_attn(seqs[0])

        # reconstruct coarse tensors and upsample+fuse
        fused16 = None
        if out_seq16 is not None:
            B, S16, C = out_seq16.shape
            H16, W16 = shapes['16']
            coarse16 = out_seq16.permute(0, 2, 1).reshape(B, C, H16, W16)
            fused16 = coarse16
        if out_seq32 is not None:
            B, S32, C = out_seq32.shape
            H32, W32 = shapes['32']
            coarse32 = out_seq32.permute(0, 2, 1).reshape(B, C, H32, W32)
            if fused16 is None and '16' in shapes:
                # upsample to 16
                fused16 = F.interpolate(coarse32, size=(shapes['16'][0], shapes['16'][1]), mode='bilinear', align_corners=False)
            else:
                # fuse coarse32 -> fused16 (if fused16 exists) or set fused16=coarse32
                if fused16 is None:
                    fused16 = coarse32
                else:
                    fused16 = self.fuser(coarse32, fused16)

        # now upsample fused16 to /8 and fuse with feat8
        proj8 = None
        fused8 = None
        if feat8 is not None:
            proj8 = self._proj(feat8)
            if fused16 is not None:
                # fuse fused16 (low-res) into feat8 (high-res)
                fused8 = self.fuser(fused16, feat8)
            else:
                fused8 = proj8
        else:
            if fused16 is None:
                raise ValueError('No /8 and no fused16 available')
            # upsample fused16 to /8
            fused8 = F.interpolate(fused16, scale_factor=2, mode='bilinear', align_corners=False)

        # local windowed attention on /8
        B, C, Hk, Wk = fused8.shape
        w = self.window_size
        pad_h = (w - (Hk % w)) % w
        pad_w = (w - (Wk % w)) % w
        if pad_h or pad_w:
            fused8 = F.pad(fused8, (0, pad_w, 0, pad_h))
            Hk_p = Hk + pad_h
            Wk_p = Wk + pad_w
        else:
            Hk_p, Wk_p = Hk, Wk

        # reshape into windows in a stable way
        fused8_reshaped = fused8.view(B, C, Hk_p // w, w, Wk_p // w, w)
        fused8_reshaped = fused8_reshaped.permute(0, 2, 4, 3, 5, 1).contiguous()
        # now shape (B, nH, nW, w, w, C) -> combine windows
        nH = Hk_p // w
        nW = Wk_p // w
        windows = fused8_reshaped.view(B * nH * nW, w * w, C)  # (B*nW*nH, S_w, C)

        # apply local attention per window
        out_windows = self.local_attn(windows)
        # out_windows shape (B*nH*nW, S_w, C)
        out_windows = out_windows.view(B, nH, nW, w, w, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        out_windows = out_windows.view(B, C, nH * w, nW * w)
        # crop to original Hk,Wk if padded
        out = out_windows[:, :, :Hk, :Wk]
        return out


class CrossImageQueryAttn(nn.Module):
    """Cross-image query-sampled attention.
    For each image in a pair, sample N queries (via provided keypoints or random),
    then compute cross-attention q=A_q, k=flatten(B, S, C) of B's feature map.
    """

    def __init__(self, fuser, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.fuser = fuser
        self.d_model = d_model
        self.nhead = nhead
        self.cross_attn = AttentionBlock(d_model=d_model, nhead=nhead)

    def _proj(self, x: torch.Tensor):
        conv = self.fuser._ensure_proj(x.shape[1], f'p{x.shape[1]}', x.device)
        return conv(x)

    def forward(self, featA: torch.Tensor, featB: torch.Tensor, keypointsA: Optional[torch.Tensor] = None, N_query: int = 10, need_weights: bool = False):
        """featA/featB: (B, C, H, W)
        keypointsA: (B, N, 2) normalized coords in [-1,1] for queries; if None sample N random grid points
        returns: outA (B, N, d_model), attn weights (B, heads, N, S_k) if requested
        """
        projA = self._proj(featA)
        projB = self._proj(featB)
        B, C, H, W = projB.shape
        seqB = projB.flatten(2).permute(0, 2, 1)  # (B, S_k, C)

        # keys are the flattened / projected feature map

        # sample queries from A
        if keypointsA is None:
            # sample grid of N points
            N = N_query
            # uniform random in [-1,1]
            keypointsA = torch.rand(B, N, 2, device=featA.device) * 2 - 1

        grid = keypointsA.view(B, keypointsA.shape[1], 1, 2)
        q = F.grid_sample(projA, grid, align_corners=True)  # (B, C, N, 1)
        q = q.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # compute cross-attention: queries from A attend to keys of B
        out, attn = self.cross_attn(q, k=seqB, v=seqB, need_weights=need_weights)
        # out: (B, N, C)
        return out, attn
