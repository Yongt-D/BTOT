"""
SSM diff module: a state-space (Mamba/S6-style) bi-temporal interaction operator,
same interface as BTOT/HBCA/MSCA (forward(pyramid1, pyramid2) -> diff_pyramid), so it
can be dropped into the train_dinobcd.py recipe for a *fair* controlled comparison of
interaction operators (difference / cross-attention / state-space / optimal transport).

Pure PyTorch (no mamba-ssm / causal-conv1d CUDA kernels). A selective diagonal SSM is
scanned bidirectionally along rows and columns (VMamba-style cross-scan) over the
concatenated bi-temporal features; the scan length is only H or W (the other axis is
folded into the batch), so the sequential recurrence stays cheap.

Per level: base difference feature + state-space change evidence, fused by a gamma-gate
residual (gamma init 0 -> starts as pure difference; safe start, same as BTOT/HBCA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM1D(nn.Module):
    """Minimal selective diagonal state-space scan over the last dim.

    Input  x: [B, C, L]
    Output y: [B, C, L]

    Diagonal recurrence with input-dependent (selective) step size:
        dt   = softplus(W_dt x + b_dt)            (per channel, per position)
        a_t  = exp(dt * (-exp(A_log)))            in (0,1), stable decay
        h_t  = a_t * h_{t-1} + dt * B_t * x_t
        y_t  = C_t * h_t + D * x_t
    B_t, C_t are input-dependent scalars (shared across channels) -> selectivity.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.dt_proj = nn.Conv1d(channels, channels, 1)
        self.bc_proj = nn.Conv1d(channels, 2, 1)          # B_t, C_t (selective, scalar)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, channels + 1).float()))
        self.D = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        dt = F.softplus(self.dt_proj(x))                  # [B,C,L] > 0
        a = torch.exp(dt * (-torch.exp(self.A_log).view(1, C, 1)))   # [B,C,L] in (0,1)
        bc = self.bc_proj(x)                              # [B,2,L]
        Bt = bc[:, 0:1, :]                                # [B,1,L]
        Ct = bc[:, 1:2, :]                                # [B,1,L]
        bx = dt * Bt * x                                  # [B,C,L]
        h = x.new_zeros(B, C)
        ys = []
        for t in range(L):
            h = a[:, :, t] * h + bx[:, :, t]
            ys.append(h)
        y = torch.stack(ys, dim=2)                        # [B,C,L]
        return Ct * y + self.D.view(1, C, 1) * x


class _CrossScan2D(nn.Module):
    """Bidirectional selective SSM along rows and columns of an [B,C,H,W] map."""

    def __init__(self, channels: int):
        super().__init__()
        self.scan = SelectiveSSM1D(channels)

    def _dir(self, x):  # x: [B,C,H,W] -> scan along W (rows folded into batch), bi-dir
        B, C, H, W = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        fwd = self.scan(seq)
        bwd = torch.flip(self.scan(torch.flip(seq, dims=[2])), dims=[2])
        out = (fwd + bwd).reshape(B, H, C, W).permute(0, 2, 1, 3)
        return out

    def forward(self, x):
        # gradient-checkpoint the sequential scans: the recurrence stores L states per
        # direction, so recomputing them in backward cuts activation memory a lot.
        if self.training and x.requires_grad:
            from torch.utils.checkpoint import checkpoint
            row = checkpoint(self._dir, x, use_reentrant=False)
            col = checkpoint(lambda t: self._dir(t.transpose(2, 3)).transpose(2, 3), x, use_reentrant=False)
        else:
            row = self._dir(x)                                  # scan along W
            col = self._dir(x.transpose(2, 3)).transpose(2, 3)  # scan along H
        return row + col


class SSMDiffModule(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        # base difference branch (identical to BTOT's, for a fair safe start)
        self.base = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # state-space branch over concatenated bi-temporal features
        self.in_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.ssm = _CrossScan2D(channels)
        self.ssm_enc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        base = self.base(torch.cat([(f1 - f2).abs(), 0.5 * (f1 + f2)], dim=1))
        x = self.in_proj(torch.cat([f1, f2], dim=1))
        ssm_feat = self.ssm_enc(self.ssm(x))
        fused = self.fuse(torch.cat([base, ssm_feat], dim=1))
        return base + self.gamma * fused


class MultiScaleSSMDiff(nn.Module):
    """Multi-scale state-space diff modeling, same interface as MultiScaleBTOTDiff."""

    def __init__(self, channels_list, dropout: float = 0.05):
        super().__init__()
        self.blocks = nn.ModuleList([SSMDiffModule(c, dropout=dropout) for c in channels_list])

    def forward(self, feats1_pyramid, feats2_pyramid):
        assert len(feats1_pyramid) == len(feats2_pyramid)
        return [self.blocks[i](f1, f2) for i, (f1, f2) in enumerate(zip(feats1_pyramid, feats2_pyramid))]
