"""
BTOT diff module: 把双时相最优传输 (BTOT) 作为变化算子, 接口同 MSCA/HBCA
(forward(pyramid1, pyramid2) -> diff_pyramid), 用于在 train_dinobcd.py 的成熟配方里
替换 HBCA/MSCA, 公平对比 (并复用 SWA/增强/部分解冻等已调好的训练设置)。

每层级: 基础差分特征 + OT 不可匹配质量证据, 用 gamma-gate 残差融合
(gamma 初始为 0 -> 初始退化为纯差分, 安全起步, 借鉴 HBCA)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dinobcd.models.btot import WindowedDustbinOT


class BTOTDiffModule(nn.Module):
    def __init__(self, channels: int, ot_dim: int = 48, window: int = 8,
                 n_iters: int = 5, dropout: float = 0.05,
                 no_gamma: bool = False, cost_only: bool = False,
                 use_dustbin: bool = True, learn_bin: bool = True):
        super().__init__()
        self.no_gamma = no_gamma        # ablation: drop the safe-start gate (gamma fixed at 1)
        self.cost_only = cost_only      # ablation: zero the dustbin unmatchable-mass channel
        # 基础差分分支: [|f1-f2|; (f1+f2)/2] -> C
        self.base = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # 窗口化 dustbin-OT, 产生 2 通道变化证据 (不可匹配质量 + 传输代价残差)
        self.ot = WindowedDustbinOT(channels, ot_dim=ot_dim, window=window, n_iters=n_iters,
                                    use_dustbin=use_dustbin, learn_bin=learn_bin)
        self.ot_enc = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # 融合 + 深度可分离精修
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
        # gamma-gate 残差: 初始 0 -> 初始为纯差分; 训练中逐步引入 OT 证据
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        base = self.base(torch.cat([(f1 - f2).abs(), 0.5 * (f1 + f2)], dim=1))
        ot_field = self.ot(f1, f2)                 # [B, 2, H, W] = [unmatchable mass, cost residual]
        if self.cost_only:                         # ablation: remove dustbin unmatchable-mass evidence
            ot_field = ot_field.clone(); ot_field[:, 0] = 0
        ot_feat = self.ot_enc(ot_field)
        fused = self.fuse(torch.cat([base, ot_feat], dim=1))
        gate = 1.0 if self.no_gamma else self.gamma
        return base + gate * fused


class MultiScaleBTOTDiff(nn.Module):
    """多尺度 BTOT 差异建模, 接口同 MultiScaleMSCA / MultiScaleDualDifference."""

    def __init__(self, channels_list, ot_dim: int = 48, window: int = 8,
                 n_iters: int = 5, dropout: float = 0.05,
                 no_gamma: bool = False, cost_only: bool = False,
                 use_dustbin: bool = True, learn_bin: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            BTOTDiffModule(c, ot_dim=ot_dim, window=window, n_iters=n_iters, dropout=dropout,
                           no_gamma=no_gamma, cost_only=cost_only,
                           use_dustbin=use_dustbin, learn_bin=learn_bin)
            for c in channels_list
        ])

    def forward(self, feats1_pyramid, feats2_pyramid):
        assert len(feats1_pyramid) == len(feats2_pyramid)
        return [self.blocks[i](f1, f2) for i, (f1, f2) in enumerate(zip(feats1_pyramid, feats2_pyramid))]
