"""
v12: Bi-Temporal Cross-Modal Attention Fusion (BT-CBAM)

对 ChangeDINO 的 DFFM (DW-conv + CBAM 单时相注意力) 的关键升级:
- CBAM 的 channel/spatial attention 显式以**对方时相**的全局统计作 conditioning
- 即 T1 的 channel attention 不只看 T1 自己, 还看 T2 的 max-pool
- 这让"双流融合"本身变成 bi-temporal aware, 是 DFFM 没有的差异化

模块输入: T1/T2 各自的 CNN backbone 特征 + DINOv3 适配特征
模块输出: 融合后的 T1/T2 金字塔特征 (per-level)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiTemporalCBAM(nn.Module):
    """跨时相 CBAM: T1 的 channel/spatial attention 由 T2 全局统计 condition,
    反之亦然. 与单时相 CBAM (ChangeDINO 用的) 的区别是 attention 信号本身就
    含 bi-temporal 信息."""

    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        # Channel attention MLP, 输入 [avg(self); max(other)] 共 2C 维
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        # Spatial attention: 输入 [avg(self), max(other)] 各 1 通道 = 2
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def _channel_attn(self, x_self, x_other):
        avg_self = F.adaptive_avg_pool2d(x_self, 1)
        max_other = F.adaptive_max_pool2d(x_other, 1)
        gate = torch.sigmoid(self.ca_mlp(torch.cat([avg_self, max_other], dim=1)))
        return x_self * gate

    def _spatial_attn(self, x_self, x_other):
        avg_self = x_self.mean(dim=1, keepdim=True)
        max_other = x_other.max(dim=1, keepdim=True).values
        gate = torch.sigmoid(self.sa_conv(torch.cat([avg_self, max_other], dim=1)))
        return x_self * gate

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: [B, C, H, W] 双时相特征 (融合后)
        Returns:
            (x1_attn, x2_attn): 经过跨时相 CBAM 加权的特征
        """
        # Bidirectional channel attention
        x1_ca = self._channel_attn(x1, x2)
        x2_ca = self._channel_attn(x2, x1)
        # Bidirectional spatial attention (用 CA 后的特征)
        x1_out = self._spatial_attn(x1_ca, x2_ca)
        x2_out = self._spatial_attn(x2_ca, x1_ca)
        return x1_out, x2_out


class DualStreamFusion(nn.Module):
    """单 FPN level 的双流融合: CNN backbone feat + DINOv3 adapted feat → 融合特征.
    流程:
      [CNN; DINO] (concat 2C) → DW-separable conv → BN+ReLU → BT-CBAM (跨时相)
    """

    def __init__(self, channels):
        super().__init__()
        # DW-separable conv 把 2C concat 压回 C
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1,
                      groups=channels * 2, bias=False),  # depthwise
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.bt_cbam = BiTemporalCBAM(channels)

    def forward(self, cnn1, cnn2, dino1, dino2):
        """
        Args:
            cnn1, cnn2: [B, C, H, W] CNN backbone 当 level 输出 (已投影到 C)
            dino1, dino2: [B, C, H, W] DINOv3 适配后的当 level 特征
        Returns:
            (f1, f2): 融合后特征, 跨时相 attention 加权
        """
        # Resize DINO feature to match CNN spatial size (if mismatched)
        if dino1.shape[-2:] != cnn1.shape[-2:]:
            dino1 = F.interpolate(dino1, size=cnn1.shape[-2:],
                                  mode='bilinear', align_corners=False)
            dino2 = F.interpolate(dino2, size=cnn2.shape[-2:],
                                  mode='bilinear', align_corners=False)
        f1 = self.fuse(torch.cat([cnn1, dino1], dim=1))
        f2 = self.fuse(torch.cat([cnn2, dino2], dim=1))
        f1, f2 = self.bt_cbam(f1, f2)
        return f1, f2


class MultiScaleBiTemporalCBAM(nn.Module):
    """对已有的双流融合金字塔 (pyramid1, pyramid2) 加一层 BT-CBAM 跨时相 attention.
    使用场景: v6 已有 EnhancedPyramidFusion (per-time fusion), v12 在其后接此模块,
    让金字塔特征再吸收 bi-temporal 信息. 这是与 ChangeDINO DFFM (单时相 CBAM) 的关键差异."""

    def __init__(self, channels, num_levels=4, reduction=8):
        super().__init__()
        self.levels = nn.ModuleList([
            BiTemporalCBAM(channels, reduction=reduction) for _ in range(num_levels)
        ])

    def forward(self, pyramid1, pyramid2):
        out1, out2 = [], []
        for i, m in enumerate(self.levels):
            f1, f2 = m(pyramid1[i], pyramid2[i])
            out1.append(f1)
            out2.append(f2)
        return out1, out2


class MultiScaleDualStreamFusion(nn.Module):
    """4 个 FPN level 各一个 DualStreamFusion."""

    def __init__(self, channels, num_levels=4):
        super().__init__()
        self.levels = nn.ModuleList([
            DualStreamFusion(channels) for _ in range(num_levels)
        ])
        self.num_levels = num_levels

    def forward(self, cnn_pyr1, cnn_pyr2, dino_pyr1, dino_pyr2):
        """
        Args:
            cnn_pyr1, cnn_pyr2: list of [B, C, H, W] CNN 金字塔 (4 个 level)
            dino_pyr1, dino_pyr2: list of [B, C, H, W] DINOv3 金字塔 (4 个 level)
        Returns:
            (fused_pyr1, fused_pyr2): 融合后金字塔
        """
        out1, out2 = [], []
        for i, m in enumerate(self.levels):
            f1, f2 = m(cnn_pyr1[i], cnn_pyr2[i], dino_pyr1[i], dino_pyr2[i])
            out1.append(f1)
            out2.append(f2)
        return out1, out2
