"""
DINOBCD - Pyramid Feature Fusion Module
金字塔特征融合 - 融合CNN和DINOv3特征

核心创新：
1. 双模态特征对齐
2. 自适应通道注意力融合
3. 自顶向下的特征细化
4. 可选：自适应特征融合(AFF)模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .adaptive_fusion import AdaptiveFeatureFusion
    AFF_AVAILABLE = True
except ImportError:
    AFF_AVAILABLE = False


class ChannelAttention(nn.Module):
    """
    通道注意力模块

    使用全局池化和MLP生成通道权重
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        # 融合
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    空间注意力模块

    使用通道池化生成空间权重
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度的平均和最大
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并卷积
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)

        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module

    串联通道注意力和空间注意力
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class DualModalFusion(nn.Module):
    """
    双模态特征融合

    融合CNN特征和DINOv3特征

    Args:
        use_aff: 是否使用自适应特征融合(AFF)模块（创新）
    """

    def __init__(self, cnn_channels, dino_channels, out_channels, use_aff=False):
        super().__init__()

        self.use_aff = use_aff and AFF_AVAILABLE

        # 通道对齐
        self.cnn_proj = nn.Sequential(
            nn.Conv2d(cnn_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        if self.use_aff:
            # 使用AFF模块（创新方案）
            self.fusion = AdaptiveFeatureFusion(
                channels=out_channels,
                reduction=8,
                use_spatial=True
            )
        else:
            # 原始CBAM方案
            self.cbam = CBAM(out_channels * 2, reduction=8)

            # 融合卷积
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, cnn_feat, dino_feat):
        """
        Args:
            cnn_feat: CNN特征 [B, C_cnn, H, W]
            dino_feat: DINOv3特征 [B, C_dino, H, W]

        Returns:
            fused_feat: 融合特征 [B, C_out, H, W]
        """
        # 确保尺寸一致
        if cnn_feat.shape[-2:] != dino_feat.shape[-2:]:
            dino_feat = F.interpolate(
                dino_feat,
                size=cnn_feat.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # 通道对齐
        cnn_feat = self.cnn_proj(cnn_feat)
        dino_feat = self.dino_proj(dino_feat)

        if self.use_aff:
            # 使用AFF模块融合（创新方案）
            fused_feat = self.fusion(cnn_feat, dino_feat)
        else:
            # 原始CBAM方案
            # 拼接
            concat_feat = torch.cat([cnn_feat, dino_feat], dim=1)

            # CBAM注意力
            concat_feat = self.cbam(concat_feat)

            # 融合
            fused_feat = self.fusion_conv(concat_feat)

        return fused_feat


class PyramidFusionModule(nn.Module):
    """
    金字塔特征融合模块

    将CNN和DINOv3的多尺度特征融合成统一的特征金字塔
    """

    def __init__(
        self,
        cnn_channels=[64, 128, 256, 512],
        dino_channels=[256, 256, 256, 256],
        out_channels=128,
        use_aff=False
    ):
        super().__init__()

        assert len(cnn_channels) == len(dino_channels) == 4

        # 为每个层级创建融合模块
        self.fusion_layers = nn.ModuleList([
            DualModalFusion(cnn_ch, dino_ch, out_channels, use_aff=use_aff)
            for cnn_ch, dino_ch in zip(cnn_channels, dino_channels)
        ])

    def forward(self, cnn_pyramid, dino_pyramid):
        """
        Args:
            cnn_pyramid: CNN特征金字塔 [c1, c2, c3, c4]
            dino_pyramid: DINOv3特征金字塔 [d1, d2, d3, d4]

        Returns:
            fused_pyramid: 融合金字塔 [p1, p2, p3, p4]
        """
        assert len(cnn_pyramid) == len(dino_pyramid) == 4

        fused_pyramid = []
        for i, (cnn_feat, dino_feat) in enumerate(zip(cnn_pyramid, dino_pyramid)):
            fused = self.fusion_layers[i](cnn_feat, dino_feat)
            fused_pyramid.append(fused)

        return fused_pyramid


class TopDownRefinement(nn.Module):
    """
    自顶向下的特征细化

    从最粗糙的层级开始，逐步上采样并融合
    """

    def __init__(self, channels=128):
        super().__init__()

        # 上采样融合模块
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1)
            for _ in range(3)  # P4->P3, P3->P2, P2->P1
        ])

        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(3)
        ])

    def forward(self, pyramid):
        """
        Args:
            pyramid: [P1, P2, P3, P4] 从细到粗

        Returns:
            refined_pyramid: 细化后的金字塔
        """
        # 自顶向下处理
        p4 = pyramid[3]  # 最粗糙的层级

        # P4 -> P3
        p3 = pyramid[2]
        p3_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral_convs[0](p3) + p3_up
        p3 = self.output_convs[0](p3)

        # P3 -> P2
        p2 = pyramid[1]
        p2_up = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_convs[1](p2) + p2_up
        p2 = self.output_convs[1](p2)

        # P2 -> P1
        p1 = pyramid[0]
        p1_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral_convs[2](p1) + p1_up
        p1 = self.output_convs[2](p1)

        return [p1, p2, p3, p4]


class EnhancedPyramidFusion(nn.Module):
    """
    增强的金字塔融合模块

    结合双模态融合和自顶向下细化
    """

    def __init__(
        self,
        cnn_channels=[64, 128, 256, 512],
        dino_channels=[256, 256, 256, 256],
        out_channels=128,
        use_aff=False
    ):
        super().__init__()

        # 双模态融合
        self.dual_fusion = PyramidFusionModule(
            cnn_channels, dino_channels, out_channels, use_aff=use_aff
        )

        # 自顶向下细化
        self.top_down = TopDownRefinement(out_channels)

    def forward(self, cnn_pyramid, dino_pyramid):
        """
        Args:
            cnn_pyramid: CNN金字塔
            dino_pyramid: DINOv3金字塔

        Returns:
            refined_pyramid: 融合并细化的金字塔
        """
        # 双模态融合
        fused = self.dual_fusion(cnn_pyramid, dino_pyramid)

        # 自顶向下细化
        refined = self.top_down(fused)

        return refined


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Testing DualModalFusion")
    print("=" * 60)

    fusion = DualModalFusion(
        cnn_channels=128,
        dino_channels=256,
        out_channels=128
    ).to(device)

    cnn_feat = torch.randn(2, 128, 32, 32).to(device)
    dino_feat = torch.randn(2, 256, 32, 32).to(device)

    fused = fusion(cnn_feat, dino_feat)
    print(f"CNN: {cnn_feat.shape}, DINO: {dino_feat.shape}")
    print(f"Fused: {fused.shape}")

    print("\n" + "=" * 60)
    print("Testing EnhancedPyramidFusion")
    print("=" * 60)

    pyramid_fusion = EnhancedPyramidFusion(
        cnn_channels=[64, 128, 256, 512],
        dino_channels=[256, 256, 256, 256],
        out_channels=128
    ).to(device)

    cnn_pyramid = [
        torch.randn(2, 64, 64, 64).to(device),
        torch.randn(2, 128, 32, 32).to(device),
        torch.randn(2, 256, 16, 16).to(device),
        torch.randn(2, 512, 8, 8).to(device)
    ]

    dino_pyramid = [
        torch.randn(2, 256, 64, 64).to(device),
        torch.randn(2, 256, 32, 32).to(device),
        torch.randn(2, 256, 16, 16).to(device),
        torch.randn(2, 256, 8, 8).to(device)
    ]

    refined = pyramid_fusion(cnn_pyramid, dino_pyramid)

    print("Refined Pyramid:")
    for i, feat in enumerate(refined):
        print(f"  P{i+1}: {feat.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in pyramid_fusion.parameters())
    print(f"\nTotal parameters: {total_params:,}")
