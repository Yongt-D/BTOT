"""
DINOBCD - Lightweight DINOv3 Feature Adapter
完全独立实现

核心创新：
1. 使用深度可分离卷积降低参数量
2. 瓶颈设计压缩特征维度
3. 多尺度特征对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块

    将标准卷积分解为：
    1. Depthwise: 每个通道独立卷积
    2. Pointwise: 1x1卷积混合通道

    参数量: O(K²·C + C·C') vs 标准卷积 O(K²·C·C')
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        activation=nn.GELU
    ):
        super().__init__()

        # Depthwise卷积 - 每个通道独立
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,  # 关键：groups=in_channels
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            activation()
        )

        # Pointwise卷积 - 1x1混合通道
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BottleneckAdapter(nn.Module):
    """
    瓶颈适配器 - 压缩-处理-扩展架构

    DINOv3 (1024D) -> 压缩 (bottleneck) -> 处理 -> 扩展 (out_dim)
    """

    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        bottleneck_dim=64,
        activation=nn.GELU
    ):
        super().__init__()

        # 1. 降维到瓶颈
        self.compress = nn.Sequential(
            nn.Conv2d(in_dim, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            activation()
        )

        # 2. 瓶颈处理 - 使用深度可分离卷积
        self.process = DepthwiseSeparableConv(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            padding=1,
            activation=activation
        )

        # 3. 升维到输出维度
        self.expand = nn.Conv2d(
            bottleneck_dim,
            out_dim,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.compress(x)
        x = self.process(x)
        x = self.expand(x)
        return x


class MultiScaleAdapter(nn.Module):
    """
    多尺度特征适配器

    将DINOv3的4层特征适配到指定的金字塔尺度

    Args:
        in_dim: DINOv3特征维度（1024 for ViT-L）
        out_dim: 输出特征维度
        target_sizes: 目标尺度列表，如[64, 32, 16, 8]
        bottleneck_dim: 瓶颈维度
        shared: 是否共享适配器权重
    """

    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        target_sizes=[64, 32, 16, 8],
        bottleneck_dim=64,
        shared=False
    ):
        super().__init__()

        self.target_sizes = target_sizes
        self.shared = shared

        if shared:
            # 共享权重 - 所有尺度使用同一个适配器
            self.adapters = nn.ModuleList([
                BottleneckAdapter(in_dim, out_dim, bottleneck_dim)
            ])
        else:
            # 独立适配器 - 每个尺度独立
            self.adapters = nn.ModuleList([
                BottleneckAdapter(in_dim, out_dim, bottleneck_dim)
                for _ in target_sizes
            ])

    def forward(self, dino_features):
        """
        Args:
            dino_features: DINOv3特征列表，每个 [B, 1024, 32, 32]

        Returns:
            adapted_features: 适配后的特征列表，每个 [B, out_dim, size, size]
        """
        outputs = []

        for i, feat in enumerate(dino_features):
            # 1. 调整到目标尺寸
            target_size = self.target_sizes[i]
            if feat.shape[-1] != target_size:
                feat = F.interpolate(
                    feat,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True
                )

            # 2. 通过适配器
            adapter = self.adapters[0] if self.shared else self.adapters[i]
            feat_adapted = adapter(feat)

            outputs.append(feat_adapted)

        return outputs


class DualPathAdapter(nn.Module):
    """
    双路径适配器 - DINOBCD创新点

    设计思想：
    1. 快速路径：直接降维（保留高频细节）
    2. 慢速路径：瓶颈处理（提取语义特征）
    3. 自适应融合
    """

    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        bottleneck_dim=64
    ):
        super().__init__()

        # 快速路径 - 直接降维
        self.fast_path = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )

        # 慢速路径 - 瓶颈处理
        self.slow_path = BottleneckAdapter(in_dim, out_dim, bottleneck_dim)

        # 自适应融合权重
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 输出卷积
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        # 两条路径
        fast_feat = self.fast_path(x)  # 保留细节
        slow_feat = self.slow_path(x)  # 语义特征

        # 自适应融合
        concat_feat = torch.cat([fast_feat, slow_feat], dim=1)
        gate = self.fusion_gate(concat_feat)

        # 门控融合
        fused = gate * fast_feat + (1 - gate) * slow_feat

        # 输出细化
        out = self.output_conv(fused)

        return out


class MultiScaleDualAdapter(nn.Module):
    """
    多尺度双路径适配器 - DINOBCD的核心适配器
    """

    def __init__(
        self,
        in_dim=1024,
        out_dim=256,
        target_sizes=[64, 32, 16, 8],
        bottleneck_dim=64
    ):
        super().__init__()

        self.target_sizes = target_sizes

        # 为每个尺度创建双路径适配器
        self.adapters = nn.ModuleList([
            DualPathAdapter(in_dim, out_dim, bottleneck_dim)
            for _ in target_sizes
        ])

    def forward(self, dino_features):
        """
        Args:
            dino_features: DINOv3特征列表

        Returns:
            adapted_features: 适配后的特征列表
        """
        outputs = []

        for i, feat in enumerate(dino_features):
            # 调整到目标尺寸
            target_size = self.target_sizes[i]
            if feat.shape[-1] != target_size:
                feat = F.interpolate(
                    feat,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True
                )

            # 通过双路径适配器
            feat_adapted = self.adapters[i](feat)
            outputs.append(feat_adapted)

        return outputs


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试多尺度双路径适配器
    adapter = MultiScaleDualAdapter(
        in_dim=1024,
        out_dim=256,
        target_sizes=[64, 32, 16, 8],
        bottleneck_dim=64
    ).to(device)

    # 模拟DINOv3特征
    dino_feats = [
        torch.randn(2, 1024, 32, 32).to(device) for _ in range(4)
    ]

    # 前向传播
    adapted_feats = adapter(dino_feats)

    print("MultiScaleDualAdapter Test:")
    for i, feat in enumerate(adapted_feats):
        print(f"  Output {i}: {feat.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in adapter.parameters())
    print(f"\nTotal parameters: {total_params:,}")
