"""
DINOBCD - Efficient CNN Backbone
轻量级骨干网络 - 使用EfficientNet-Lite风格设计

采用MBConv模块构建，具有良好的性能/参数量平衡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    结构:
    1. Expansion: 1x1 conv扩展通道
    2. Depthwise: 3x3深度卷积
    3. SE: Squeeze-and-Excitation注意力
    4. Projection: 1x1 conv压缩通道
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio=4,
        stride=1,
        se_ratio=0.25
    ):
        super().__init__()

        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = int(in_channels * expand_ratio)

        # 1. Expansion
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand = nn.Identity()

        # 2. Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=3, stride=stride, padding=1,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # 3. Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # 4. Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        # Expansion
        x = self.expand(x)

        # Depthwise
        x = self.depthwise(x)

        # SE attention
        if self.se is not None:
            x = x * self.se(x)

        # Projection
        x = self.project(x)

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x


class EfficientBackbone(nn.Module):
    """
    Efficient CNN Backbone for DINOBCD

    输出4个层次的特征:
    - Stage 1: 1/4,  C1=64
    - Stage 2: 1/8,  C2=128
    - Stage 3: 1/16, C3=256
    - Stage 4: 1/32, C4=512

    总参数量: ~4-5M (与MobileNetV2相当)
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        stage_channels=[64, 128, 256, 512],
        stage_depths=[2, 3, 4, 3]
    ):
        super().__init__()

        self.stage_channels = stage_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )

        # Stage 1: 1/4
        self.stage1 = self._make_stage(
            base_channels, stage_channels[0],
            num_blocks=stage_depths[0],
            stride=2
        )

        # Stage 2: 1/8
        self.stage2 = self._make_stage(
            stage_channels[0], stage_channels[1],
            num_blocks=stage_depths[1],
            stride=2
        )

        # Stage 3: 1/16
        self.stage3 = self._make_stage(
            stage_channels[1], stage_channels[2],
            num_blocks=stage_depths[2],
            stride=2
        )

        # Stage 4: 1/32
        self.stage4 = self._make_stage(
            stage_channels[2], stage_channels[3],
            num_blocks=stage_depths[3],
            stride=2
        )

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """创建一个stage"""
        blocks = []

        # 第一个block进行下采样
        blocks.append(
            MBConvBlock(in_channels, out_channels, stride=stride)
        )

        # 剩余的blocks
        for _ in range(num_blocks - 1):
            blocks.append(
                MBConvBlock(out_channels, out_channels, stride=1)
            )

        return nn.Sequential(*blocks)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            features: List of [B, Ci, H/si, W/si]
                     s1=4, s2=8, s3=16, s4=32
        """
        # Stem
        x = self.stem(x)  # 1/2

        # Multi-scale features
        c1 = self.stage1(x)   # 1/4
        c2 = self.stage2(c1)  # 1/8
        c3 = self.stage3(c2)  # 1/16
        c4 = self.stage4(c3)  # 1/32

        return [c1, c2, c3, c4]


class LightweightBackbone(nn.Module):
    """
    更轻量级的backbone (< 2M params)

    适用于显存受限的场景
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=24,
        stage_channels=[48, 96, 192, 320],
        stage_depths=[2, 2, 3, 2]
    ):
        super().__init__()

        self.stage_channels = stage_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )

        # Stages
        self.stage1 = self._make_stage(base_channels, stage_channels[0], stage_depths[0], stride=2)
        self.stage2 = self._make_stage(stage_channels[0], stage_channels[1], stage_depths[1], stride=2)
        self.stage3 = self._make_stage(stage_channels[1], stage_channels[2], stage_depths[2], stride=2)
        self.stage4 = self._make_stage(stage_channels[2], stage_channels[3], stage_depths[3], stride=2)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        blocks = []
        blocks.append(MBConvBlock(in_channels, out_channels, expand_ratio=3, stride=stride))
        for _ in range(num_blocks - 1):
            blocks.append(MBConvBlock(out_channels, out_channels, expand_ratio=3, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return [c1, c2, c3, c4]


def build_backbone(backbone_type="efficient", **kwargs):
    """
    构建backbone

    Args:
        backbone_type: 'efficient' or 'lightweight'
    """
    if backbone_type == "efficient":
        return EfficientBackbone(**kwargs)
    elif backbone_type == "lightweight":
        return LightweightBackbone(**kwargs)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Testing EfficientBackbone")
    print("=" * 60)

    backbone = EfficientBackbone().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)

    features = backbone(x)
    for i, feat in enumerate(features):
        print(f"Stage {i+1}: {feat.shape}")

    # 计算参数量和FLOPs
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    print("\n" + "=" * 60)
    print("Testing LightweightBackbone")
    print("=" * 60)

    lightweight = LightweightBackbone().to(device)
    features = lightweight(x)
    for i, feat in enumerate(features):
        print(f"Stage {i+1}: {feat.shape}")

    total_params = sum(p.numel() for p in lightweight.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
