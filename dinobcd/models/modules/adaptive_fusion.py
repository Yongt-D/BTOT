"""
Adaptive Feature Fusion (AFF) Module
自适应特征融合模块

创新点：
1. 双分支注意力机制（空间+通道）
2. 自适应融合CNN和DINOv3特征
3. 轻量级设计（参数量<1M）
4. 即插即用，不改变主体架构

预期提升：+0.5-0.8% IoU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (Squeeze-and-Excitation风格)

    学习"融合什么特征"
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化 + 最大池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # 融合
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """
    空间注意力模块

    学习"在哪里融合特征"（边界 vs 内部）
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

        # 拼接
        concat = torch.cat([avg_out, max_out], dim=1)

        # 空间注意力图
        out = self.sigmoid(self.conv(concat))
        return x * out


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块 (核心创新)

    将CNN和DINOv3特征自适应融合：
    1. 通道注意力：选择重要的特征通道
    2. 空间注意力：选择重要的空间位置
    3. 可学习的融合权重：动态平衡两种特征

    Args:
        channels: 特征通道数
        reduction: 通道注意力的降维比例
        use_spatial: 是否使用空间注意力
    """
    def __init__(self, channels=128, reduction=8, use_spatial=True):
        super().__init__()

        self.use_spatial = use_spatial

        # 通道注意力
        self.channel_attn = ChannelAttention(channels * 2, reduction)

        # 空间注意力
        if use_spatial:
            self.spatial_attn = SpatialAttention(kernel_size=7)

        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        # 输出投影
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, cnn_feat, dino_feat):
        """
        Args:
            cnn_feat: CNN特征 [B, C, H, W]
            dino_feat: DINOv3特征 [B, C, H, W]

        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 1. 拼接两个特征
        concat = torch.cat([cnn_feat, dino_feat], dim=1)

        # 2. 通道注意力：学习每个通道的重要性
        concat = self.channel_attn(concat)

        # 3. 分离
        cnn_refined, dino_refined = torch.chunk(concat, 2, dim=1)

        # 4. 空间注意力（可选）
        if self.use_spatial:
            # 初步融合
            combined = cnn_refined + dino_refined

            # 空间注意力
            combined = self.spatial_attn(combined)

            # 加权融合
            w = torch.softmax(self.fusion_weight, dim=0)
            output = w[0] * cnn_refined + w[1] * dino_refined + combined
        else:
            # 仅加权融合
            w = torch.softmax(self.fusion_weight, dim=0)
            output = w[0] * cnn_refined + w[1] * dino_refined

        # 5. 输出投影
        output = self.output_conv(output)

        return output


class BoundaryAwareAttention(nn.Module):
    """
    边界感知注意力模块 (可选创新)

    自动检测边界区域并增强边界特征
    """
    def __init__(self, channels=128):
        super().__init__()

        # Sobel边界检测算子
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).view(1, 1, 3, 3))

        # 边界特征增强
        self.boundary_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def detect_boundary(self, x):
        """
        使用Sobel算子检测边界

        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            边界图 [B, 1, H, W]
        """
        # 对所有通道求平均
        x_gray = x.mean(dim=1, keepdim=True)

        # Sobel边界检测
        edge_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        # 归一化到[0, 1]
        edge = edge / (edge.max() + 1e-6)

        return edge

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]

        Returns:
            边界增强后的特征 [B, C, H, W]
        """
        # 1. 检测边界
        boundary_map = self.detect_boundary(x)

        # 2. 边界特征增强
        boundary_feat = self.boundary_enhance(x)

        # 3. 门控融合
        concat = torch.cat([x, boundary_feat], dim=1)
        gate = self.gate(concat)

        # 4. 边界区域使用增强特征，内部区域保持原始特征
        output = x + gate * boundary_feat * boundary_map

        return output


if __name__ == "__main__":
    """测试代码"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试AFF
    aff = AdaptiveFeatureFusion(channels=128).to(device)
    cnn_feat = torch.randn(2, 128, 32, 32).to(device)
    dino_feat = torch.randn(2, 128, 32, 32).to(device)

    output = aff(cnn_feat, dino_feat)
    print(f"AFF Test:")
    print(f"  Input:  CNN {cnn_feat.shape} + DINO {dino_feat.shape}")
    print(f"  Output: {output.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in aff.parameters())
    print(f"  Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")

    # 测试BAA
    print(f"\nBAA Test:")
    baa = BoundaryAwareAttention(channels=128).to(device)
    feat = torch.randn(2, 128, 32, 32).to(device)
    output = baa(feat)
    print(f"  Input:  {feat.shape}")
    print(f"  Output: {output.shape}")

    total_params = sum(p.numel() for p in baa.parameters())
    print(f"  Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
