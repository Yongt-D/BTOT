"""
DINOBCD - Dual Difference Modeling Module
双重差异建模模块

核心思想：
1. 变化差异流（Change Stream）：捕获时序变化 |T1 - T2|
2. 不变差异流（Invariant Stream）：捕获语义相似度 T1 ⊗ T2
3. 自适应融合：动态加权两种差异表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeStream(nn.Module):
    """
    变化差异流 - 捕获时序变化

    使用绝对差 + 卷积处理来增强变化特征
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: 时间1特征 [B, C, H, W]
            feat2: 时间2特征 [B, C, H, W]

        Returns:
            change_feat: 变化特征 [B, C_out, H, W]
        """
        # 计算绝对差
        diff = torch.abs(feat1 - feat2)

        # 卷积增强
        change_feat = self.conv(diff)

        return change_feat


class InvariantStream(nn.Module):
    """
    不变差异流 - 捕获语义相似度

    使用点乘（Hadamard product）来建模语义相关性
    高相似度区域（未变化）会有高响应
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: 时间1特征 [B, C, H, W]
            feat2: 时间2特征 [B, C, H, W]

        Returns:
            invariant_feat: 不变特征 [B, C_out, H, W]
        """
        # 计算点乘（Hadamard product）
        product = feat1 * feat2

        # 卷积处理
        invariant_feat = self.conv(product)

        return invariant_feat


class AdaptiveFusionGate(nn.Module):
    """
    自适应融合门控

    学习如何动态融合变化流和不变流
    """

    def __init__(self, channels):
        super().__init__()

        # 使用通道注意力生成融合权重
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, kernel_size=1),  # 2个权重：change和invariant
            nn.Softmax(dim=1)  # 归一化权重
        )

    def forward(self, change_feat, invariant_feat):
        """
        Args:
            change_feat: 变化特征 [B, C, H, W]
            invariant_feat: 不变特征 [B, C, H, W]

        Returns:
            fused_feat: 融合特征 [B, C, H, W]
        """
        # 拼接两种特征
        concat_feat = torch.cat([change_feat, invariant_feat], dim=1)

        # 生成融合权重 [B, 2, 1, 1]
        weights = self.gate_conv(concat_feat)

        # 分离权重
        w_change = weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        w_invariant = weights[:, 1:2, :, :]  # [B, 1, 1, 1]

        # 自适应融合
        fused = w_change * change_feat + w_invariant * invariant_feat

        return fused


class DualDifferenceModule(nn.Module):
    """
    双重差异建模模块 - 整合变化流和不变流

    这是DINOBCD的核心创新之一
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 变化流
        self.change_stream = ChangeStream(in_channels, out_channels)

        # 不变流
        self.invariant_stream = InvariantStream(in_channels, out_channels)

        # 自适应融合
        self.fusion_gate = AdaptiveFusionGate(out_channels)

        # 输出细化
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: T1时刻特征 [B, C_in, H, W]
            feat2: T2时刻特征 [B, C_in, H, W]

        Returns:
            diff_feat: 差异特征 [B, C_out, H, W]
        """
        # 1. 计算变化流
        change_feat = self.change_stream(feat1, feat2)

        # 2. 计算不变流
        invariant_feat = self.invariant_stream(feat1, feat2)

        # 3. 自适应融合
        fused_feat = self.fusion_gate(change_feat, invariant_feat)

        # 4. 输出细化
        diff_feat = self.refine(fused_feat)

        return diff_feat


class MultiScaleDualDifference(nn.Module):
    """
    多尺度双重差异建模

    在金字塔的每个层级应用双重差异建模
    """

    def __init__(self, channels_list):
        """
        Args:
            channels_list: 每个金字塔层级的通道数列表
                          例如：[128, 128, 128, 128] 表示4个层级
        """
        super().__init__()

        self.diff_modules = nn.ModuleList([
            DualDifferenceModule(c, c) for c in channels_list
        ])

    def forward(self, feats1_pyramid, feats2_pyramid):
        """
        Args:
            feats1_pyramid: T1时刻的金字塔特征列表
            feats2_pyramid: T2时刻的金字塔特征列表

        Returns:
            diff_pyramid: 差异特征金字塔列表
        """
        assert len(feats1_pyramid) == len(feats2_pyramid)

        diff_pyramid = []
        for i, (f1, f2) in enumerate(zip(feats1_pyramid, feats2_pyramid)):
            diff = self.diff_modules[i](f1, f2)
            diff_pyramid.append(diff)

        return diff_pyramid


class EnhancedDualDifference(nn.Module):
    """
    增强型双重差异模块

    额外功能：
    1. 跨尺度差异对比
    2. 时序一致性约束
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 基础双重差异
        self.dual_diff = DualDifferenceModule(in_channels, out_channels)

        # 时序一致性分支
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: T1特征 [B, C, H, W]
            feat2: T2特征 [B, C, H, W]

        Returns:
            enhanced_diff: 增强差异特征 [B, C_out, H, W]
        """
        # 1. 双重差异建模
        diff_feat = self.dual_diff(feat1, feat2)

        # 2. 时序上下文（拼接原始特征）
        temporal_feat = torch.cat([feat1, feat2], dim=1)
        temporal_feat = self.temporal_conv(temporal_feat)

        # 3. 融合差异和时序信息
        concat = torch.cat([diff_feat, temporal_feat], dim=1)
        enhanced_diff = self.final_fusion(concat)

        return enhanced_diff


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 测试基础双重差异模块
    print("=" * 60)
    print("Testing DualDifferenceModule")
    print("=" * 60)

    module = DualDifferenceModule(in_channels=256, out_channels=128).to(device)

    feat1 = torch.randn(2, 256, 32, 32).to(device)
    feat2 = torch.randn(2, 256, 32, 32).to(device)

    diff = module(feat1, feat2)
    print(f"Input: {feat1.shape}")
    print(f"Output: {diff.shape}")

    # 测试多尺度双重差异
    print("\n" + "=" * 60)
    print("Testing MultiScaleDualDifference")
    print("=" * 60)

    ms_module = MultiScaleDualDifference(
        channels_list=[128, 128, 128, 128]
    ).to(device)

    feats1 = [torch.randn(2, 128, s, s).to(device) for s in [64, 32, 16, 8]]
    feats2 = [torch.randn(2, 128, s, s).to(device) for s in [64, 32, 16, 8]]

    diff_pyramid = ms_module(feats1, feats2)

    for i, d in enumerate(diff_pyramid):
        print(f"Level {i}: {d.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in ms_module.parameters())
    print(f"\nTotal parameters: {total_params:,}")
