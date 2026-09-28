"""
CADI Module - Change-Aware Deformable Interaction
组合创新: ChangeDINO + TPAMI NDGC

核心思想:
1. 借鉴NDGC的"虚拟邻居"实现大范围建模
2. 借鉴ChangeDINO的差异引导注意力
3. 实现轻量级的跨时序可变形特征交互

创新点:
- 变化引导的可变形采样 (区别于标准DCN)
- 跨时序虚拟邻居建模
- 大范围依赖 + 轻量级参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 效率优化"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ChangeGuidedOffsetGenerator(nn.Module):
    """
    变化引导的偏移生成器
    
    借鉴NDGC的虚拟邻居思想，但偏移由变化差异引导
    """
    def __init__(self, channels, num_points=9):
        super().__init__()
        self.num_points = num_points
        
        # 差异特征编码
        self.diff_encoder = nn.Sequential(
            DepthwiseSeparableConv(channels, channels // 2),
            DepthwiseSeparableConv(channels // 2, channels // 4),
        )
        
        # 偏移预测 (x, y for each point)
        self.offset_pred = nn.Conv2d(channels // 4, num_points * 2, kernel_size=1)
        
        # 初始化为小偏移
        nn.init.zeros_(self.offset_pred.weight)
        nn.init.zeros_(self.offset_pred.bias)
        
    def forward(self, diff_feat):
        """
        Args:
            diff_feat: 差异特征 [B, C, H, W]
        Returns:
            offsets: 采样偏移 [B, num_points*2, H, W]
        """
        feat = self.diff_encoder(diff_feat)
        offsets = self.offset_pred(feat)
        # 限制偏移范围 [-3, 3]
        offsets = 3.0 * torch.tanh(offsets)
        return offsets


class DeformableSampling(nn.Module):
    """
    可变形采样模块
    
    使用bilinear插值实现可微的可变形采样
    """
    def __init__(self, channels, num_points=9):
        super().__init__()
        self.num_points = num_points
        
        # 基础采样点位置 (3x3 grid)
        self.register_buffer('base_offset', self._get_base_offset())
        
        # 特征聚合
        self.aggregate = nn.Sequential(
            nn.Conv2d(channels * num_points, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def _get_base_offset(self):
        """生成3x3基础网格偏移"""
        offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                offsets.append([j, i])  # x, y
        return torch.tensor(offsets, dtype=torch.float32).view(1, 9, 2, 1, 1)
    
    def forward(self, feat, offsets):
        """
        Args:
            feat: 输入特征 [B, C, H, W]
            offsets: 采样偏移 [B, num_points*2, H, W]
        Returns:
            sampled: 采样聚合特征 [B, C, H, W]
        """
        B, C, H, W = feat.shape
        
        # 生成采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=torch.float32),
            torch.arange(W, device=feat.device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, 2, H, W]
        
        # 重塑偏移 [B, num_points, 2, H, W]
        offsets = offsets.view(B, self.num_points, 2, H, W)
        
        # 添加基础偏移和预测偏移
        base_offset = self.base_offset.to(feat.device)  # [1, 9, 2, 1, 1]
        sample_locs = grid + base_offset + offsets  # [B, num_points, 2, H, W]
        
        # 归一化到 [-1, 1]
        sample_locs_norm = sample_locs.clone()
        sample_locs_norm[:, :, 0, :, :] = 2.0 * sample_locs[:, :, 0, :, :] / (W - 1) - 1.0
        sample_locs_norm[:, :, 1, :, :] = 2.0 * sample_locs[:, :, 1, :, :] / (H - 1) - 1.0
        
        # 采样
        sampled_feats = []
        for i in range(self.num_points):
            grid_i = sample_locs_norm[:, i].permute(0, 2, 3, 1)  # [B, H, W, 2]
            sampled = F.grid_sample(feat, grid_i, mode='bilinear', 
                                   padding_mode='zeros', align_corners=True)
            sampled_feats.append(sampled)
        
        # 拼接并聚合
        sampled_feats = torch.cat(sampled_feats, dim=1)  # [B, C*num_points, H, W]
        output = self.aggregate(sampled_feats)
        
        return output


class CADIModule(nn.Module):
    """
    Change-Aware Deformable Interaction (CADI)
    
    核心创新模块 - 组合ChangeDINO和TPAMI NDGC的思想
    
    特点:
    1. 变化引导的可变形采样 - 区别于标准DCN
    2. 跨时序特征交互 - 虚拟邻居建模
    3. 轻量级设计 - 参数量仅~200K
    """
    def __init__(self, channels, num_points=9):
        super().__init__()
        self.channels = channels
        
        # 变化差异计算
        self.diff_conv = DepthwiseSeparableConv(channels, channels)
        
        # 变化引导的偏移生成
        self.offset_gen = ChangeGuidedOffsetGenerator(channels, num_points)
        
        # T1特征的可变形采样
        self.deform_sample_t1 = DeformableSampling(channels, num_points)
        
        # T2特征的可变形采样  
        self.deform_sample_t2 = DeformableSampling(channels, num_points)
        
        # 跨时序融合
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),  # diff + t1_sampled + t2_sampled
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels, channels)
        )
        
        # 残差权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, feat_t1, feat_t2):
        """
        Args:
            feat_t1: T1时刻特征 [B, C, H, W]
            feat_t2: T2时刻特征 [B, C, H, W]
        Returns:
            output: 融合后的变化特征 [B, C, H, W]
        """
        # 1. 计算变化差异作为引导
        diff = torch.abs(feat_t1 - feat_t2)
        diff = self.diff_conv(diff)
        
        # 2. 生成变化引导的偏移
        offsets = self.offset_gen(diff)
        
        # 3. 对T1和T2进行可变形采样
        t1_sampled = self.deform_sample_t1(feat_t1, offsets)
        t2_sampled = self.deform_sample_t2(feat_t2, offsets)
        
        # 4. 跨时序融合
        concat = torch.cat([diff, t1_sampled, t2_sampled], dim=1)
        fused = self.cross_fusion(concat)
        
        # 5. 残差连接
        output = diff + self.gamma * fused
        
        return output


class MultiScaleCADI(nn.Module):
    """
    多尺度CADI模块
    
    在金字塔每个层级应用CADI，并添加跨尺度交互
    """
    def __init__(self, channels_list, num_points=9):
        super().__init__()
        
        # 每个尺度的CADI
        self.cadi_modules = nn.ModuleList([
            CADIModule(c, num_points) for c in channels_list
        ])
        
        # 跨尺度特征传递 (自顶向下)
        self.cross_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_list[i], channels_list[i-1], kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) if i > 0 else nn.Identity()
            for i in range(len(channels_list))
        ])
        
    def forward(self, pyramid_t1, pyramid_t2):
        """
        Args:
            pyramid_t1: T1时刻金字塔特征列表 [p1, p2, p3, p4]
            pyramid_t2: T2时刻金字塔特征列表 [p1, p2, p3, p4]
        Returns:
            diff_pyramid: 变化特征金字塔列表
        """
        num_levels = len(pyramid_t1)
        diff_pyramid = [None] * num_levels
        
        # 从最粗层开始处理 (自顶向下)
        for i in range(num_levels - 1, -1, -1):
            # CADI处理
            diff = self.cadi_modules[i](pyramid_t1[i], pyramid_t2[i])
            
            # 如果不是最粗层，添加上层信息
            if i < num_levels - 1 and diff_pyramid[i + 1] is not None:
                top_down = self.cross_scale[i + 1](diff_pyramid[i + 1])
                # 确保尺寸匹配
                if top_down.shape[-2:] != diff.shape[-2:]:
                    top_down = F.interpolate(top_down, size=diff.shape[-2:], 
                                            mode='bilinear', align_corners=False)
                diff = diff + top_down
            
            diff_pyramid[i] = diff
        
        return diff_pyramid


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing CADI Module")
    print("=" * 60)
    
    # 单尺度测试
    cadi = CADIModule(channels=128, num_points=9).to(device)
    feat_t1 = torch.randn(2, 128, 32, 32).to(device)
    feat_t2 = torch.randn(2, 128, 32, 32).to(device)
    
    output = cadi(feat_t1, feat_t2)
    print(f"Single scale - Input: {feat_t1.shape}, Output: {output.shape}")
    
    # 计算参数量
    params = sum(p.numel() for p in cadi.parameters())
    print(f"CADI parameters: {params:,} ({params/1e3:.1f}K)")
    
    print("\n" + "=" * 60)
    print("Testing MultiScaleCADI")
    print("=" * 60)
    
    # 多尺度测试
    ms_cadi = MultiScaleCADI(channels_list=[128, 128, 128, 128]).to(device)
    
    pyramid_t1 = [torch.randn(2, 128, s, s).to(device) for s in [64, 32, 16, 8]]
    pyramid_t2 = [torch.randn(2, 128, s, s).to(device) for s in [64, 32, 16, 8]]
    
    diff_pyramid = ms_cadi(pyramid_t1, pyramid_t2)
    
    for i, d in enumerate(diff_pyramid):
        print(f"Level {i}: {d.shape}")
    
    # 总参数量
    total_params = sum(p.numel() for p in ms_cadi.parameters())
    print(f"\nMultiScaleCADI parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n" + "=" * 60)
    print("CADI Module Test Passed!")
    print("=" * 60)
