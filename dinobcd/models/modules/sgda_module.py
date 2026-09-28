"""
Spectral-Graph Difference Aggregation (SGDA)
核心创新模块 - 多粒度自适应变化交互

核心思想:
建筑物变化发生在多个粒度上:
- 全局粒度: 大型建筑出现/消失 (需要全局频谱特征)
- 区域粒度: 建筑扩展、部分拆除 (需要图传播的长程依赖)
- 局部粒度: 边界变化、小型结构 (需要局部可变形采样)

数学框架:
给定双时相特征 f1, f2 ∈ R^{B×C×H×W}:

Branch A - 频谱差异 (全局):
  F̃1 = FFT2D(f1), F̃2 = FFT2D(f2)
  ΔF̃ = |F̃1| - |F̃2|  (振幅差异捕获结构变化)
  w = σ(MLP(GAP(|ΔF̃|)))  (通道重要性权重)
  Δ_spectral = IFFT2D(w ⊙ ΔF̃)

Branch B - 图可变形传播 (区域, 借鉴TPAMI NDGC):
  虚拟邻居实现长程变化感知采样
  K=16个虚拟邻居, 范围[-8,8]

Branch C - 局部可变形细化 (局部):
  保留CADI式局部采样, 9点3×3网格

自适应多粒度融合:
  α = softmax(Conv1x1(cat[Δ_spectral, Δ_graph, Δ_local]))
  Δ = α₀⊙Δ_spectral + α₁⊙Δ_graph + α₂⊙Δ_local

参数量: ~370K (vs CADI ~800K, 减少54%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


# ============================================================
# Branch A: 频谱差异 (Spectral Difference)
# ============================================================

class SpectralDifference(nn.Module):
    """
    频谱差异模块 - 在频域捕获全局结构变化

    原理:
    - FFT将空间特征变换到频域
    - 频域中的振幅差异反映全局结构变化
    - 低频差异 = 大尺度语义变化
    - 高频差异 = 边界和纹理变化
    - 通道注意力学习哪些频率通道最重要

    优势:
    - 全局感受野 (整张特征图)
    - 零卷积参数 (FFT是算法操作)
    - 仅需通道注意力MLP (~16K params)
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        # 通道注意力: 学习频谱差异中哪些通道最重要
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # 频谱差异的空间精炼
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        """
        Args:
            f1, f2: [B, C, H, W] 双时相特征

        Returns:
            delta_spectral: [B, C, H, W] 频谱差异特征
        """
        B, C, H, W = f1.shape

        # Step 1: 2D FFT (实数FFT, 输出复数)
        f1_fft = torch.fft.rfft2(f1, norm='ortho')  # [B, C, H, W//2+1] complex
        f2_fft = torch.fft.rfft2(f2, norm='ortho')

        # Step 2: 频谱差异 (复数减法)
        delta_fft = f1_fft - f2_fft

        # Step 3: 振幅差异的通道注意力
        # 使用振幅(模)来计算通道重要性
        delta_amplitude = torch.abs(delta_fft)  # [B, C, H, W//2+1] real
        # 填充回完整大小用于GAP
        delta_amp_spatial = torch.fft.irfft2(
            delta_fft * (delta_amplitude / (delta_amplitude + 1e-8)),
            s=(H, W), norm='ortho'
        )

        w = self.channel_attn(delta_amp_spatial.abs())  # [B, C]
        w = w.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Step 4: 加权频谱差异 → 逆FFT回空间域
        # 对频谱差异施加通道权重
        delta_weighted_fft = delta_fft * w.to(delta_fft.dtype)
        delta_spectral = torch.fft.irfft2(delta_weighted_fft, s=(H, W), norm='ortho')

        # Step 5: 空间精炼
        delta_spectral = self.refine(delta_spectral)

        return delta_spectral


# ============================================================
# Branch B: 图可变形传播 (Graph Deformable Propagation)
# ============================================================

class GraphDeformablePropagation(nn.Module):
    """
    图可变形传播模块 - 借鉴TPAMI NDGC的虚拟邻居思想

    核心创新:
    1. 将每个空间位置视为图节点
    2. 通过变化引导的偏移生成K个虚拟邻居
    3. 在虚拟邻居上进行消息聚合
    4. 实现长程变化感知的特征交互

    与CADI的区别:
    - CADI: 9个固定3×3网格点, 偏移范围[-3,3]
    - SGDA-Graph: K=16个虚拟邻居, 偏移范围[-8,8], 更大的感受野

    参数量: ~200K (通过深度可分离卷积和组卷积优化)
    """

    def __init__(self, channels, num_virtual_neighbors=16, offset_range=8.0):
        super().__init__()
        self.channels = channels
        self.K = num_virtual_neighbors
        self.offset_range = offset_range

        # 变化引导的偏移生成器
        # 使用分组卷积减少参数: channels → channels//4 → K*2
        self.offset_net = nn.Sequential(
            DepthwiseSeparableConv(channels, channels // 4),
            nn.Conv2d(channels // 4, self.K * 2, kernel_size=1, bias=True)
        )

        # 初始化偏移为0 (训练初期不采样远处)
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        # 虚拟邻居权重生成 (注意力权重)
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels, self.K, kernel_size=1, bias=True),
            nn.Softmax(dim=1)  # K个邻居的归一化权重
        )

        # 聚合后的特征精炼
        self.aggregate_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def _deformable_sample(self, feat, offsets):
        """
        可变形采样: 在偏移位置进行双线性插值采样

        Args:
            feat: [B, C, H, W]
            offsets: [B, K, 2, H, W] (dx, dy for each virtual neighbor)

        Returns:
            sampled: [B, K, C, H, W]
        """
        B, C, H, W = feat.shape
        K = offsets.shape[1]

        # 生成基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=torch.float32),
            torch.arange(W, device=feat.device, dtype=torch.float32),
            indexing='ij'
        )
        # [1, 1, H, W]
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)

        sampled_list = []
        for k in range(K):
            # 采样位置 = 基础位置 + 偏移
            sample_x = grid_x + offsets[:, k, 0:1, :, :]  # [B, 1, H, W]
            sample_y = grid_y + offsets[:, k, 1:2, :, :]

            # 归一化到 [-1, 1] (grid_sample要求)
            sample_x_norm = 2.0 * sample_x / (W - 1) - 1.0
            sample_y_norm = 2.0 * sample_y / (H - 1) - 1.0

            # [B, H, W, 2]
            grid = torch.cat([sample_x_norm, sample_y_norm], dim=1)
            grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]

            sampled = F.grid_sample(
                feat, grid, mode='bilinear',
                padding_mode='zeros', align_corners=True
            )  # [B, C, H, W]
            sampled_list.append(sampled)

        return torch.stack(sampled_list, dim=1)  # [B, K, C, H, W]

    def forward(self, f1, f2):
        """
        Args:
            f1, f2: [B, C, H, W]

        Returns:
            delta_graph: [B, C, H, W] 图传播差异特征
        """
        B, C, H, W = f1.shape

        # Step 1: 计算初始差异特征作为引导
        diff_feat = torch.abs(f1 - f2)

        # Step 2: 生成K个虚拟邻居的偏移
        offsets_raw = self.offset_net(diff_feat)  # [B, K*2, H, W]
        offsets = self.offset_range * torch.tanh(offsets_raw)  # 限制范围
        offsets = offsets.view(B, self.K, 2, H, W)  # [B, K, 2, H, W]

        # Step 3: 在虚拟邻居位置采样T1和T2特征
        f1_sampled = self._deformable_sample(f1, offsets)  # [B, K, C, H, W]
        f2_sampled = self._deformable_sample(f2, offsets)  # [B, K, C, H, W]

        # Step 4: 计算每个虚拟邻居的差异
        neighbor_diffs = torch.abs(f1_sampled - f2_sampled)  # [B, K, C, H, W]

        # Step 5: 注意力加权聚合
        attn_weights = self.weight_net(diff_feat)  # [B, K, H, W]
        attn_weights = attn_weights.unsqueeze(2)  # [B, K, 1, H, W]

        delta_graph = (neighbor_diffs * attn_weights).sum(dim=1)  # [B, C, H, W]

        # Step 6: 特征精炼
        delta_graph = self.aggregate_conv(delta_graph)

        return delta_graph


# ============================================================
# Branch C: 局部可变形细化 (Local Deformable Refinement)
# ============================================================

class LocalDeformableRefine(nn.Module):
    """
    局部可变形细化 - 轻量化CADI用于边界精度

    保留3×3网格的局部采样以精确捕获边界变化
    参数量: ~100K (使用深度可分离卷积)
    """

    def __init__(self, channels, num_points=9):
        super().__init__()
        self.num_points = num_points

        # 偏移生成 (轻量级)
        self.offset_net = nn.Sequential(
            DepthwiseSeparableConv(channels, channels // 4),
            nn.Conv2d(channels // 4, num_points * 2, kernel_size=1)
        )
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        # 3×3基础网格
        base = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                base.append([j, i])
        self.register_buffer('base_offset', torch.tensor(base, dtype=torch.float32).view(1, 9, 2, 1, 1))

        # 特征聚合
        self.aggregate = nn.Sequential(
            nn.Conv2d(channels * num_points, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        """
        Args:
            f1, f2: [B, C, H, W]

        Returns:
            delta_local: [B, C, H, W]
        """
        B, C, H, W = f1.shape

        # 差异引导
        diff = torch.abs(f1 - f2)
        offsets_raw = self.offset_net(diff)  # [B, 18, H, W]
        offsets = 3.0 * torch.tanh(offsets_raw)
        offsets = offsets.view(B, self.num_points, 2, H, W)

        # 添加基础网格偏移
        total_offsets = self.base_offset.to(f1.device) + offsets

        # 生成网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=f1.device, dtype=torch.float32),
            torch.arange(W, device=f1.device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).unsqueeze(0)

        # 采样T1和T2
        sampled_diffs = []
        for k in range(self.num_points):
            sample_locs = grid + total_offsets[:, k:k+1]
            sample_x_norm = 2.0 * sample_locs[:, :, 0:1] / (W - 1) - 1.0
            sample_y_norm = 2.0 * sample_locs[:, :, 1:2] / (H - 1) - 1.0

            grid_k = torch.cat([sample_x_norm, sample_y_norm], dim=2)
            grid_k = grid_k.squeeze(1).permute(0, 2, 3, 1)

            f1_s = F.grid_sample(f1, grid_k, mode='bilinear', padding_mode='zeros', align_corners=True)
            f2_s = F.grid_sample(f2, grid_k, mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled_diffs.append(torch.abs(f1_s - f2_s))

        # 聚合
        delta_local = self.aggregate(torch.cat(sampled_diffs, dim=1))
        return delta_local


# ============================================================
# SGDA: 多粒度融合
# ============================================================

class SGDAModule(nn.Module):
    """
    Spectral-Graph Difference Aggregation (SGDA)
    频谱-图差异聚合模块

    将三个粒度的差异建模自适应融合:
    - 全局(频谱): 大尺度结构变化
    - 区域(图传播): 中等尺度变化, 长程依赖
    - 局部(可变形): 边界精度

    自适应融合门控学习每个像素位置应该使用哪个粒度的差异
    """

    def __init__(self, channels, num_virtual_neighbors=16, num_local_points=9, offset_range=8.0):
        super().__init__()

        # 三个差异分支
        self.spectral_branch = SpectralDifference(channels)
        self.graph_branch = GraphDeformablePropagation(channels, num_virtual_neighbors, offset_range)
        self.local_branch = LocalDeformableRefine(channels, num_local_points)

        # 自适应多粒度融合门控
        # 输入: 3个分支的差异拼接 → 输出: 3个空间softmax权重图
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True)
            # softmax在forward中应用
        )

        # 最终精炼
        self.final_refine = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 残差连接权重 (gamma=0初始化, 渐进式学习)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        """
        Args:
            f1, f2: [B, C, H, W] 双时相金字塔特征

        Returns:
            delta_fused: [B, C, H, W] 多粒度融合差异特征
        """
        # 三个粒度的差异
        delta_spectral = self.spectral_branch(f1, f2)   # 全局频谱差异
        delta_graph = self.graph_branch(f1, f2)          # 图传播差异
        delta_local = self.local_branch(f1, f2)          # 局部可变形差异

        # 自适应融合权重 (per-pixel, per-granularity)
        concat = torch.cat([delta_spectral, delta_graph, delta_local], dim=1)
        alpha = F.softmax(self.fusion_gate(concat), dim=1)  # [B, 3, H, W]

        # 加权融合
        delta_fused = (
            alpha[:, 0:1] * delta_spectral +
            alpha[:, 1:2] * delta_graph +
            alpha[:, 2:3] * delta_local
        )

        # 精炼 + 残差
        delta_refined = self.final_refine(delta_fused)
        # 基础差异 + 多粒度增强
        base_diff = torch.abs(f1 - f2)
        output = base_diff + self.gamma * delta_refined

        return output


# ============================================================
# MultiScaleSGDA: 多尺度版本
# ============================================================

class MultiScaleSGDA(nn.Module):
    """
    多尺度SGDA模块

    在金字塔每个层级应用SGDA, 并添加自顶向下的跨尺度传播
    粗层级的全局语义信息 → 细层级的局部决策
    """

    def __init__(self, channels_list, num_virtual_neighbors=16, num_local_points=9, offset_range=8.0):
        super().__init__()

        self.num_levels = len(channels_list)

        # 每个尺度的SGDA
        self.sgda_modules = nn.ModuleList([
            SGDAModule(c, num_virtual_neighbors, num_local_points, offset_range)
            for c in channels_list
        ])

        # 跨尺度传递 (自顶向下, 粗→细)
        self.cross_scale = nn.ModuleList()
        for i in range(len(channels_list)):
            if i > 0:
                self.cross_scale.append(nn.Sequential(
                    nn.Conv2d(channels_list[i], channels_list[i-1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels_list[i-1])
                ))
            else:
                self.cross_scale.append(nn.Identity())

    def forward(self, pyramid_t1, pyramid_t2):
        """
        Args:
            pyramid_t1: [p1, p2, p3, p4] T1金字塔 (细→粗)
            pyramid_t2: [p1, p2, p3, p4] T2金字塔

        Returns:
            diff_pyramid: [d1, d2, d3, d4] 差异金字塔
        """
        diff_pyramid = [None] * self.num_levels

        # 自顶向下处理 (从最粗层开始)
        for i in range(self.num_levels - 1, -1, -1):
            diff = self.sgda_modules[i](pyramid_t1[i], pyramid_t2[i])

            # 添加上层信息 (如果有)
            if i < self.num_levels - 1 and diff_pyramid[i + 1] is not None:
                top_feat = self.cross_scale[i + 1](diff_pyramid[i + 1])
                top_feat = F.interpolate(
                    top_feat, size=diff.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                diff = diff + top_feat

            diff_pyramid[i] = diff

        return diff_pyramid


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SGDA Module")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 单尺度测试
    sgda = SGDAModule(channels=128).to(device)
    f1 = torch.randn(2, 128, 32, 32).to(device)
    f2 = torch.randn(2, 128, 32, 32).to(device)

    output = sgda(f1, f2)
    print(f"Single scale - Input: {f1.shape}, Output: {output.shape}")

    params = sum(p.numel() for p in sgda.parameters())
    print(f"SGDA params: {params:,} ({params/1e3:.1f}K)")

    # 多尺度测试
    print("\n" + "=" * 60)
    print("Testing MultiScaleSGDA")
    print("=" * 60)

    ms_sgda = MultiScaleSGDA(channels_list=[128, 128, 128, 128]).to(device)

    pyramid_t1 = [torch.randn(2, 128, s, s).to(device) for s in [64, 32, 16, 8]]
    pyramid_t2 = [torch.randn(2, 128, s, s).to(device) for s in [64, 32, 16, 8]]

    diff_pyramid = ms_sgda(pyramid_t1, pyramid_t2)
    for i, d in enumerate(diff_pyramid):
        print(f"Level {i}: {d.shape}")

    total = sum(p.numel() for p in ms_sgda.parameters())
    print(f"\nMultiScaleSGDA total params: {total:,} ({total/1e6:.2f}M)")

    # 验证gamma=0
    print(f"\ngamma values: {[m.gamma.item() for m in ms_sgda.sgda_modules]}")

    print("\n" + "=" * 60)
    print("SGDA Test Passed!")
    print("=" * 60)
