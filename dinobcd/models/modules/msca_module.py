"""
MSCA - Multi-Scale Change Aggregation
多尺度变化聚合模块

核心思想:
CADI使用9点局部可变形采样, 感受野仅为±3像素 (3×3邻域).
对于大型建筑物变化 (跨度20-50px), CADI只能感知到建筑物的局部角落,
无法获取完整的变化语义, 导致系统性漏检 (Recall天花板 ~86%).

MSCA使用ASPP风格的多膨胀率深度可分离卷积, 显式建模三个尺度的变化.

v2 (Exp14): SGBD - Semantic-Guided Bidirectional Difference
在原有|f1-f2|基础上增加:
1. 双向差异流: ReLU(f1-f2) + ReLU(f2-f1) 保留变化方向信息
2. 余弦不相似度加权: 对光照/季节伪变化鲁棒
3. 语义上下文门控: 利用(f1+f2)/2的语义信息选择性放大建筑区域变化

数学形式 (SGBD):
    给定 T1 特征 f₁, T2 特征 f₂:
    1. 双向差异:  d_pos = ReLU(f₁ - f₂),  d_neg = ReLU(f₂ - f₁)
    2. 幅度差异:  d_abs = |f₁ - f₂|
    3. 余弦加权:  w_cos = 1 - cos(f₁, f₂)  ∈ [0, 2]
    4. 融合差异:  d_fuse = W_fuse · [d_pos; d_neg; w_cos ⊙ d_abs]  (3C → C)
    5. 语义门控:  ctx = (f₁ + f₂) / 2
                  gate = σ(W_gate · [d_fuse; ctx])  (2C → C)
                  d_out = d_fuse ⊙ (2 · gate)
       注: 2·gate ∈ (0,2), 初始化时gate≈0.5→恒等. 可抑制伪变化(gate<0.5)
           也可放大真变化(gate>0.5), 比(1+gate)多了抑制能力
    6. 后续: 多尺度膨胀卷积 + SE注意力 (同原MSCA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCAModule(nn.Module):
    """
    Multi-Scale Change Aggregation (单尺度)

    对给定通道数 C 的 T1/T2 特征对, 提取多尺度差异表示.
    支持两种差异建模模式:
    - use_sgbd=False: 原始 |f1-f2| (向后兼容)
    - use_sgbd=True:  SGBD 语义引导双向差异增强 (Exp14创新)
    """

    def __init__(self, channels, dilations=(1, 3, 6), se_ratio=4, dropout=0.1,
                 use_context_gate=False, use_cosine_weight=False,
                 use_sgbd=False, use_sgbd_gate=False):
        """
        Args:
            channels (int): 输入特征通道数 C
            dilations (tuple): 膨胀率列表. 默认 (1,3,6)
            se_ratio (int): SE通道压缩比
            dropout (float): Dropout2d概率
            use_context_gate (bool): [旧接口] 启用语义上下文门控. use_sgbd=True时忽略.
            use_cosine_weight (bool): [旧接口] 启用余弦不相似度加权. use_sgbd=True时忽略.
            use_sgbd (bool): 启用SGBD语义引导双向差异增强 (推荐, 替代上面两个旧选项)
            use_sgbd_gate (bool): SGBD是否使用语义门控 (默认False, 减少过拟合风险)
        """
        super().__init__()
        C = channels
        inner = max(C // 4, 32)
        n_d = len(dilations)
        self.use_sgbd = use_sgbd
        self.use_sgbd_gate = use_sgbd_gate and use_sgbd
        # 旧接口: 仅在use_sgbd=False时生效
        self.use_context_gate = use_context_gate and not use_sgbd
        self.use_cosine_weight = use_cosine_weight and not use_sgbd

        # ================================================================
        # SGBD: 语义引导双向差异增强 (Semantic-Guided Bidirectional Difference)
        # ================================================================
        if use_sgbd:
            # 双向差异融合: [d_pos; d_neg; cos_weighted_abs] → C
            # 3C → C 的1×1卷积, 学习如何组合三种差异信号
            self.sgbd_fuse = nn.Sequential(
                nn.Conv2d(C * 3, C, 1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True)
            )

            # 语义上下文门控 (可选, 默认关闭以减少过拟合)
            if use_sgbd_gate:
                self.sgbd_gate = nn.Sequential(
                    nn.Conv2d(C * 2, C, 1, bias=False),
                    nn.BatchNorm2d(C),
                    nn.Sigmoid()
                )

        # ================================================================
        # 旧接口保持不变 (向后兼容)
        # ================================================================
        if self.use_context_gate:
            self.context_gate = nn.Sequential(
                nn.Conv2d(C * 2, C, 1, bias=False),
                nn.BatchNorm2d(C),
                nn.Sigmoid()
            )

        # ── 步骤1: 降维投影 (C → inner) ──────────────────────────────
        self.proj_in = nn.Sequential(
            nn.Conv2d(C, inner, 1, bias=False),
            nn.BatchNorm2d(inner),
            nn.ReLU(inplace=True)
        )

        # ── 步骤2: 多膨胀率深度可分离卷积 ───────────────────────────
        self.dil_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inner, inner, kernel_size=3,
                          padding=r, dilation=r, groups=inner, bias=False),
                nn.Conv2d(inner, inner, 1, bias=False),
                nn.BatchNorm2d(inner),
                nn.ReLU(inplace=True)
            )
            for r in dilations
        ])

        # ── 步骤3: 多分支聚合 ──────────────────────────────────────
        fused = inner * (n_d + 1)
        self.proj_out = nn.Sequential(
            nn.Conv2d(fused, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

        # ── 步骤4: SE通道注意力 ────────────────────────────────────
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, max(C // se_ratio, 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(C // se_ratio, 16), C),
            nn.Sigmoid()
        )

    def _compute_diff_sgbd(self, f1, f2):
        """
        SGBD差异计算: 双向差异 + 余弦加权 + 语义门控

        Args:
            f1: [B, C, H, W] T1特征
            f2: [B, C, H, W] T2特征

        Returns:
            diff_enhanced: [B, C, H, W] 增强后的差异特征
        """
        # 1. 双向差异 (保留变化方向信息)
        # d_pos: f1中有但f2没有的 (建筑拆除信号)
        # d_neg: f2中有但f1没有的 (建筑新建信号)
        d_pos = F.relu(f1 - f2)     # [B, C, H, W]
        d_neg = F.relu(f2 - f1)     # [B, C, H, W]

        # 2. 余弦不相似度加权的幅度差异
        # 余弦距离对光照变化鲁棒: 光照改变向量幅度但不改变方向
        # 真实变化(建筑出现/消失)改变特征方向 → 余弦距离大
        d_abs = torch.abs(f1 - f2)  # [B, C, H, W]
        cos_sim = F.cosine_similarity(f1, f2, dim=1, eps=1e-6)  # [B, H, W]
        cos_dissim = (1.0 - cos_sim).unsqueeze(1).clamp(min=0)  # [B, 1, H, W]
        d_cos = d_abs * cos_dissim  # 在特征方向不同处放大差异

        # 3. 融合三种差异信号
        d_fuse = self.sgbd_fuse(torch.cat([d_pos, d_neg, d_cos], dim=1))  # [B, C, H, W]

        # 4. 语义上下文门控 (可选)
        if self.use_sgbd_gate:
            context = (f1 + f2) * 0.5
            gate = self.sgbd_gate(torch.cat([d_fuse, context], dim=1))
            diff_enhanced = d_fuse * (2.0 * gate)
        else:
            # 无门控: 直接使用融合差异, 减少过拟合风险
            diff_enhanced = d_fuse

        return diff_enhanced

    def forward(self, f1, f2):
        """
        Args:
            f1: [B, C, H, W] T1 (时间1) 特征
            f2: [B, C, H, W] T2 (时间2) 特征

        Returns:
            [B, C, H, W] 多尺度变化聚合特征
        """
        # ── 差异信号 ───────────────────────────────────────────────
        if self.use_sgbd:
            # SGBD: 语义引导双向差异增强
            diff = self._compute_diff_sgbd(f1, f2)
        else:
            # 原始: 简单绝对差异
            diff = torch.abs(f1 - f2)

            # 旧接口: 余弦不相似度加权
            if self.use_cosine_weight:
                cos_sim = F.cosine_similarity(f1, f2, dim=1, eps=1e-6)
                cos_dissim = (1.0 - cos_sim).unsqueeze(1).clamp(min=0)
                diff = diff * (1.0 + cos_dissim)

            # 旧接口: 语义上下文门控
            if self.use_context_gate:
                context = (f1 + f2) * 0.5
                gate = self.context_gate(torch.cat([diff, context], dim=1))
                diff = diff * (1.0 + gate)

        # ── 降维 ──────────────────────────────────────────────────
        x = self.proj_in(diff)

        # 多尺度分支 + identity分支
        branches = [x]
        for branch in self.dil_branches:
            branches.append(branch(x))

        # 聚合
        x = torch.cat(branches, dim=1)
        out = self.proj_out(x)

        # SE通道注意力
        B = out.shape[0]
        w = self.se(out).view(B, -1, 1, 1)
        out = out * w

        # 残差连接
        return out + diff


class MultiScaleMSCA(nn.Module):
    """
    多尺度MSCA模块 (在金字塔4个层级上独立应用)

    与MultiScaleCADI接口兼容, 可直接替换.
    额外提供自顶向下的跨尺度上下文传播.
    """

    def __init__(self, channels_list, dilations=(1, 3, 6), se_ratio=4, dropout=0.1,
                 use_context_gate=False, use_cosine_weight=False,
                 use_sgbd=False, use_sgbd_gate=False):
        """
        Args:
            channels_list (list): 各pyramid层级的通道数
            dilations (tuple): 膨胀率
            se_ratio (int): SE压缩比
            dropout (float): Dropout2d概率
            use_context_gate (bool): [旧接口] 启用语义上下文门控
            use_cosine_weight (bool): [旧接口] 启用余弦不相似度加权
            use_sgbd (bool): 启用SGBD语义引导双向差异增强
            use_sgbd_gate (bool): SGBD是否使用语义门控
        """
        super().__init__()
        self.num_levels = len(channels_list)

        # 各层级独立的MSCA
        self.msca_modules = nn.ModuleList([
            MSCAModule(c, dilations=dilations, se_ratio=se_ratio, dropout=dropout,
                       use_context_gate=use_context_gate, use_cosine_weight=use_cosine_weight,
                       use_sgbd=use_sgbd, use_sgbd_gate=use_sgbd_gate)
            for c in channels_list
        ])

        # 自顶向下跨尺度融合
        self.top_down = nn.ModuleList()
        for i in range(1, len(channels_list)):
            self.top_down.append(nn.Sequential(
                nn.Conv2d(channels_list[i], channels_list[i - 1], 1, bias=False),
                nn.BatchNorm2d(channels_list[i - 1])
            ))

        self.cross_scale_gamma = nn.Parameter(
            torch.full((len(channels_list) - 1,), 0.1)
        )

        if use_sgbd:
            mode_str = "SGBD+Gate" if use_sgbd_gate else "SGBD"
        else:
            mode_str = "ContextGate+Cosine" if (use_context_gate or use_cosine_weight) else "Standard"
        print(f"[MSCA] MultiScaleMSCA initialized (mode={mode_str}):")
        for i, c in enumerate(channels_list):
            print(f"  Level {i}: C={c}, dilations={dilations}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total_params:,} ({total_params / 1e3:.1f}K)")

    def forward(self, pyramid1, pyramid2):
        """
        Args:
            pyramid1: list of T1 features [p1, p2, p3, p4]
            pyramid2: list of T2 features

        Returns:
            list of change features, same length as input
        """
        # 各层级独立计算MSCA
        outputs = []
        for i, (msca, f1, f2) in enumerate(zip(self.msca_modules, pyramid1, pyramid2)):
            outputs.append(msca(f1, f2))

        # 自顶向下融合
        for i in range(len(outputs) - 1):
            deep = outputs[i]
            shallow = outputs[i + 1]

            target_h, target_w = shallow.shape[2], shallow.shape[3]
            deep_up = F.interpolate(deep, size=(target_h, target_w),
                                    mode='bilinear', align_corners=False)

            deep_proj = self.top_down[i](deep_up)
            gamma = self.cross_scale_gamma[i]
            outputs[i + 1] = shallow + gamma * deep_proj

        return outputs
