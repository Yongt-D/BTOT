"""
HBCA - Hierarchical Bi-temporal Cross-Attention
层次化双时序交叉注意力模块

核心创新:
传统变化检测使用|T1-T2|差分或局部卷积融合(如TFF)来建模时序变化。
这些方法的感受野有限，对"拆了重建"等全局语义变化场景效果不佳。

HBCA使用交叉注意力让T1的每个位置主动查询T2对应区域的语义变化，
反之亦然。这样可以建模全局上下文依赖，对语义相似但结构不同的变化
（如新旧建筑替换）有更强的检测能力。

与现有方法的区别:
- vs |T1-T2| 差分: HBCA保留了T1/T2各自的特征流，不丢失方向信息
- vs TFF (ChangeDINO): TFF用局部3x3卷积做时序融合，感受野有限;
  HBCA用注意力机制，可以建模任意距离的依赖关系
- vs BIT (Bitemporal Image Transformer): BIT在token级做注意力，
  HBCA在FPN多尺度上做，并带有跨尺度上下文传播

参数效率设计 (避免WHU过拟合):
- 窗口注意力: 将特征图划分为不重叠窗口，在窗口内做交叉注意力
- 低秩投影: Q/K/V用低维投影 (channels → attn_dim)
- 共享权重: 所有FPN层级共享注意力投影权重 (可选)
- Gated fusion: ΔT1和ΔT2通过学习门控融合，门控初始化为0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WindowCrossAttention(nn.Module):
    """
    窗口交叉注意力 (Window Cross-Attention)

    将特征图划分为不重叠的窗口，在每个窗口内做交叉注意力。
    这样既有全局建模能力（不同窗口覆盖不同区域），
    又控制了计算量（窗口内token数有限）。

    Q来自T1（或T2），KV来自T2（或T1），
    实现"T1主动查询T2中哪些位置发生了变化"。
    """

    def __init__(self, channels, attn_dim=64, num_heads=4, window_size=8,
                 differential=False, lambda_init=0.8):
        """
        Args:
            channels: 输入特征通道数
            attn_dim: 注意力计算的降维维度
            num_heads: 注意力头数
            window_size: 窗口大小 (window_size x window_size)
            differential: v12 - 启用差分注意力 (A1 - λ A2), 抑制噪声/伪相关
            lambda_init: 差分系数 λ 初始化值 (per-head learnable, sigmoid 后用)
        """
        super().__init__()
        self.channels = channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.differential = differential

        # Q/K/V 低秩投影 (channels → attn_dim)
        self.q_proj = nn.Linear(channels, attn_dim, bias=False)
        self.k_proj = nn.Linear(channels, attn_dim, bias=False)
        self.v_proj = nn.Linear(channels, attn_dim, bias=False)

        # 输出投影 (attn_dim → channels)
        self.out_proj = nn.Sequential(
            nn.Linear(attn_dim, channels, bias=False),
            nn.LayerNorm(channels)
        )

        # v12 差分注意力的 per-head learnable lambda (用 logit 参数化 + sigmoid)
        if self.differential:
            # logit 使 sigmoid 输出 ≈ lambda_init
            init_logit = torch.logit(torch.tensor(float(lambda_init)).clamp(0.01, 0.99))
            self.lambda_logit = nn.Parameter(torch.full((num_heads,), init_logit.item()))
            # 差分时 head_dim 减半 (Q/K 各 split 成 2 段)
            self.diff_head_dim = self.head_dim // 2
            assert self.diff_head_dim > 0, \
                f"head_dim ({self.head_dim}) too small for differential split"

    def _window_partition(self, x, window_size):
        """将特征图划分为不重叠窗口
        Args:
            x: [B, H, W, C]
            window_size: int
        Returns:
            windows: [B*nH*nW, ws*ws, C]
        """
        B, H, W, C = x.shape
        # 如果不能整除，padding
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = x.shape
        nH, nW = Hp // window_size, Wp // window_size

        x = x.view(B, nH, window_size, nW, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, nH, nW, ws, ws, C]
        x = x.view(B * nH * nW, window_size * window_size, C)
        return x, (Hp, Wp, nH, nW)

    def _window_unpartition(self, windows, info, original_H, original_W):
        """将窗口还原为特征图
        Args:
            windows: [B*nH*nW, ws*ws, C]
            info: (Hp, Wp, nH, nW)
            original_H, original_W: 原始尺寸
        Returns:
            x: [B, H, W, C]
        """
        Hp, Wp, nH, nW = info
        ws = self.window_size
        B = windows.shape[0] // (nH * nW)
        C = windows.shape[-1]

        x = windows.view(B, nH, nW, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, nH, ws, nW, ws, C]
        x = x.view(B, Hp, Wp, C)

        # 去除padding
        if Hp > original_H or Wp > original_W:
            x = x[:, :original_H, :original_W, :]
        return x

    def forward(self, query_feat, kv_feat, shift_size=0):
        """
        交叉注意力: query_feat的每个位置查询kv_feat

        Args:
            query_feat: [B, C, H, W] - 查询侧特征 (T1或T2)
            kv_feat:    [B, C, H, W] - 键值侧特征 (T2或T1)
            shift_size: int - cyclic shift偏移量 (Swin-style shifted window)
                        0 = 普通窗口, window_size//2 = 偏移半个窗口
                        让边缘像素能跨窗口接收上下文
        Returns:
            output: [B, C, H, W] - 注意力输出 (变化感知特征)
        """
        B, C, H, W = query_feat.shape

        # [B, C, H, W] → [B, H, W, C]
        q_hw = query_feat.permute(0, 2, 3, 1).contiguous()
        kv_hw = kv_feat.permute(0, 2, 3, 1).contiguous()

        # ⭐ Cyclic Shift (Swin-style): 让窗口边界对齐到不同位置
        # 边缘像素经过shift后会进入新的窗口, 看到原本被切断的上下文
        if shift_size > 0:
            q_hw = torch.roll(q_hw, shifts=(-shift_size, -shift_size), dims=(1, 2))
            kv_hw = torch.roll(kv_hw, shifts=(-shift_size, -shift_size), dims=(1, 2))

        # 窗口划分
        q_win, info = self._window_partition(q_hw, self.window_size)   # [B*nW, ws², C]
        kv_win, _ = self._window_partition(kv_hw, self.window_size)    # [B*nW, ws², C]

        # Q/K/V 投影
        nB, N, _ = q_win.shape
        q = self.q_proj(q_win).view(nB, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_win).view(nB, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_win).view(nB, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.differential:
            # === v12: Differential attention (Ye et al. 2024) ===
            # Split Q, K head_dim in half: Q=[Q1;Q2], K=[K1;K2], V stays full
            q1, q2 = q.chunk(2, dim=-1)  # [nB, heads, N, head_dim/2]
            k1, k2 = k.chunk(2, dim=-1)
            scale_d = self.diff_head_dim ** -0.5
            a1 = (q1 @ k1.transpose(-2, -1)) * scale_d
            a2 = (q2 @ k2.transpose(-2, -1)) * scale_d
            a1 = a1.softmax(dim=-1)
            a2 = a2.softmax(dim=-1)
            # per-head positive lambda via sigmoid
            lam = torch.sigmoid(self.lambda_logit).view(1, -1, 1, 1)  # [1, heads, 1, 1]
            diff_attn = a1 - lam * a2     # 差分注意力, 抑制噪声/伪对应
            out = (diff_attn @ v)
        else:
            # Standard scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [nB, heads, N, N]
            attn = attn.softmax(dim=-1)
            out = (attn @ v)

        out = out.permute(0, 2, 1, 3).contiguous().view(nB, N, self.attn_dim)
        out = self.out_proj(out)  # [nB, N, C]

        # 还原空间结构
        out = self._window_unpartition(out, info, H, W)  # [B, H, W, C]

        # ⭐ Reverse cyclic shift: 把空间还原回原始位置
        if shift_size > 0:
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(1, 2))

        out = out.permute(0, 3, 1, 2).contiguous()        # [B, C, H, W]

        return out


class HBCALevel(nn.Module):
    """
    单层级的 HBCA (Hierarchical Bi-temporal Cross-Attention)

    对一个FPN层级的T1/T2特征做双向交叉注意力，生成变化特征。

    流程:
        T1 ──→ CrossAttn(Q=T1, KV=T2) ──→ ΔT1 (T1视角的变化)
        T2 ──→ CrossAttn(Q=T2, KV=T1) ──→ ΔT2 (T2视角的变化)
        ΔT1, ΔT2 ──→ GatedFusion ──→ 变化特征

    门控融合设计:
        gate = σ(W_gate · [ΔT1; ΔT2; |T1-T2|])
        output = gate ⊙ ΔT1 + (1-gate) ⊙ ΔT2 + |T1-T2|

    这里保留了|T1-T2|作为残差，确保即使注意力模块完全失效，
    也不会比简单差分差。gamma初始化为0进一步保障稳定性。
    """

    def __init__(self, channels, attn_dim=64, num_heads=4, window_size=8,
                 dropout=0.0, use_shifted=True, use_gamma_gate=True,
                 asymmetric=False, use_edge_branch=False,
                 use_bi_whitening=False, style_recovery_init=0.5,
                 differential=False, diff_lambda_init=0.8):
        """
        Args:
            channels: 特征通道数
            attn_dim: 注意力降维维度
            num_heads: 注意力头数
            window_size: 窗口大小
            dropout: Dropout概率
            use_shifted: 是否启用Shifted Window (T2→T1方向用shift, 默认True)
                         设为False可对照消融
            use_gamma_gate: 是否使用gamma门控残差 (默认True)
                           False时直接相加: change_feat = abs_diff + fused (消融用)
            asymmetric: ⭐ v7 - 不对称 HBCA. 若 True, T1→T2 和 T2→T1 使用
                        独立的 Q/K/V 投影 (cross_attn_fwd / cross_attn_bwd).
                        这让前向路径学"新增建筑"特征, 反向路径学"拆除建筑"特征,
                        配合训练时 swap-equivariance loss 可获得方向性 disentanglement.
                        默认 False 保持 v6 行为 (前后向共享权重).
        """
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.use_shifted = use_shifted
        self.use_gamma_gate = use_gamma_gate
        self.asymmetric = asymmetric
        self.shift_size = window_size // 2 if use_shifted else 0

        # 双向交叉注意力
        # v6 (asymmetric=False): T1→T2 和 T2→T1 共享同一组 Q/K/V 投影
        # v7 (asymmetric=True):  两个方向使用独立投影, 可分别学"新增" vs "拆除"
        # v12 (differential=True): A1 - λ A2 差分注意力 (Ye et al. 2024 Differential Transformer)
        self.cross_attn = WindowCrossAttention(
            channels=channels,
            attn_dim=attn_dim,
            num_heads=num_heads,
            window_size=window_size,
            differential=differential,
            lambda_init=diff_lambda_init,
        )
        if asymmetric:
            # 反向 (T2→T1) 路径专用注意力 — 独立 Q/K/V 投影
            self.cross_attn_bwd = WindowCrossAttention(
                channels=channels,
                attn_dim=attn_dim,
                num_heads=num_heads,
                window_size=window_size,
                differential=differential,
                lambda_init=diff_lambda_init,
            )
        else:
            self.cross_attn_bwd = None

        # 门控融合: 学习如何组合ΔT1和ΔT2
        # 输入: [ΔT1; ΔT2; |T1-T2|] = 3C → C
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # 输出细化
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False,
                      groups=channels),  # depthwise
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # pointwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

        # gamma初始化为0: 训练初期HBCA不改变|T1-T2|差分结果
        # 随着训练进行，gamma逐渐增大，注意力模块逐渐起作用
        self.gamma = nn.Parameter(torch.zeros(1))

        # === v8: Edge-Bridged 分支 (与 spatial 分支并行) ===
        # 假设: 建筑变化的 MI 集中在边缘 Fourier 系数 → 显式建模 edge cross-attention
        # 用可学 depthwise 3x3 + pointwise 1x1 提取 edge 响应, 再做窗口交叉注意力
        # gamma_e 初始化 0, 训练初期退化为 v6/v7 行为, 渐进激活 edge 分支
        self.use_edge_branch = use_edge_branch
        if use_edge_branch:
            self.edge_extract = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                          groups=channels, bias=False),  # depthwise gradient-like
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # pointwise mix
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
            self.cross_attn_edge = WindowCrossAttention(
                channels=channels,
                attn_dim=attn_dim,
                num_heads=num_heads,
                window_size=window_size,
            )
            self.gate_e_conv = nn.Sequential(
                nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.Sigmoid(),
            )
            self.refine_e = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, groups=channels),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.gamma_e = nn.Parameter(torch.zeros(1))

        # === v9: Bi-Temporal Joint Whitening + Adaptive Style Recovery ===
        # BIW 把 T1/T2 共有的 nuisance (光照、传感器增益、季节) 显式抹掉,
        # 让 HBCA 只关注 content 差异。gamma_style 残差允许必要时把 style 信号加回来。
        # 零容量 (μ/σ 不是可学参数), 唯一新参数是 1 个 scalar gamma_style。
        self.use_bi_whitening = use_bi_whitening
        if use_bi_whitening:
            self.gamma_style = nn.Parameter(torch.tensor(float(style_recovery_init)))

    def _bi_whitening(self, f1, f2):
        """双时序联合白化: 用 (T1, T2) 联合均值/方差对各自做归一化.
        Args:
            f1, f2: [B, C, H, W]
        Returns:
            f1_w, f2_w: [B, C, H, W] 白化后特征 (零均值, 单位方差近似)
        """
        B, C, H, W = f1.shape
        f1_flat = f1.reshape(B, C, -1)
        f2_flat = f2.reshape(B, C, -1)

        mu1 = f1_flat.mean(dim=-1, keepdim=True)   # [B, C, 1]
        mu2 = f2_flat.mean(dim=-1, keepdim=True)
        mu = 0.5 * (mu1 + mu2)                      # 联合通道均值

        var1 = f1_flat.var(dim=-1, keepdim=True, unbiased=False)
        var2 = f2_flat.var(dim=-1, keepdim=True, unbiased=False)
        sigma = (0.5 * (var1 + var2) + 1e-6).sqrt()  # 联合通道标准差

        f1_w = (f1_flat - mu) / sigma
        f2_w = (f2_flat - mu) / sigma
        return f1_w.view(B, C, H, W), f2_w.view(B, C, H, W)

    def forward(self, f1, f2):
        """
        Args:
            f1: [B, C, H, W] T1特征
            f2: [B, C, H, W] T2特征
        Returns:
            change_feat: [B, C, H, W] 变化特征
        """
        # === v9: Bi-Temporal Joint Whitening ===
        # 在 HBCA 处理之前抹掉 T1/T2 共有的 nuisance, 保留 content 差异
        if self.use_bi_whitening:
            f1_orig, f2_orig = f1, f2  # 保留原始特征做 style recovery
            f1, f2 = self._bi_whitening(f1, f2)

        # 基础差分 (保底)
        abs_diff = torch.abs(f1 - f2)

        # 双向交叉注意力
        # ΔT1: T1查询T2 → "T2中哪些地方和T1不同" (普通窗口)
        delta_t1 = self.cross_attn(f1, f2, shift_size=0)
        # ΔT2: T2查询T1 → "T1中哪些地方和T2不同" (shifted窗口)
        # 边缘像素在 delta_t2 中能跨窗口看到上下文, 解决img512/7377类失败
        # v7 asymmetric=True 时使用独立投影 cross_attn_bwd, 否则共享 cross_attn
        if self.asymmetric:
            delta_t2 = self.cross_attn_bwd(f2, f1, shift_size=self.shift_size)
        else:
            delta_t2 = self.cross_attn(f2, f1, shift_size=self.shift_size)

        # 门控融合
        gate_input = torch.cat([delta_t1, delta_t2, abs_diff], dim=1)  # [B, 3C, H, W]
        gate = self.gate_conv(gate_input)  # [B, C, H, W], ∈ (0, 1)

        # 加权融合: gate控制ΔT1和ΔT2的贡献比例
        fused = gate * delta_t1 + (1 - gate) * delta_t2

        # 细化
        fused = self.refine(fused)

        if self.dropout is not None:
            fused = self.dropout(fused)

        # === v8: Edge-Bridged 并行分支 ===
        if self.use_edge_branch:
            e1 = self.edge_extract(f1)
            e2 = self.edge_extract(f2)
            abs_edge_diff = torch.abs(e1 - e2)
            # 边缘双向交叉注意力 (与 spatial 分支同窗口大小和 shift 策略)
            delta_e1 = self.cross_attn_edge(e1, e2, shift_size=0)
            delta_e2 = self.cross_attn_edge(e2, e1, shift_size=self.shift_size)
            gate_e_in = torch.cat([delta_e1, delta_e2, abs_edge_diff], dim=1)
            gate_e = self.gate_e_conv(gate_e_in)
            fused_e = gate_e * delta_e1 + (1 - gate_e) * delta_e2
            fused_e = self.refine_e(fused_e)
            if self.dropout is not None:
                fused_e = self.dropout(fused_e)
        else:
            fused_e = None

        # gamma-gated residual: 基础差分 + gamma * spatial + gamma_e * edge
        # gamma=0时退化为|T1-T2|，保证不会比baseline差
        if self.use_gamma_gate:
            change_feat = abs_diff + self.gamma * fused
            if self.use_edge_branch:
                change_feat = change_feat + self.gamma_e * fused_e
        else:
            # 消融: 直接相加，不用gamma门控
            change_feat = abs_diff + fused
            if self.use_edge_branch:
                change_feat = change_feat + fused_e

        # === v9: Adaptive Style Recovery ===
        # 把原始 (未白化) 特征的差异作残差加回, 允许模型在必要时恢复 style 信号
        # gamma_style 初始化 0.5, 训练中可学习地决定是否需要 style
        if self.use_bi_whitening:
            change_feat = change_feat + self.gamma_style * torch.abs(f1_orig - f2_orig)

        return change_feat


class HierarchicalBitemporalCrossAttention(nn.Module):
    """
    HBCA - 层次化双时序交叉注意力 (多尺度版本)

    在FPN的4个层级上分别做双向交叉注意力，
    并通过自顶向下的上下文传播连接各层级。

    层级配置 (自适应窗口大小):
    - P1 (64×64): window_size=8, 64个窗口, 每窗口64 tokens
    - P2 (32×32): window_size=8, 16个窗口, 每窗口64 tokens
    - P3 (16×16): window_size=8, 4个窗口, 每窗口64 tokens
    - P4 (8×8):   window_size=8, 全局注意力 (整个特征图就是一个窗口)

    跨尺度上下文传播 (自顶向下):
    P4的变化特征上采样后与P3融合，P3融合后与P2融合...
    这样高层语义变化信息可以引导低层的细粒度变化检测。

    与MultiScaleMSCA接口兼容，可直接替换。
    """

    def __init__(self, channels_list, attn_dim=64, num_heads=4,
                 window_size=8, dropout=0.0, share_weights=False, use_shifted=True,
                 use_gamma_gate=True, use_cross_scale=True, asymmetric=False,
                 use_edge_branch=False, use_bi_whitening=False,
                 style_recovery_init=0.5, differential=False, diff_lambda_init=0.8):
        """
        Args:
            channels_list: 各层级通道数 [C, C, C, C]
            attn_dim: 注意力降维维度
            num_heads: 注意力头数
            window_size: 窗口大小
            dropout: Dropout概率
            share_weights: 是否共享各层级的注意力权重 (True可减少参数)
            use_shifted: 启用Shifted Window跨窗口通信 (默认True, 解决边缘失败)
            use_gamma_gate: gamma门控残差 (默认True, 消融用)
            use_cross_scale: 跨尺度上下文传播 (默认True, 消融用)
        """
        super().__init__()
        self.num_levels = len(channels_list)
        self.share_weights = share_weights
        self.use_shifted = use_shifted
        self.use_cross_scale = use_cross_scale

        if share_weights:
            # 所有层级共享一个HBCA模块 (参数最少)
            assert all(c == channels_list[0] for c in channels_list), \
                "share_weights=True requires all channels to be the same"
            self.shared_hbca = HBCALevel(
                channels=channels_list[0],
                attn_dim=attn_dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
                use_shifted=use_shifted,
                use_gamma_gate=use_gamma_gate,
                asymmetric=asymmetric,
                use_edge_branch=use_edge_branch,
                use_bi_whitening=use_bi_whitening,
                style_recovery_init=style_recovery_init,
                differential=differential,
                diff_lambda_init=diff_lambda_init,
            )
        else:
            # 各层级独立的HBCA模块
            self.hbca_modules = nn.ModuleList([
                HBCALevel(
                    channels=c,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    dropout=dropout,
                    use_shifted=use_shifted,
                    use_gamma_gate=use_gamma_gate,
                    asymmetric=asymmetric,
                    use_edge_branch=use_edge_branch,
                    use_bi_whitening=use_bi_whitening,
                    style_recovery_init=style_recovery_init,
                    differential=differential,
                    diff_lambda_init=diff_lambda_init,
                )
                for c in channels_list
            ])

        # 跨尺度上下文传播 (自顶向下)
        # 高层变化特征上采样后与低层融合
        if use_cross_scale:
            self.cross_scale_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels_list[i], channels_list[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels_list[i]),
                    nn.ReLU(inplace=True)
                )
                for i in range(self.num_levels - 1)  # P3, P2, P1 各一个
            ])

    def forward(self, pyramid1, pyramid2):
        """
        Args:
            pyramid1: list of T1 features [p1, p2, p3, p4] (从细到粗)
            pyramid2: list of T2 features [p1, p2, p3, p4]
        Returns:
            diff_pyramid: list of change features [d1, d2, d3, d4]
        """
        # Step 1: 各层级独立计算HBCA
        outputs = []
        for i in range(self.num_levels):
            f1, f2 = pyramid1[i], pyramid2[i]
            if self.share_weights:
                out = self.shared_hbca(f1, f2)
            else:
                out = self.hbca_modules[i](f1, f2)
            outputs.append(out)

        # Step 2: 自顶向下跨尺度上下文传播
        # 从最粗层(P4)开始，逐步将高层语义变化信息传递到低层
        if self.use_cross_scale:
            for i in range(self.num_levels - 1, 0, -1):
                # 高层上采样到低层尺寸
                high_feat = F.interpolate(
                    outputs[i], size=outputs[i-1].shape[-2:],
                    mode='bilinear', align_corners=False
                )
                # 融合: 低层特征 + 高层上下文
                outputs[i-1] = outputs[i-1] + self.cross_scale_convs[i-1](high_feat)

        return outputs
