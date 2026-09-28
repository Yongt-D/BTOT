"""
PFTI - Post-Fusion Temporal Interaction
融合后时序交互模块

核心发现:
BTI 只在原始 DINO 特征 (32×32 patch tokens) 上做跨时序注意力.
经过 HFA 聚合 + CNN融合 + TopDown细化后, T1和T2的金字塔特征
再也没有互相"看"过对方 → 差异模块只能靠 |f1-f2| 做逐像素比较.

这导致:
1. 微妙的建筑变化 (特征差异很小) 被差异模块忽略
2. 大面积变化区域缺乏全局一致性 (每个像素独立判断)
3. 召回率天花板 ~85.5% (漏检14.5%的变化)

PFTI 在金字塔融合后添加轻量级跨时序注意力:
- 在每个金字塔层级, T1和T2特征互相attend
- T1看到T2后, 原来特征一致的地方会保持不变
- 原来特征有差异的地方会被放大 (attention mismatch → larger diff)
- 这使得后续 diff_module 能更容易检测到变化

使用线性注意力 (ELU+1 kernel) 实现 O(N·D²) 复杂度:
- 标准注意力: O(N²·D), 在 p1 (64×64=4096 tokens) 上不可行
- 线性注意力: O(N·D²/h), 全部层级均可高效计算

数学形式:
    标准注意力: Attn(Q,K,V) = softmax(QK^T/√d) · V   [O(N²)]
    线性注意力: Attn(Q,K,V) = φ(Q)(φ(K)^T · V) / (φ(Q) · Σφ(K))   [O(N·d²)]
    其中 φ(x) = ELU(x) + 1 (非负核函数)

    跨时序: Q来自T1, K/V来自T2 (反之亦然)
    残差: out = x + γ · Attn(x, other)  (γ初始化为0, 渐进激活)

参数量: ~66K/层 × 4层 ≈ 264K (比BTI的800K更轻量)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearCrossAttention(nn.Module):
    """
    线性复杂度跨时序注意力 (单层级)

    T1 attends to T2, T2 attends to T1.
    使用 ELU+1 核函数实现 O(N·d²) 线性注意力.
    """

    def __init__(self, channels, num_heads=4, dropout=0.0):
        """
        Args:
            channels: 特征通道数
            num_heads: 注意力头数
            dropout: attention dropout
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, f"channels({channels}) must be divisible by num_heads({num_heads})"

        # 共享 Q/K/V 投影 (T1和T2共享参数 → 减少过拟合)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)

        # LayerNorm (分别对 T1 和 T2 归一化)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # 残差缩放 (初始化为0 → 模块初始时为恒等映射, 渐进激活)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _linear_attention(self, Q, K, V):
        """
        ELU+1 线性注意力

        标准注意力: softmax(QK^T/√d) · V → O(N²d)
        线性注意力: φ(Q)(φ(K)^TV) / (φ(Q)·Σφ(K)) → O(Nd²)

        Args:
            Q: [B, h, N, d]
            K: [B, h, N, d]
            V: [B, h, N, d]

        Returns:
            output: [B, h, N, d]
        """
        # ELU+1 核: 保证非负 (线性注意力要求)
        Q = F.elu(Q) + 1.0
        K = F.elu(K) + 1.0

        # 线性注意力: 先算 K^T·V [d×d], 再乘 Q
        # 而非先算 Q·K^T [N×N], 再乘 V
        KV = torch.matmul(K.transpose(-2, -1), V)       # [B, h, d, d]
        QKV = torch.matmul(Q, KV)                        # [B, h, N, d]

        # 归一化 (除以 Q·sum(K))
        K_sum = K.sum(dim=-2, keepdim=True)              # [B, h, 1, d]
        Z = torch.matmul(Q, K_sum.transpose(-2, -1))    # [B, h, N, 1]
        output = QKV / (Z + 1e-6)

        return output

    def forward(self, x1, x2):
        """
        双向跨时序注意力

        Args:
            x1: [B, C, H, W] T1 金字塔特征
            x2: [B, C, H, W] T2 金字塔特征

        Returns:
            e1: [B, C, H, W] 增强后的 T1 特征
            e2: [B, C, H, W] 增强后的 T2 特征
        """
        B, C, H, W = x1.shape
        N = H * W
        h = self.num_heads
        d = self.head_dim

        # Flatten + LayerNorm
        x1_flat = self.norm1(x1.reshape(B, C, N).transpose(1, 2))  # [B, N, C]
        x2_flat = self.norm2(x2.reshape(B, C, N).transpose(1, 2))  # [B, N, C]

        # 共享QKV投影
        qkv1 = self.qkv(x1_flat).reshape(B, N, 3, h, d)           # [B, N, 3, h, d]
        qkv2 = self.qkv(x2_flat).reshape(B, N, 3, h, d)

        # 拆分 Q, K, V 并转置为 [B, h, N, d]
        q1, k1, v1 = qkv1.permute(2, 0, 3, 1, 4).unbind(0)        # each [B, h, N, d]
        q2, k2, v2 = qkv2.permute(2, 0, 3, 1, 4).unbind(0)

        # 跨时序注意力: T1的Q attend到T2的K,V; 反之亦然
        out1 = self._linear_attention(q1, k2, v2)                   # T1 看 T2
        out2 = self._linear_attention(q2, k1, v1)                   # T2 看 T1

        out1 = self.attn_drop(out1)
        out2 = self.attn_drop(out2)

        # Reshape → 投影
        out1 = self.proj(out1.transpose(1, 2).reshape(B, N, C))    # [B, N, C]
        out2 = self.proj(out2.transpose(1, 2).reshape(B, N, C))

        # Reshape back to spatial
        out1 = out1.transpose(1, 2).reshape(B, C, H, W)
        out2 = out2.transpose(1, 2).reshape(B, C, H, W)

        # 残差连接 (gamma=0 初始化 → 模块初始为恒等, 渐进学习)
        return x1 + self.gamma * out1, x2 + self.gamma * out2


class PostFusionTemporalInteraction(nn.Module):
    """
    多尺度融合后时序交互 (PFTI)

    在金字塔的每个层级独立进行 T1↔T2 跨时序注意力.

    层级覆盖 (256×256 输入, 1m/px):
        - Level 0 (64×64 = 4096 tokens): 像素级精细变化
        - Level 1 (32×32 = 1024 tokens): 局部建筑变化
        - Level 2 (16×16 = 256 tokens):  区域建筑变化
        - Level 3 (8×8 = 64 tokens):     全局语义变化

    每个层级共享 QKV 投影参数 → 跨尺度参数共享 → 正则化
    """

    def __init__(self, channels_list, num_heads=4, dropout=0.0):
        """
        Args:
            channels_list: 各层级通道数, e.g. [128, 128, 128, 128]
            num_heads: 注意力头数
            dropout: attention dropout
        """
        super().__init__()
        self.levels = nn.ModuleList([
            LinearCrossAttention(c, num_heads=num_heads, dropout=dropout)
            for c in channels_list
        ])

        total_params = sum(p.numel() for p in self.parameters())
        print(f"[PFTI] PostFusionTemporalInteraction initialized:")
        print(f"  - {len(channels_list)} levels, num_heads={num_heads}")
        print(f"  - Total params: {total_params:,} ({total_params / 1e3:.1f}K)")

    def forward(self, pyramid1, pyramid2):
        """
        Args:
            pyramid1: list of T1 features [p1(64×64), p2(32×32), p3(16×16), p4(8×8)]
            pyramid2: list of T2 features (same structure)

        Returns:
            enhanced_pyramid1, enhanced_pyramid2
        """
        out1, out2 = [], []
        for level, f1, f2 in zip(self.levels, pyramid1, pyramid2):
            e1, e2 = level(f1, f2)
            out1.append(e1)
            out2.append(e2)
        return out1, out2
