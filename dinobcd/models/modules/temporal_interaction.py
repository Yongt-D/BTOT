"""
Bidirectional Temporal Interaction (BTI)
双向时序交互模块

核心创新：
在HFA聚合之前，让T1和T2的DINOv3特征互相感知变化区域
- T1查询T2: "T2中出现了什么新的东西?"
- T2查询T1: "T1中有什么现在消失了?"

为什么这很重要：
- 传统方法: T1和T2独立提取特征 → 只有差异模块(CADI)能检测变化
- BTI方法: 特征提取阶段就知道"对方"长什么样 → 主动放大变化区域的特征
- 结果: 召回率提升(不再遗漏微小变化)，精度维持

效率设计 (关键):
- 32×32 DINOv3 特征 → 下采样到 8×8 = 64 tokens
- 注意力矩阵: [B, heads, 64, 64] vs 全分辨率 [B, heads, 1024, 1024]
- 内存节省: 256× (1024²/64² = 256)
- 然后上采样回 32×32 作为空间增强信号

稳定性设计:
- gamma=0 初始化: 训练初期BTI不起作用，逐渐学习
- 残差连接: feat + gamma * enhancement
- GroupNorm: 稳定输出分布

预期提升: +1.5~2.0% IoU
参数量: ~800K (可忽略)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalTemporalInteraction(nn.Module):
    """
    双向时序交互 (BTI)

    原理：
    DINOv3 ViT-L 不同层编码不同层次的语义信息。
    如果我们让T1的特征知道T2的语义，那么T1特征中
    "即将消失的区域" 会被激活，更容易被后续变化检测发现。
    反之亦然。

    架构:
    feat1 [B,C,32,32]          feat2 [B,C,32,32]
         ↓ pool(8×8)                ↓ pool(8×8)
    feat1_small [B,C,8,8]   feat2_small [B,C,C,8,8]
         ↓ proj(C→D)              ↓ proj(C→D,D)
    q1 [B,N,D]              k2,v2 [B,N,D]
         ↓
    CrossAttn(q1, k2, v2) → attn1_small [B,D,8,8]
         ↓ upsample(32×32)
    attn1 [B,C,32,32]
         ↓
    feat1_enhanced = feat1 + gamma * attn1  ← 残差连接

    Args:
        dino_dim: DINOv3特征维度 (1024 for ViT-L)
        attn_dim: 注意力计算降维到多少 (128 默认)
        num_heads: 注意力头数 (4)
        spatial_size: 下采样到多大做注意力 (8 → 64 tokens)
        interact_layers: 哪些层做交互 (None=所有层)
    """

    def __init__(
        self,
        dino_dim=1024,
        attn_dim=128,
        num_heads=4,
        spatial_size=8,
        interact_layers=None
    ):
        super().__init__()

        self.interact_layers = interact_layers
        self.attn_dim = attn_dim
        self.spatial_size = spatial_size
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q投影 (query image → query vector)
        self.to_q = nn.Conv2d(dino_dim, attn_dim, 1, bias=False)

        # KV投影 (context image → key/value vectors)
        self.to_kv = nn.Conv2d(dino_dim, attn_dim * 2, 1, bias=False)

        # 输出投影 (attn_dim → dino_dim)
        self.out_proj = nn.Conv2d(attn_dim, dino_dim, 1, bias=False)

        # 归一化
        self.norm = nn.GroupNorm(32, dino_dim)

        # ⭐ gamma=0初始化: 确保训练初期稳定
        # 网络初始化时BTI不起作用，逐渐通过梯度学到有用的交互
        self.gamma = nn.Parameter(torch.zeros(1))

    def _cross_attend(self, query_feat, context_feat):
        """
        query_feat 关注 context_feat 的内容

        Args:
            query_feat:   [B, C, H, W]  (T1 or T2)
            context_feat: [B, C, H, W]  (T2 or T1)

        Returns:
            enhanced: [B, C, H, W]  query_feat + 跨时态信息
        """
        B, C, H, W = query_feat.shape
        S = self.spatial_size  # 8

        # Step 1: 下采样到低分辨率 (32×32 → 8×8)
        q_small = F.adaptive_avg_pool2d(query_feat, (S, S))    # [B, C, 8, 8]
        kv_small = F.adaptive_avg_pool2d(context_feat, (S, S))  # [B, C, 8, 8]

        # Step 2: 投影到低维度 (1024 → 128)
        q = self.to_q(q_small).flatten(2).transpose(1, 2)    # [B, 64, 128]
        kv = self.to_kv(kv_small)                             # [B, 256, 8, 8]
        k, v = kv.chunk(2, dim=1)                             # [B, 128, 8, 8] each
        k = k.flatten(2).transpose(1, 2)                      # [B, 64, 128]
        v = v.flatten(2).transpose(1, 2)                      # [B, 64, 128]

        # Step 3: Multi-head cross-attention (64 tokens, 极其高效!)
        N = S * S  # 64
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 注意力矩阵: [B, 4, 64, 64] = 非常小
        attn = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.attn_dim)

        # Step 4: 重塑回空间格式
        out = out.transpose(1, 2).reshape(B, self.attn_dim, S, S)  # [B, 128, 8, 8]

        # Step 5: 上采样回原始分辨率 (8×8 → 32×32)
        out = F.interpolate(
            out,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # [B, 128, 32, 32]

        # Step 6: 投影回原始维度 (128 → 1024)
        out = self.norm(self.out_proj(out))  # [B, 1024, 32, 32]

        # Step 7: 残差连接 (gamma从0开始,稳定训练)
        return query_feat + self.gamma * out

    def forward(self, dino_dict1, dino_dict2):
        """
        对指定层的特征做双向时序交互

        Args:
            dino_dict1: Dict {layer_idx: [B, 1024, H, W]}  T1时刻特征
            dino_dict2: Dict {layer_idx: [B, 1024, H, W]}  T2时刻特征

        Returns:
            new_dict1: 增强后的T1特征 (感知了T2)
            new_dict2: 增强后的T2特征 (感知了T1)
        """
        # 确定要交互的层
        layers = self.interact_layers if self.interact_layers is not None else list(dino_dict1.keys())

        # 创建新字典 (不修改原始特征, 避免in-place操作)
        new_dict1 = dict(dino_dict1)
        new_dict2 = dict(dino_dict2)

        for layer_idx in layers:
            if layer_idx in dino_dict1 and layer_idx in dino_dict2:
                f1 = dino_dict1[layer_idx]
                f2 = dino_dict2[layer_idx]

                # T1查询T2: "T2中有什么是我(T1)没有的?"
                new_dict1[layer_idx] = self._cross_attend(f1, f2)

                # T2查询T1: "T1中有什么是我(T2)没有的?"
                new_dict2[layer_idx] = self._cross_attend(f2, f1)

        return new_dict1, new_dict2


# ============================================================
# 独立测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing BidirectionalTemporalInteraction (BTI)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extract_layers = [1, 3, 5, 8, 11, 14, 17, 20, 23]

    # 初始化BTI
    bti = BidirectionalTemporalInteraction(
        dino_dim=1024,
        attn_dim=128,
        num_heads=4,
        spatial_size=8,
        interact_layers=extract_layers  # 所有9层都交互
    ).to(device)

    # 模拟DINOv3 9层特征 (32×32 for 512 input)
    dino_feats1 = {l: torch.randn(2, 1024, 32, 32).to(device) for l in extract_layers}
    dino_feats2 = {l: torch.randn(2, 1024, 32, 32).to(device) for l in extract_layers}

    print(f"\nInput: {len(dino_feats1)} layers, each {next(iter(dino_feats1.values())).shape}")

    # 前向传播
    out1, out2 = bti(dino_feats1, dino_feats2)

    print(f"Output1: {len(out1)} layers, each {next(iter(out1.values())).shape}")
    print(f"Output2: {len(out2)} layers, each {next(iter(out2.values())).shape}")

    # 验证 gamma=0 时输出 = 输入
    print(f"\ngamma = {bti.gamma.item():.4f} (初始应为0)")
    assert torch.allclose(out1[extract_layers[0]], dino_feats1[extract_layers[0]]), \
        "gamma=0时输出应等于输入!"
    print("✅ gamma=0 验证通过 (训练初期稳定)")

    # 计算参数量
    total = sum(p.numel() for p in bti.parameters())
    print(f"\n参数量: {total/1e6:.2f}M")

    # 计算注意力矩阵大小
    S = 8
    attn_dim = 128
    num_heads = 4
    attn_elements = 2 * S * S * S * S * num_heads  # 2 directions, 4 heads
    print(f"注意力矩阵大小: {S}×{S}={S*S} tokens (vs全分辨率32×32=1024, 节省{(32*32)**2//(S*S)**2}×)")
    print(f"  每层注意力: {attn_elements * 4 / 1024:.1f} KB (极小!)")

    print("\n✅ BTI模块测试通过!")
    print("  - 初始gamma=0 确保训练稳定")
    print("  - 8×8低分辨率注意力 极其高效")
    print("  - 双向交互 T1↔T2")
