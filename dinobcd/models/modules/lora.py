"""
LoRA - Low-Rank Adaptation for Foundation Model Backbone
低秩适配器 - 参数高效的骨干网络微调

核心创新:
    6轮实验证明: 差异模块/解码器改动对IoU的影响 < ±0.2% (0.820~0.824).
    瓶颈在于 DINOv3 骨干网络完全冻结, 无法学习"什么构成有意义的建筑变化".

    LoRA 解决方案:
    在冻结权重 W₀ ∈ R^{d×k} 旁注入低秩分解 ΔW = BA:
        h = W₀x + BAx
        其中 A ∈ R^{d×r}, B ∈ R^{r×k}, r << min(d,k)

    优势:
    - 参数量极小: rank=8, 4个ViT block → ~200K params (vs 冻结304M)
    - 保留预训练知识: W₀ 不变, 仅学习增量 ΔW
    - 变化感知: LoRA 让 DINOv3 特征变得对时序差异敏感
    - 训练稳定: A用Kaiming初始化, B用零初始化 → 初始 ΔW=0

数学分析:
    原始DINOv3特征: f = W₀x  (冻结, 对变化不敏感)
    LoRA适配特征:   f' = W₀x + (α/r)·BAx  (学习变化敏感的增量)

    α/r 是缩放因子:
    - α: LoRA缩放超参 (控制适配幅度)
    - r: 秩 (控制表达能力)
    - 典型设置: α=2r → 缩放=2

参考:
    - Hu et al., 2021 "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
    - 在视觉领域已成功应用于SAM, CLIP等基础模型的下游适配
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA-adapted Linear Layer

    将原始冻结的 nn.Linear 替换为:
        output = original(x) + (alpha/rank) * B(A(x))

    其中 A ∈ R^{in×r}, B ∈ R^{r×out} 是可训练的低秩矩阵.

    Args:
        original_linear: 原始的 nn.Linear (将被冻结)
        rank: 低秩分解的秩 (推荐4-16)
        alpha: 缩放因子 (推荐 2×rank)
        dropout: LoRA dropout概率
    """

    def __init__(self, original_linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original_linear
        # 暴露in_features/out_features (SelfAttention.compute_attention访问self.qkv.in_features)
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 确保原始权重冻结
        for param in self.original.parameters():
            param.requires_grad = False

        # LoRA低秩矩阵
        # A: 降维投影 (in → rank), Kaiming初始化
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        # B: 升维投影 (rank → out), 零初始化 → 初始ΔW=0
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # LoRA dropout (可选)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # 原始路径 (冻结) + LoRA路径 (可训练)
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return original_out + lora_out


def inject_lora(dino_model, target_blocks, rank=8, alpha=16, dropout=0.0,
                target_modules=('qkv', 'proj')):
    """
    向DINOv3模型的指定block注入LoRA

    Args:
        dino_model: DINOv3的DinoVisionTransformer模型
        target_blocks: 要注入LoRA的block索引列表 (e.g., [20, 21, 22, 23])
        rank: LoRA秩
        alpha: LoRA缩放
        dropout: LoRA dropout
        target_modules: 要注入LoRA的模块名列表
            - 'qkv': 注意力QKV投影
            - 'proj': 注意力输出投影
            - 'fc1'/'w1': MLP第一层
            - 'fc2'/'w3': MLP输出层

    Returns:
        lora_params: 所有LoRA参数列表 (用于优化器)
        num_lora_params: LoRA参数总量
    """
    # 找到blocks
    if hasattr(dino_model, 'blocks'):
        blocks = dino_model.blocks
    else:
        raise ValueError("Cannot find 'blocks' attribute in DINOv3 model")

    num_blocks = len(blocks)
    lora_params = []
    total_lora = 0

    for block_idx in target_blocks:
        if block_idx >= num_blocks:
            print(f"[LoRA] WARNING: block {block_idx} does not exist (model has {num_blocks} blocks), skipping")
            continue

        block = blocks[block_idx]

        # 注意力模块
        if hasattr(block, 'attn'):
            attn = block.attn

            # QKV投影
            if 'qkv' in target_modules and hasattr(attn, 'qkv'):
                original_qkv = attn.qkv
                # 处理 qkv 可能是 LinearKMaskedBias 或 nn.Linear
                if hasattr(original_qkv, 'linear'):
                    # LinearKMaskedBias: 内部有 .linear 属性
                    lora_layer = LoRALinear(original_qkv.linear, rank, alpha, dropout)
                    original_qkv.linear = lora_layer
                elif isinstance(original_qkv, nn.Linear):
                    lora_layer = LoRALinear(original_qkv, rank, alpha, dropout)
                    attn.qkv = lora_layer
                else:
                    # 尝试直接替换
                    lora_layer = LoRALinear(original_qkv, rank, alpha, dropout)
                    attn.qkv = lora_layer

                n_params = sum(p.numel() for p in [lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                total_lora += n_params
                lora_params.extend([lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                print(f"  [LoRA] Block {block_idx}.attn.qkv: rank={rank}, +{n_params:,} params")

            # 输出投影
            if 'proj' in target_modules and hasattr(attn, 'proj'):
                original_proj = attn.proj
                if isinstance(original_proj, nn.Linear):
                    lora_layer = LoRALinear(original_proj, rank, alpha, dropout)
                    attn.proj = lora_layer
                elif hasattr(original_proj, 'linear'):
                    lora_layer = LoRALinear(original_proj.linear, rank, alpha, dropout)
                    original_proj.linear = lora_layer
                else:
                    lora_layer = LoRALinear(original_proj, rank, alpha, dropout)
                    attn.proj = lora_layer

                n_params = sum(p.numel() for p in [lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                total_lora += n_params
                lora_params.extend([lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                print(f"  [LoRA] Block {block_idx}.attn.proj: rank={rank}, +{n_params:,} params")

        # MLP模块
        if hasattr(block, 'mlp'):
            mlp = block.mlp

            # SwiGLU FFN (w1, w2, w3)
            if hasattr(mlp, 'w1'):
                for name in ['w1', 'w2', 'w3']:
                    if name in target_modules and hasattr(mlp, name):
                        original = getattr(mlp, name)
                        if isinstance(original, nn.Linear):
                            lora_layer = LoRALinear(original, rank, alpha, dropout)
                            setattr(mlp, name, lora_layer)
                            n_params = sum(p.numel() for p in [lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                            total_lora += n_params
                            lora_params.extend([lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                            print(f"  [LoRA] Block {block_idx}.mlp.{name}: rank={rank}, +{n_params:,} params")
            else:
                # Standard MLP (fc1, fc2)
                for name in ['fc1', 'fc2']:
                    if name in target_modules and hasattr(mlp, name):
                        original = getattr(mlp, name)
                        if isinstance(original, nn.Linear):
                            lora_layer = LoRALinear(original, rank, alpha, dropout)
                            setattr(mlp, name, lora_layer)
                            n_params = sum(p.numel() for p in [lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                            total_lora += n_params
                            lora_params.extend([lora_layer.lora_A.weight, lora_layer.lora_B.weight])
                            print(f"  [LoRA] Block {block_idx}.mlp.{name}: rank={rank}, +{n_params:,} params")

    print(f"\n[LoRA] Total LoRA parameters: {total_lora:,} ({total_lora/1e3:.1f}K)")
    print(f"[LoRA] Target blocks: {target_blocks}")
    print(f"[LoRA] Target modules: {target_modules}")
    print(f"[LoRA] Rank: {rank}, Alpha: {alpha}")

    return lora_params, total_lora
