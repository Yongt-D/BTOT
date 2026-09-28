"""
DINOBCD - DINOv3 Feature Extractor Wrapper
完全独立实现，用于DINOBCD研究

核心功能：
1. 封装DINOv3 ViT-L/16模型
2. 从多个层提取多尺度特征
3. 支持LoRA适配 (参数高效微调)
4. 冻结DINOv3基础参数，仅LoRA参数可训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from .lora import inject_lora


class DINOv3FeatureExtractor(nn.Module):
    """
    DINOv3特征提取器 (支持LoRA适配)

    Args:
        weights_path: DINOv3预训练权重路径
        extract_layers: 提取特征的层索引，默认[5, 11, 17, 23]
        input_size: DINOv3输入尺寸，默认512（匹配预训练）
        device: 计算设备
        lora_config: LoRA配置字典 (None=不使用LoRA)
            - enable: 是否启用LoRA
            - rank: LoRA秩 (推荐4-16)
            - alpha: LoRA缩放 (推荐2*rank)
            - target_blocks: 注入LoRA的block索引列表
            - target_modules: 注入LoRA的模块名列表
            - dropout: LoRA dropout
    """

    def __init__(
        self,
        weights_path="dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        extract_layers=[5, 11, 17, 23],
        input_size=512,
        device="cuda",
        lora_config=None,
        unfreeze_config=None
    ):
        super().__init__()

        self.device = device
        self.extract_layers = extract_layers
        self.input_size = input_size

        # 加载DINOv3模型
        model_name = "dinov3_vitl16"
        self.dino_model = torch.hub.load(
            "dinov3",
            model_name,
            source="local",
            weights=weights_path
        )

        # 设置为评估模式并移到设备
        self.dino_model.eval().to(device)

        # 获取模型信息
        self.num_layers = self._get_num_layers(model_name)
        self.patch_size = self._get_patch_size(model_name)
        self.feature_dim = 1024  # ViT-L的特征维度

        # 冻结所有参数
        for param in self.dino_model.parameters():
            param.requires_grad = False

        # ⭐ LoRA适配 (在冻结之后注入, LoRA参数自动可训练)
        self.use_lora = False
        self.lora_params = []
        if lora_config is not None and lora_config.get('enable', False):
            self._setup_lora(lora_config)

        # ⭐ Backbone Partial Unfreezing (更简单稳定的替代LoRA方案)
        # 直接解冻最后N个ViT block, 无需新参数, 梯度直接流过原始权重
        # 比LoRA更稳定: 无缩放因子噪声, 无初始化问题, 标准迁移学习实践
        self.use_unfreeze = False
        self.backbone_params = []
        if unfreeze_config is not None and unfreeze_config.get('enable', False):
            self._setup_unfreeze(unfreeze_config)

    def _setup_lora(self, lora_config):
        """注入LoRA到DINOv3骨干网络"""
        rank = lora_config.get('rank', 8)
        alpha = lora_config.get('alpha', 16)
        dropout = lora_config.get('dropout', 0.0)
        target_blocks = lora_config.get('target_blocks', [20, 21, 22, 23])
        target_modules = lora_config.get('target_modules', ['qkv', 'proj'])

        print(f"\n[LoRA] Injecting LoRA into DINOv3 backbone...")
        self.lora_params, num_lora = inject_lora(
            self.dino_model,
            target_blocks=target_blocks,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=tuple(target_modules)
        )
        self.use_lora = True

        # 验证: 确保LoRA参数可训练
        trainable_lora = sum(1 for p in self.lora_params if p.requires_grad)
        print(f"[LoRA] Trainable LoRA param tensors: {trainable_lora}/{len(self.lora_params)}")

    def get_lora_params(self):
        """返回LoRA参数列表 (用于创建优化器参数组)"""
        return self.lora_params

    def _setup_unfreeze(self, unfreeze_config):
        """解冻DINOv3骨干网络最后N个block (标准迁移学习)

        与LoRA对比:
        - LoRA: 添加新参数(A,B矩阵) + 缩放因子 → 容易不稳定
        - Unfreeze: 直接训练原始权重 → 渐进式适配, 更稳定

        数学分析:
        LoRA: h = W₀x + (α/r)·BAx  (新参数从零开始, 缩放可能放大噪声)
        Unfreeze: h = (W₀ + ΔW)x    (ΔW由优化器控制, 步长=lr很小)

        当backbone_lr << main_lr时, ΔW变化极缓慢, 下游模块能逐步适配.
        """
        num_blocks_to_unfreeze = unfreeze_config.get('num_blocks', 4)
        num_blocks = len(self.dino_model.blocks)
        start_block = max(0, num_blocks - num_blocks_to_unfreeze)

        print(f"\n[Unfreeze] Unfreezing DINOv3 blocks [{start_block}-{num_blocks-1}]...")

        self.backbone_params = []
        for i in range(start_block, num_blocks):
            block = self.dino_model.blocks[i]
            for param in block.parameters():
                param.requires_grad = True
            block_params = sum(p.numel() for p in block.parameters())
            self.backbone_params.extend(list(block.parameters()))
            print(f"  [Unfreeze] Block {i}: {block_params:,} params unfrozen")

        # 同时解冻最终的LayerNorm (用于特征归一化)
        if hasattr(self.dino_model, 'norm'):
            for param in self.dino_model.norm.parameters():
                param.requires_grad = True
            norm_params = sum(p.numel() for p in self.dino_model.norm.parameters())
            self.backbone_params.extend(list(self.dino_model.norm.parameters()))
            print(f"  [Unfreeze] Final LayerNorm: {norm_params:,} params unfrozen")

        total_unfrozen = sum(p.numel() for p in self.backbone_params)
        self.use_unfreeze = True
        print(f"[Unfreeze] Total unfrozen backbone params: {total_unfrozen:,} ({total_unfrozen/1e6:.2f}M)")
        print(f"[Unfreeze] Frozen backbone params: {sum(p.numel() for p in self.dino_model.parameters() if not p.requires_grad):,}")

    def get_backbone_params(self):
        """返回解冻的骨干网络参数列表 (用于创建优化器参数组)"""
        return self.backbone_params

    def _get_num_layers(self, model_name):
        """根据模型名称获取层数"""
        model_type = re.sub(r"\d+", "", model_name.split("_")[-1]).upper()
        layer_mapping = {
            "VITS": 12,
            "VITSP": 12,
            "VITB": 12,
            "VITL": 24,
            "VITHP": 32,
            "VIT7B": 40
        }
        return layer_mapping.get(model_type, 24)

    def _get_patch_size(self, model_name):
        """从模型名称提取patch size"""
        numbers = re.findall(r"\d+", model_name.split("_")[-1])
        return int(numbers[-1]) if numbers else 16

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            features: 多层特征列表，每个元素为 [B, C, H', W']
        """
        B, _, H, W = x.shape

        # 上采样到512x512以匹配DINOv3预训练尺寸
        if H != self.input_size or W != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=True,
                antialias=True
            )

        # autocast需要device_type为"cuda"或"cpu", 不能是"cuda:0"
        device_type = self.device.split(':')[0] if isinstance(self.device, str) else 'cuda'

        if self.use_lora or self.use_unfreeze:
            # ⭐ 适配模式: 需要梯度计算 (LoRA/解冻参数需要训练)
            # 冻结层的权重仍然 requires_grad=False, 不消耗额外梯度内存
            with torch.autocast(device_type=device_type, dtype=torch.float32):
                selected_features = self.dino_model.get_intermediate_layers(
                    x,
                    n=self.extract_layers,
                    reshape=True,
                    norm=True
                )
                selected_features = list(selected_features)
        else:
            # 完全冻结模式: 无梯度计算，节省显存
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.float32):
                    selected_features = self.dino_model.get_intermediate_layers(
                        x,
                        n=self.extract_layers,
                        reshape=True,
                        norm=True
                    )
                    selected_features = list(selected_features)

        return selected_features

    def get_feature_info(self):
        """返回特征信息"""
        feat_h = feat_w = self.input_size // self.patch_size
        return {
            'num_layers': self.num_layers,
            'extract_layers': self.extract_layers,
            'feature_dim': self.feature_dim,
            'patch_size': self.patch_size,
            'feature_resolution': (feat_h, feat_w),
            'use_lora': self.use_lora
        }


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = DINOv3FeatureExtractor(device=device)

    # 打印模型信息
    print("DINOv3 Feature Extractor Info:")
    info = extractor.get_feature_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256).to(device)
    features = extractor(x)

    print("\nExtracted Features:")
    for i, feat in enumerate(features):
        print(f"  Layer {extractor.extract_layers[i]}: {feat.shape}")
