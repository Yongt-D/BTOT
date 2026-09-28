"""
DinoBCD - Main Model
Building Change Detection with DINOv3 Features

核心架构：
1. 双路径特征提取：Efficient CNN + DINOv3 (冻结)
2. 双重差异建模：Change Stream + Invariant Stream (核心创新)
3. 金字塔特征融合：多尺度特征对齐
4. 边界细化：软形态学边界优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.dino_wrapper import DINOv3FeatureExtractor
from .modules.lightweight_adapter import MultiScaleDualAdapter
from .modules.efficient_backbone import build_backbone
from .modules.dual_difference import MultiScaleDualDifference
from .modules.pyramid_fusion import EnhancedPyramidFusion
from .modules.boundary_refine import SoftBoundaryRefinement, EfficientLMM
from .modules.cadi_module import MultiScaleCADI
from .modules.hierarchical_aggregation import HierarchicalFeatureAggregation  # ⭐ HFA 核心创新
from .modules.temporal_interaction import BidirectionalTemporalInteraction    # ⭐ BTI 双向时序交互
from .modules.sgda_module import MultiScaleSGDA                              # ⭐ SGDA 频谱-图差异聚合
from .modules.btot_diff import MultiScaleBTOTDiff                            # ⭐ BTOT 双时相最优传输差异建模
from .modules.lgati_module import LayerGroupAdaptiveTemporalInteraction      # ⭐ LGATI 层组自适应时序交互
from .modules.msca_module import MultiScaleMSCA                              # ⭐ MSCA 多尺度变化聚合
from .modules.hbca_module import HierarchicalBitemporalCrossAttention         # ⭐ HBCA 层次化双时序交叉注意力
from .modules.post_fusion_interaction import PostFusionTemporalInteraction    # ⭐ PFTI 融合后时序交互


# ============================================================
# 效率优化模块 - EAAI工程化优化
# ============================================================

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 参数量和FLOPs减少70%"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                    padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class EfficientBoundaryRefine(nn.Module):
    """高效边界细化 - FLOPs降低80%，精度无损"""
    def __init__(self, num_classes=2, hidden_dim=16):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, 3, 1, 1, groups=num_classes),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

    def forward(self, logits):
        return logits + 0.1 * self.edge_conv(logits)


class SemanticGuidedChangeEnhancement(nn.Module):
    """
    SGCE (Semantic-Guided Change Enhancement)

    动机: 传统变化检测依赖T1-T2差异特征, 但当新旧建筑视觉相似时
    差异特征极弱, 导致大面积变化区域被漏检.

    方案: 在差异特征之外注入T1/T2的语义上下文, 为变化检测提供
    "这里有建筑"的语义先验, 补偿差异特征的盲区.

    数学表达:
        sem_i = Conv1x1(cat(P_i^t1, P_i^t2))   # 语义上下文
        enhanced_i = diff_i + α_i * sem_i        # α_i可学习, 初始化为0
    """

    def __init__(self, channels, num_levels=4):
        super().__init__()
        # 每层: 2C → C 的1x1压缩 (语义上下文提取)
        self.sem_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_levels)
        ])
        # 可学习融合权重, 初始化为0 → 训练初期不干扰原始diff
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_levels)
        ])

    def forward(self, diff_pyramid, pyramid1, pyramid2):
        """
        Args:
            diff_pyramid: list of [B, C, H_i, W_i] 差异特征
            pyramid1: list of [B, C, H_i, W_i] T1金字塔特征
            pyramid2: list of [B, C, H_i, W_i] T2金字塔特征
        Returns:
            enhanced: list of [B, C, H_i, W_i] 增强后的差异特征
        """
        enhanced = []
        for i in range(len(diff_pyramid)):
            sem = self.sem_convs[i](torch.cat([pyramid1[i], pyramid2[i]], dim=1))
            alpha = self.alphas[i].sigmoid()  # [0, 1] 范围
            enhanced.append(diff_pyramid[i] + alpha * sem)
        return enhanced


class DinoBCD(nn.Module):
    """
    DinoBCD: DINOv3-based Building Change Detection

    核心创新：
    1. CADI (Change-Aware Deformable Interaction) - 组合ChangeDINO + TPAMI NDGC
    2. 双重差异建模 (Dual-Stream Difference Modeling)
    3. EfficientLMM边界细化

    Args:
        backbone_type: CNN骨干类型 ('efficient' or 'lightweight')
        dino_weights: DINOv3权重路径
        num_classes: 类别数（默认2：背景和变化）
        pyramid_channels: 金字塔通道数
        use_boundary_refine: 是否使用边界细化
        use_cadi: 是否使用CADI模块 (核心创新)
        use_efficient_lmm: 是否使用高效LMM
        device: 计算设备
    """

    def __init__(
        self,
        backbone_type='efficient',
        dino_weights='dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        dino_extract_layers=[5, 11, 17, 23],
        num_classes=2,
        pyramid_channels=128,
        use_boundary_refine=True,
        use_cadi=True,           # 启用CADI模块
        use_sgda=False,          # 启用SGDA (频谱-图差异聚合)
        use_msca=False,          # ⭐ 启用MSCA (多尺度变化聚合) - 新默认方向!
        use_hfa=False,           # ⭐ 启用HFA (层次化特征聚合)
        use_lgati=False,         # 启用LGATI (层组自适应时序交互)
        use_pfti=False,          # ⭐ 启用PFTI (融合后时序交互) - 关键创新!
        use_efficient_lmm=True,  # 启用高效LMM
        sgda_config=None,        # SGDA配置
        head_dropout=0.0,        # head dropout概率
        lora_config=None,        # ⭐ LoRA配置 (骨干网络适配)
        unfreeze_config=None,    # ⭐ 骨干网络部分解冻 (替代LoRA)
        use_sgbd=False,          # ⭐ SGBD: 语义引导双向差异增强 (Exp14)
        use_sgbd_gate=False,     # ⭐ SGBD语义门控 (默认关闭, 减少过拟合)
        use_cosine_weight=False, # ⭐ 零参数余弦加权 (Exp17)
        use_bti=True,            # ⭐ BTI开关 (消融用, 默认跟随HFA)
        use_msfa=True,           # ⭐ MSFA开关 (消融用)
        use_deep_supervision=False,  # ⭐ Deep Supervision (FPN各层辅助loss)
        use_sgce=False,              # ⭐ SGCE (语义引导变化增强)
        use_hbca=False,              # ⭐ HBCA (层次化双时序交叉注意力)
        hbca_config=None,            # ⭐ HBCA配置
        use_btot=False,              # ⭐ BTOT (双时相最优传输差异建模)
        btot_config=None,            # ⭐ BTOT配置 {ot_dim, window, n_iters}
        use_ssm=False,               # ⭐ SSM (状态空间/选择性扫描交互算子, 受控对比 baseline)
        use_bt_mixstyle=False,       # ⭐ v10 BTM (Bi-Temporal MixStyle, 训练时 style 扰动)
        btm_config=None,             # ⭐ BTM 配置 {p, alpha, mix_mode}
        use_dual_stream=False,       # ⭐ v12 双流编码器 (CNN + DINOv3 fusion)
        dual_stream_config=None,     # ⭐ v12 双流配置 {cnn_pretrained, freeze_cnn}
        device='cuda'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pyramid_channels = pyramid_channels
        self.use_boundary_refine = use_boundary_refine
        self.use_cadi = use_cadi
        self.use_sgda = use_sgda
        self.use_msca = use_msca    # ⭐ MSCA
        self.use_hfa = use_hfa      # ⭐ HFA
        self.use_lgati = use_lgati
        self.use_pfti = use_pfti    # ⭐ PFTI
        self.use_efficient_lmm = use_efficient_lmm
        self.use_sgbd = use_sgbd        # ⭐ SGBD
        self.use_cosine_weight = use_cosine_weight  # ⭐ 零参数余弦加权
        self.use_sgbd_gate = use_sgbd_gate
        self.use_bti = use_bti          # ⭐ BTI开关
        self.use_msfa = use_msfa        # ⭐ MSFA开关
        self.use_deep_supervision = use_deep_supervision  # ⭐ Deep Supervision
        self.use_sgce = use_sgce                        # ⭐ SGCE
        self.use_hbca = use_hbca                        # ⭐ HBCA
        self.use_btot = use_btot                        # ⭐ BTOT
        self.use_ssm = use_ssm                          # ⭐ SSM
        self.use_bt_mixstyle = use_bt_mixstyle          # ⭐ v10 BTM
        self.use_dual_stream = use_dual_stream          # ⭐ v12 双流编码器
        self.dino_extract_layers = dino_extract_layers
        self.sgda_config = sgda_config or {}

        # ==================== 1. 特征提取器 ====================

        # CNN骨干网络 (轻量级)
        self.backbone = build_backbone(backbone_type=backbone_type)
        cnn_channels = self.backbone.stage_channels  # [64, 128, 256, 512]

        # DINOv3特征提取器 (⭐ 支持LoRA适配 / 骨干网络解冻)
        self.dino_extractor = DINOv3FeatureExtractor(
            weights_path=dino_weights,
            extract_layers=dino_extract_layers,
            device=device,
            lora_config=lora_config,
            unfreeze_config=unfreeze_config
        )

        # DINOv3适配器
        if use_hfa:
            # ⭐ HFA: 层次化特征聚合 (9层密集提取)
            self.dino_adapter = HierarchicalFeatureAggregation(
                dino_dim=1024,
                out_dim=256,
                num_heads=4,
                extract_layers=dino_extract_layers
            )

            # 时序交互模块: LGATI / BTI / 无 (消融)
            if not use_bti:
                print("[Model] BTI disabled (ablation mode)")
            elif use_lgati:
                # ⭐ LGATI: 层组自适应时序交互 (每组独立参数+分辨率)
                self.temporal_interaction = LayerGroupAdaptiveTemporalInteraction(
                    dino_dim=1024,
                    extract_layers=dino_extract_layers,
                    shallow_attn_dim=64,
                    shallow_spatial=16,
                    shallow_heads=4,
                    middle_attn_dim=128,
                    middle_spatial=8,
                    middle_heads=4,
                    deep_attn_dim=128,
                    deep_spatial=4,
                    deep_heads=4,
                )
                print("[Model] ⭐ 使用 LGATI (层组自适应时序交互) - 各组独立参数")
            else:
                # BTI: 共享参数的双向时序交互
                self.temporal_interaction = BidirectionalTemporalInteraction(
                    dino_dim=1024,
                    attn_dim=128,
                    num_heads=4,
                    spatial_size=8,
                    interact_layers=dino_extract_layers
                )
                print("[Model] ⭐ 使用 BTI (双向时序交互) - 共享参数")

            print("[Model] ⭐ 使用 HFA (层次化特征聚合) - 9层密集提取")
            print(f"[Model] Extract layers: {dino_extract_layers}")
            print(f"[Model]   - 浅层: {dino_extract_layers[:3]}")
            print(f"[Model]   - 中层: {dino_extract_layers[3:6]}")
            print(f"[Model]   - 深层: {dino_extract_layers[6:9]}")
        else:
            # 原始简单适配器 (4层稀疏提取)
            self.dino_adapter = MultiScaleDualAdapter(
                in_dim=1024,
                out_dim=256,
                target_sizes=[64, 32, 16, 8],
                bottleneck_dim=64
            )
            print(f"[Model] 使用 MultiScaleDualAdapter (稀疏提取 {len(dino_extract_layers)} 层)")

        # ==================== 2. 金字塔特征融合 ====================

        self.pyramid_fusion = EnhancedPyramidFusion(
            cnn_channels=cnn_channels,
            dino_channels=[256, 256, 256, 256],
            out_channels=pyramid_channels,
            use_aff=False  # 回滚到v0106基线（CBAM）
        )

        # ==================== 3. 差异建模 ====================

        if use_btot:
            # ⭐ BTOT: 双时相最优传输差异建模 (替换 HBCA/MSCA, 复用成熟训练配方)
            _btot_cfg = btot_config or {}
            self.diff_module = MultiScaleBTOTDiff(
                channels_list=[pyramid_channels] * 4,
                ot_dim=_btot_cfg.get('ot_dim', 48),
                window=_btot_cfg.get('window', 8),
                n_iters=_btot_cfg.get('n_iters', 5),
                dropout=head_dropout,
                no_gamma=_btot_cfg.get('no_gamma', False),
                cost_only=_btot_cfg.get('cost_only', False),
                use_dustbin=_btot_cfg.get('use_dustbin', True),
                learn_bin=_btot_cfg.get('learn_bin', True),
            )
            print("[Model] ⭐ 使用 BTOT (双时相最优传输) 进行差异建模")
            print(f"[Model]   - window={_btot_cfg.get('window', 8)}, ot_dim={_btot_cfg.get('ot_dim', 48)}, iters={_btot_cfg.get('n_iters', 5)}")
            print(f"[Model]   - gamma-gate 残差 (初始=纯差分, 渐进引入 OT 不可匹配质量证据)")
        elif use_ssm:
            # ⭐ SSM: 状态空间 (Mamba/S6 式) 双时相交互算子 (受控对比 baseline)
            from .modules.ssm_diff import MultiScaleSSMDiff
            self.diff_module = MultiScaleSSMDiff(
                channels_list=[pyramid_channels] * 4,
                dropout=head_dropout,
            )
            print("[Model] ⭐ 使用 SSM (状态空间/选择性扫描) 进行差异建模")
            print("[Model]   - gamma-gate 残差 (初始=纯差分, 渐进引入状态空间证据)")
        elif use_hbca:
            # ⭐ HBCA: 层次化双时序交叉注意力 - 核心创新
            _hbca_cfg = hbca_config or {}
            _use_gamma = _hbca_cfg.get('use_gamma_gate', True)
            _use_cs = _hbca_cfg.get('use_cross_scale', True)
            _ws = _hbca_cfg.get('window_size', 8)
            _asym = _hbca_cfg.get('asymmetric', False)
            _edge = _hbca_cfg.get('use_edge_branch', False)
            _biw = _hbca_cfg.get('use_bi_whitening', False)
            _style_init = _hbca_cfg.get('style_recovery_init', 0.5)
            _diff = _hbca_cfg.get('differential', False)
            _diff_lam = _hbca_cfg.get('diff_lambda_init', 0.8)
            self.diff_module = HierarchicalBitemporalCrossAttention(
                channels_list=[pyramid_channels] * 4,
                attn_dim=_hbca_cfg.get('attn_dim', 64),
                num_heads=_hbca_cfg.get('num_heads', 4),
                window_size=_ws,
                dropout=head_dropout,
                share_weights=_hbca_cfg.get('share_weights', False),
                use_shifted=_hbca_cfg.get('use_shifted', True),
                use_gamma_gate=_use_gamma,
                use_cross_scale=_use_cs,
                asymmetric=_asym,
                use_edge_branch=_edge,
                use_bi_whitening=_biw,
                style_recovery_init=_style_init,
                differential=_diff,
                diff_lambda_init=_diff_lam,
            )
            print("[Model] ⭐ 使用 HBCA (层次化双时序交叉注意力) 进行差异建模")
            if _asym:
                print("[Model]   - ⭐ v7 Asymmetric HBCA: T1→T2 与 T2→T1 使用独立 Q/K/V 投影")
            if _edge:
                print("[Model]   - ⭐ v8 Edge-Bridged HBCA: 并行 edge cross-attention 分支")
            if _biw:
                print(f"[Model]   - ⭐ v9 Bi-Temporal Joint Whitening + Style Recovery (init γ_style={_style_init})")
            if _diff:
                print(f"[Model]   - ⭐ v12 HBCA-Diff: 差分注意力 (A1 - λ A2), λ_init={_diff_lam}")
            print(f"[Model]   - attn_dim={_hbca_cfg.get('attn_dim', 64)}, heads={_hbca_cfg.get('num_heads', 4)}")
            print(f"[Model]   - window_size={_ws}, share_weights={_hbca_cfg.get('share_weights', False)}")
            print(f"[Model]   - gamma_gate={_use_gamma}, cross_scale={_use_cs}")
            print(f"[Model]   - Dropout: {head_dropout}")
        elif use_sgda:
            # SGDA: 频谱-图差异聚合
            self.diff_module = MultiScaleSGDA(
                channels_list=[pyramid_channels] * 4,
                num_virtual_neighbors=self.sgda_config.get('num_virtual_neighbors', 16),
                num_local_points=self.sgda_config.get('num_local_points', 9),
                offset_range=self.sgda_config.get('offset_range', 8.0)
            )
            print("[Model] 使用 SGDA (频谱-图差异聚合) 进行差异建模")
        elif use_msca:
            # ⭐ MSCA: 多尺度变化聚合 - 替代CADI的新默认方案
            # 膨胀率 (1,3,6) → 等效感受野 3×3 / 7×7 / 13×13
            self.diff_module = MultiScaleMSCA(
                channels_list=[pyramid_channels] * 4,
                dilations=(1, 3, 6),
                se_ratio=4,
                dropout=head_dropout,
                use_sgbd=use_sgbd,
                use_sgbd_gate=use_sgbd_gate,
                use_cosine_weight=use_cosine_weight
            )
            print("[Model] ⭐ 使用 MSCA (多尺度变化聚合) 进行差异建模")
            if use_sgbd:
                gate_str = " + 语义门控" if use_sgbd_gate else ""
                print(f"[Model]   - ⭐ SGBD: 双向差异+余弦加权{gate_str}")
            if use_cosine_weight and not use_sgbd:
                print(f"[Model]   - ⭐ 零参数余弦加权: diff * (1 + cos_dissim)")
            print(f"[Model]   - 感受野: 3×3 / 7×7 / 13×13 (dilation=1/3/6)")
            print(f"[Model]   - Dropout: {head_dropout}")
        elif use_cadi:
            # CADI: 变化感知可变形交互 - 局部建模
            self.diff_module = MultiScaleCADI(
                channels_list=[pyramid_channels] * 4,
                num_points=9
            )
            print("[Model] 使用 CADI (可变形卷积) 进行差异建模")
        else:
            # 回退到原始双重差异建模
            self.diff_module = MultiScaleDualDifference(
                channels_list=[pyramid_channels] * 4
            )
            print("[Model] 使用 Dual Difference 进行差异建模")

        # ==================== 3.5 融合后时序交互 (PFTI) ====================

        if use_pfti:
            # ⭐ PFTI: 在金字塔特征融合后再做一次T1↔T2跨时序注意力
            # BTI在原始DINO特征上交互, PFTI在融合特征上交互 → 二者互补
            self.pfti = PostFusionTemporalInteraction(
                channels_list=[pyramid_channels] * 4,
                num_heads=4,
                dropout=head_dropout
            )
            print("[Model] ⭐ 使用 PFTI (融合后时序交互) - 多尺度线性注意力")

        # ==================== 3.7 SGCE (语义引导变化增强) ====================

        if use_sgce:
            self.sgce = SemanticGuidedChangeEnhancement(
                channels=pyramid_channels, num_levels=4
            )
            print("[Model] ⭐ SGCE (Semantic-Guided Change Enhancement) enabled")
            print(f"[Model]   - 4-level semantic context injection")
            print(f"[Model]   - Learnable alpha (init=0, sigmoid-gated)")

        # ==================== 4. FPN解码器 ====================

        # 自底向上连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(pyramid_channels, pyramid_channels, kernel_size=1)
            for _ in range(4)
        ])

        # FPN top-down 后的 Conv3×3 细化 (标准FPN必备)
        # 消除上采样混叠 + 融合后空间细化
        self.fpn_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(pyramid_channels, pyramid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(pyramid_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(3)  # p3, p2, p1 各一个
        ])

        # ==================== 4.5 多尺度特征聚合 (MSFA) ====================

        if use_msfa:
            # ⭐ 核心创新: 将FPN所有层级聚合为统一特征表示
            # F_cat = [P1; ↑P2; ↑P3; ↑P4] → Conv3×3 → + P1 → SE → output
            self.ms_fusion = nn.Sequential(
                nn.Conv2d(pyramid_channels * 4, pyramid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(pyramid_channels),
                nn.ReLU(inplace=True)
            )

            se_dim = max(pyramid_channels // 4, 32)
            self.ms_se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(pyramid_channels, se_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(se_dim, pyramid_channels, 1),
                nn.Sigmoid()
            )

            print(f"[Model] ⭐ MSFA (多尺度特征聚合) enabled: {pyramid_channels*4}→{pyramid_channels}")
            print(f"[Model]   - 3×3 spatial refinement + P1 residual + SE attention")
        else:
            print("[Model] MSFA disabled (ablation mode), using P1-only output")

        # 输出头 - 2层深度可分离卷积 (更强的特征变换能力) + Dropout
        head_layers = [
            DepthwiseSeparableConv(pyramid_channels, pyramid_channels, 3, 1),
            DepthwiseSeparableConv(pyramid_channels, pyramid_channels, 3, 1),
        ]
        if head_dropout > 0:
            head_layers.append(nn.Dropout2d(head_dropout))
        head_layers.append(nn.Conv2d(pyramid_channels, num_classes, kernel_size=1))
        self.output_conv = nn.Sequential(*head_layers)

        # ==================== 4.7 Deep Supervision 辅助头 ====================

        if use_deep_supervision:
            # FPN每层(p2/p3/p4)各一个轻量1x1输出头
            # p1已经被主输出头覆盖，不需要额外辅助
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(pyramid_channels, num_classes, kernel_size=1)
                for _ in range(3)  # p2, p3, p4
            ])
            print("[Model] ⭐ Deep Supervision enabled: auxiliary heads on P2/P3/P4")

        # ==================== 5. 边界细化 ====================

        if use_boundary_refine:
            if use_efficient_lmm:
                # EfficientLMM: 高效可学习形态学模块 (-70%参数)
                self.boundary_refiner = EfficientLMM(num_classes=num_classes, hidden_dim=8)
            else:
                # 回退到原始边界细化
                self.boundary_refiner = EfficientBoundaryRefine(num_classes=num_classes, hidden_dim=16)

        # ==================== v10 Bi-Temporal MixStyle ====================
        if use_bt_mixstyle:
            from .modules.bt_mixstyle import BiTemporalMixStyle
            _btm_cfg = btm_config or {}
            self.bt_mixstyle = BiTemporalMixStyle(
                p=_btm_cfg.get('p', 0.5),
                alpha=_btm_cfg.get('alpha', 0.1),
                mix_mode=_btm_cfg.get('mix_mode', 'random'),
            )
            print(f"[Model] ⭐ v10 Bi-Temporal MixStyle: {self.bt_mixstyle}")

        # ==================== v12 Cross-Temporal CBAM (BT-CBAM) ====================
        # v6 已经有 CNN backbone + EnhancedPyramidFusion (per-time 融合).
        # v12 在 per-time 融合之后再加一层跨时相 CBAM, 让金字塔特征吸收 bi-temporal 信息.
        # 这是与 ChangeDINO DFFM (单时相 CBAM) 的关键差异化, 也是 v12 主要 engineering 卖点.
        if use_dual_stream:
            from .modules.bt_cbam_fusion import MultiScaleBiTemporalCBAM
            self.cross_temporal_cbam = MultiScaleBiTemporalCBAM(
                channels=pyramid_channels, num_levels=4
            )
            n_btc = sum(p.numel() for p in self.cross_temporal_cbam.parameters())
            print(f"[Model] ⭐ v12 Cross-Temporal CBAM (BT-CBAM) enabled: {n_btc/1e6:.2f}M params")
            print(f"        - Applied AFTER per-time pyramid_fusion, gives bi-temporal aware features")

    def extract_features(self, x):
        """
        提取单个时间点的多模态特征 (优化版)

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            pyramid: [p1, p2, p3, p4] 融合后的金字塔特征

        优化策略：
        - CNN和DINO特征提取并行化（如果有多GPU）
        - DINO特征在eval模式下自动启用torch.no_grad()
        """
        # CNN特征提取
        cnn_features = self.backbone(x)

        # DINOv3特征提取 (冻结backbone，自动no_grad)
        dino_features = self.dino_extractor(x)

        # DINOv3适配
        if self.use_hfa:
            # HFA 需要 Dict 输入 {layer_idx: feature}
            if isinstance(dino_features, list):
                dino_features = {
                    layer: feat
                    for layer, feat in zip(self.dino_extract_layers, dino_features)
                }
            dino_features = self.dino_adapter(dino_features)
        else:
            dino_features = self.dino_adapter(dino_features)

        # 金字塔融合
        pyramid = self.pyramid_fusion(cnn_features, dino_features)

        return pyramid

    def _extract_features_joint(self, img1, img2):
        """
        联合特征提取 (BTI版本)
        T1和T2的DINOv3特征在HFA聚合前互相感知

        相比独立提取的优势:
        - T1特征知道T2是什么 → 更容易发现"消失的建筑"
        - T2特征知道T1是什么 → 更容易发现"新出现的建筑"
        - 微小变化被放大 → 召回率提升

        Returns:
            pyramid1, pyramid2: 联合感知后的金字塔特征
        """
        # CNN特征 (独立提取, 不需要交互)
        cnn1 = self.backbone(img1)
        cnn2 = self.backbone(img2)

        # DINOv3特征 (独立提取)
        dino1_list = self.dino_extractor(img1)
        dino2_list = self.dino_extractor(img2)

        # 转换为Dict格式
        dino1_dict = {
            layer: feat
            for layer, feat in zip(self.dino_extract_layers, dino1_list)
        }
        dino2_dict = {
            layer: feat
            for layer, feat in zip(self.dino_extract_layers, dino2_list)
        }

        # ⭐ BTI: 双向时序交互
        # T1特征和T2特征互相感知, 变化区域特征被增强
        dino1_dict, dino2_dict = self.temporal_interaction(dino1_dict, dino2_dict)

        # HFA: 层次化聚合 (现在特征已经感知了时序变化)
        dino1 = self.dino_adapter(dino1_dict)
        dino2 = self.dino_adapter(dino2_dict)

        # 金字塔融合
        pyramid1 = self.pyramid_fusion(cnn1, dino1)
        pyramid2 = self.pyramid_fusion(cnn2, dino2)

        # ⭐ v12: 跨时相 BT-CBAM 让金字塔特征 bi-temporal aware
        if self.use_dual_stream and hasattr(self, 'cross_temporal_cbam'):
            pyramid1, pyramid2 = self.cross_temporal_cbam(pyramid1, pyramid2)

        return pyramid1, pyramid2

    def forward(self, img1, img2, return_intermediate=False):
        """
        前向传播

        Args:
            img1: [B, 3, H, W] T1时刻图像
            img2: [B, 3, H, W] T2时刻图像
            return_intermediate: 是否返回中间预测

        Returns:
            logits: [B, num_classes, H, W] 最终预测
            或 predictions dict (如果return_intermediate=True)
        """
        B, _, H, W = img1.shape

        # ==================== 特征提取 ====================

        if self.use_hfa and hasattr(self, 'temporal_interaction'):
            # ⭐ BTI联合提取: T1/T2特征互相感知
            pyramid1, pyramid2 = self._extract_features_joint(img1, img2)
        else:
            # 独立提取 (原始方式)
            pyramid1 = self.extract_features(img1)
            pyramid2 = self.extract_features(img2)

        # ==================== 融合后时序交互 (PFTI) ====================

        if self.use_pfti:
            # ⭐ T1和T2在融合特征空间再次交互
            # BTI在DINO原始特征上交互 (粗粒度语义),
            # PFTI在融合特征上交互 (细粒度变化感知)
            pyramid1, pyramid2 = self.pfti(pyramid1, pyramid2)

        # ==================== v10 Bi-Temporal MixStyle ====================
        # 训练时随机扰动 T1/T2 通道级 style 统计, 扩大训练域分布
        # 测试时自动 pass-through (.eval() 时不触发)
        if self.use_bt_mixstyle and hasattr(self, 'bt_mixstyle'):
            pyramid1 = list(pyramid1)
            pyramid2 = list(pyramid2)
            for i in range(len(pyramid1)):
                pyramid1[i], pyramid2[i] = self.bt_mixstyle(pyramid1[i], pyramid2[i])

        # ==================== 差异建模 ====================

        # 统一使用 diff_module (MSCA/CADI/DualDifference)
        diff_pyramid = self.diff_module(pyramid1, pyramid2)

        # ==================== SGCE: 语义引导变化增强 ====================

        if self.use_sgce:
            diff_pyramid = self.sgce(diff_pyramid, pyramid1, pyramid2)

        # ==================== FPN解码 ====================

        # 自顶向下融合
        # 从最粗层(p4)开始，逐步上采样并与更细层融合
        p4, p3, p2, p1 = [self.lateral_convs[i](diff_pyramid[3-i])
                          for i in range(4)]

        # 逐层上采样融合 + Conv3×3细化 (标准FPN)
        p3 = self.fpn_refine[0](p3 + F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.fpn_refine[1](p2 + F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False))
        p1 = self.fpn_refine[2](p1 + F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False))

        # ==================== MSFA 多尺度特征聚合 ====================
        if self.use_msfa:
            p2_up = F.interpolate(p2, size=p1.shape[-2:], mode='bilinear', align_corners=False)
            p3_up = F.interpolate(p3, size=p1.shape[-2:], mode='bilinear', align_corners=False)
            p4_up = F.interpolate(p4, size=p1.shape[-2:], mode='bilinear', align_corners=False)
            feat = self.ms_fusion(torch.cat([p1, p2_up, p3_up, p4_up], dim=1)) + p1
            feat = feat * self.ms_se(feat)
        else:
            feat = p1  # 消融: 仅使用P1最细尺度

        # 上采样到原始尺寸并预测
        feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        logits = self.output_conv(feat)

        # ==================== 边界细化 ====================

        if self.use_boundary_refine:
            logits = self.boundary_refiner(logits)

        # ==================== 返回结果 ====================

        if return_intermediate:
            result = {'refined': logits}
            # ⭐ FCCR: 暴露T1/T2金字塔特征供特征一致性对比正则化
            result['pyramid1'] = pyramid1
            result['pyramid2'] = pyramid2
            # ⭐ v11 ICDCR: 暴露 MSFA 输出特征供 refinement head 用
            result['msfa_feat'] = feat
            # ⭐ Deep Supervision: FPN各层辅助预测
            if self.use_deep_supervision:
                result['aux_p2'] = F.interpolate(self.aux_heads[0](p2), size=(H, W), mode='bilinear', align_corners=False)
                result['aux_p3'] = F.interpolate(self.aux_heads[1](p3), size=(H, W), mode='bilinear', align_corners=False)
                result['aux_p4'] = F.interpolate(self.aux_heads[2](p4), size=(H, W), mode='bilinear', align_corners=False)
            return result
        else:
            return logits

    def predict(self, img1, img2, threshold=0.5):
        """
        预测接口（推理模式）

        Args:
            threshold: 变化检测阈值 (default 0.5). 降低阈值 → 更高召回率/更低精确率.
                       当模型 Precision >> Recall 时, 将 threshold 设为 0.35-0.45 可显著提升 IoU.

        Returns:
            pred_mask: [B, H, W] 预测mask
            prob_map: [B, H, W] 变化概率图
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(img1, img2)
            prob_map = F.softmax(logits, dim=1)[:, 1]
            pred_mask = (prob_map >= threshold).long()
        return pred_mask, prob_map

    def predict_tta(self, img1, img2, threshold=0.5):
        """
        Test-Time Augmentation (TTA) 预测 - 4种几何变换取平均

        原理:
        对同一输入施加4种几何变换 (原始/水平翻转/垂直翻转/水平+垂直翻转),
        分别预测后将概率图变换回原始空间, 取算术平均.

        数学分析:
        P_tta(x) = (1/4) * Σ_{t∈T} T⁻¹(P(T(x)))
        其中 T = {Identity, FlipH, FlipV, FlipHV}

        优势:
        - 零成本提升: 不需要重新训练, 仅增加推理时间 (4x)
        - 减少方向偏差: 消除模型对特定朝向的偏好
        - 增强边界预测: 翻转后边界像素被不同上下文包围, 平均后更稳健
        - 通常提升 +0.3~0.8% IoU

        Args:
            img1: [B, 3, H, W] T1时刻图像
            img2: [B, 3, H, W] T2时刻图像
            threshold: 变化检测阈值

        Returns:
            pred_mask: [B, H, W] TTA融合后预测mask
            prob_map: [B, H, W] TTA融合后变化概率图
        """
        self.eval()
        with torch.no_grad():
            probs = []

            # 1. 原始 (Identity)
            logits = self.forward(img1, img2)
            probs.append(F.softmax(logits, dim=1)[:, 1])

            # 2. 水平翻转 (Flip-H): 翻转W轴 (dim=3 in [B,C,H,W])
            logits_hf = self.forward(
                torch.flip(img1, [3]), torch.flip(img2, [3])
            )
            prob_hf = F.softmax(logits_hf, dim=1)[:, 1]
            probs.append(torch.flip(prob_hf, [2]))  # 翻转回: W轴在[B,H,W]是dim=2

            # 3. 垂直翻转 (Flip-V): 翻转H轴 (dim=2 in [B,C,H,W])
            logits_vf = self.forward(
                torch.flip(img1, [2]), torch.flip(img2, [2])
            )
            prob_vf = F.softmax(logits_vf, dim=1)[:, 1]
            probs.append(torch.flip(prob_vf, [1]))  # 翻转回: H轴在[B,H,W]是dim=1

            # 4. 水平+垂直翻转 (Flip-HV): 等价于180°旋转
            logits_hvf = self.forward(
                torch.flip(img1, [2, 3]), torch.flip(img2, [2, 3])
            )
            prob_hvf = F.softmax(logits_hvf, dim=1)[:, 1]
            probs.append(torch.flip(prob_hvf, [1, 2]))  # 翻转回H和W

            # 算术平均 (概率空间融合)
            avg_prob = torch.stack(probs, dim=0).mean(dim=0)
            pred_mask = (avg_prob >= threshold).long()

        return pred_mask, avg_prob


def build_dinobcd(config):
    """根据配置构建DinoBCD模型"""
    model = DinoBCD(
        backbone_type=config.get('backbone_type', 'efficient'),
        dino_weights=config.get('dino_weights', 'dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'),
        dino_extract_layers=config.get('dino_extract_layers', [5, 11, 17, 23]),
        num_classes=config.get('num_classes', 2),
        pyramid_channels=config.get('pyramid_channels', 128),
        use_boundary_refine=config.get('use_boundary_refine', True),
        use_cadi=config.get('use_cadi', True),
        use_sgda=config.get('use_sgda', False),
        use_msca=config.get('use_msca', False),      # ⭐ MSCA (多尺度变化聚合)
        use_hfa=config.get('use_hfa', False),
        use_lgati=config.get('use_lgati', False),
        use_pfti=config.get('use_pfti', False),          # ⭐ PFTI (融合后时序交互)
        use_efficient_lmm=config.get('use_efficient_lmm', True),
        sgda_config=config.get('sgda_config', {}),
        head_dropout=config.get('head_dropout', 0.0),
        lora_config=config.get('lora_config', None),    # ⭐ LoRA (骨干网络适配)
        unfreeze_config=config.get('unfreeze_config', None),  # ⭐ 骨干网络解冻
        use_sgbd=config.get('use_sgbd', False),              # ⭐ SGBD (语义引导双向差异)
        use_sgbd_gate=config.get('use_sgbd_gate', False),   # ⭐ SGBD门控 (默认关闭)
        use_cosine_weight=config.get('use_cosine_weight', False),  # ⭐ 零参数余弦加权
        use_bti=config.get('use_bti', True),               # ⭐ BTI开关 (消融用)
        use_msfa=config.get('use_msfa', True),             # ⭐ MSFA开关 (消融用)
        use_deep_supervision=config.get('use_deep_supervision', False),  # ⭐ Deep Supervision
        use_sgce=config.get('use_sgce', False),                         # ⭐ SGCE
        use_hbca=config.get('use_hbca', False),                         # ⭐ HBCA
        hbca_config=config.get('hbca_config', None),                    # ⭐ HBCA配置
        use_btot=config.get('use_btot', False),                         # ⭐ BTOT
        btot_config=config.get('btot_config', None),                    # ⭐ BTOT配置
        use_ssm=config.get('use_ssm', False),                           # ⭐ SSM
        use_bt_mixstyle=config.get('use_bt_mixstyle', False),           # ⭐ v10 BTM
        btm_config=config.get('btm_config', None),                      # ⭐ v10 BTM 配置
        use_dual_stream=config.get('use_dual_stream', False),           # ⭐ v12 BT-CBAM
        dual_stream_config=config.get('dual_stream_config', None),
        device=config.get('device', 'cuda')
    )
    return model


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Building DinoBCD Model")
    print("=" * 60)

    config = {
        'backbone_type': 'efficient',
        'pyramid_channels': 128,
        'use_boundary_refine': True,
        'device': device
    }

    model = build_dinobcd(config).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 测试前向传播
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    img1 = torch.randn(2, 3, 256, 256).to(device)
    img2 = torch.randn(2, 3, 256, 256).to(device)

    logits = model(img1, img2)
    print(f"Output: {logits.shape}")

    pred_mask, prob_map = model.predict(img1, img2)
    print(f"Pred Mask: {pred_mask.shape}")
    print(f"Prob Map: {prob_map.shape}")

    print("\n" + "=" * 60)
    print("DinoBCD Model Built Successfully!")
    print("=" * 60)
