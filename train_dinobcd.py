"""
DINOBCD Training Script
完全独立的训练脚本

使用方法:
    python train_dinobcd.py --config dinobcd/configs/default_config.yaml
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            print("[TensorBoard] tensorboard is not installed; scalar logging is disabled.")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None
from torch.amp import autocast, GradScaler  # 混合精度训练 (PyTorch 2.0+)
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn  # ⭐ SWA
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinobcd.models.dinobcd import build_dinobcd
from dinobcd.losses.combined_loss import DinoBCDLoss
from dinobcd.losses.fccr_loss import FCCRLoss
from dinobcd.datasets import build_dataloaders
from dinobcd.utils.class_balance import compute_adaptive_alpha
from dinobcd.utils.ema import ModelEMA


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """DINOBCD训练器"""

    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()

        # 设置随机种子
        self._set_seed(config['experiment']['seed'])

        # 创建模型
        print("=" * 60)
        print("Building DINOBCD Model...")
        print("=" * 60)

        model_config = {
            'backbone_type': config['model']['backbone_type'],
            'dino_weights': config['model']['dino_weights'],
            'dino_extract_layers': config['model']['dino_extract_layers'],
            'num_classes': config['model']['num_classes'],
            'pyramid_channels': config['model']['pyramid_channels'],
            'use_boundary_refine': config['model'].get('use_boundary_refine', True),
            'use_cadi': config['model'].get('use_cadi', True),
            'use_sgda': config['model'].get('use_sgda', False),
            'use_msca': config['model'].get('use_msca', False),     # ⭐ MSCA
            'use_hfa': config['model'].get('use_hfa', False),
            'use_lgati': config['model'].get('use_lgati', False),
            'use_pfti': config['model'].get('use_pfti', False),     # ⭐ PFTI
            'use_efficient_lmm': config['model'].get('use_efficient_lmm', True),
            'sgda_config': config['model'].get('sgda_config', {}),
            'head_dropout': config['model'].get('head_dropout', 0.0),  # ⭐ head dropout
            'lora_config': config['model'].get('lora', None),           # ⭐ LoRA
            'unfreeze_config': config['model'].get('backbone_unfreeze', None),  # ⭐ 骨干网络解冻
            'use_sgbd': config['model'].get('use_sgbd', False),        # ⭐ SGBD (语义引导双向差异)
            'use_sgbd_gate': config['model'].get('use_sgbd_gate', False),  # ⭐ SGBD门控
            'use_cosine_weight': config['model'].get('use_cosine_weight', False),  # ⭐ 零参数余弦加权
            'use_bti': config['model'].get('use_bti', True),                  # ⭐ BTI开关 (消融用)
            'use_msfa': config['model'].get('use_msfa', True),                # ⭐ MSFA开关 (消融用)
            'use_deep_supervision': config['model'].get('use_deep_supervision', False),  # ⭐ Deep Supervision
            'use_sgce': config['model'].get('use_sgce', False),  # ⭐ SGCE
            'use_hbca': config['model'].get('use_hbca', False),  # ⭐ HBCA
            'hbca_config': config['model'].get('hbca_config', None),  # ⭐ HBCA配置
            'use_btot': config['model'].get('use_btot', False),  # ⭐ BTOT
            'btot_config': config['model'].get('btot_config', None),  # ⭐ BTOT配置
            'use_ssm': config['model'].get('use_ssm', False),  # ⭐ SSM
            'device': self.device
        }

        self.model = build_dinobcd(model_config).to(self.device)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

        # ⭐ 检测LoRA参数 (用于分组学习率)
        self.has_lora = self.model.dino_extractor.use_lora
        if self.has_lora:
            lora_params = self.model.dino_extractor.get_lora_params()
            lora_count = sum(p.numel() for p in lora_params)
            print(f"  - LoRA parameters: {lora_count:,} ({lora_count/1e3:.1f}K)")
            print(f"  - Non-LoRA trainable: {trainable_params - lora_count:,}")

        # ⭐ 检测骨干网络解冻 (用于分组学习率)
        self.has_unfreeze = self.model.dino_extractor.use_unfreeze
        if self.has_unfreeze:
            backbone_params = self.model.dino_extractor.get_backbone_params()
            backbone_count = sum(p.numel() for p in backbone_params)
            print(f"  - Unfrozen backbone params: {backbone_count:,} ({backbone_count/1e6:.2f}M)")
            print(f"  - Head/decoder trainable: {trainable_params - backbone_count:,}")

        # 创建数据加载器
        print("\nLoading Dataset...")
        self.train_loader, self.val_loader = self._build_dataloaders(config['data'])

        # 获取focal_alpha，支持自适应计算
        focal_alpha = config['training']['loss'].get('focal_alpha', 0.75)
        
        # 如果启用自适应类别平衡 (DACB)
        if config['training']['loss'].get('auto_alpha', False):
            print("\n[DACB] Computing adaptive alpha...")
            focal_alpha = compute_adaptive_alpha(
                self.train_loader,
                num_samples=1000,
                method='inverse_freq'
            )
            print(f"[DACB] Using adaptive focal_alpha = {focal_alpha:.4f}")
        
        # 创建损失函数 - 支持APRB (Adaptive Precision-Recall Balance) + Tversky
        self.criterion = DinoBCDLoss(
            focal_alpha=focal_alpha,
            focal_gamma=config['training']['loss'].get('focal_gamma', 2.0),
            focal_weight=config['training']['loss'].get('focal_weight', 1.0),
            dice_weight=config['training']['loss'].get('dice_weight', 1.0),
            use_edge_loss=config['training']['loss'].get('use_edge_loss', True),
            edge_loss_weight=config['training']['loss'].get('edge_loss_weight', 0.3),
            # APRB参数
            use_adaptive_dice=config['training']['loss'].get('use_adaptive_dice', True),
            use_pr_monitor=config['training']['loss'].get('use_pr_monitor', True),
            pr_window_size=config['training']['loss'].get('pr_window_size', 100),
            # Tversky Loss参数 (召回率优化)
            use_tversky=config['training']['loss'].get('use_tversky', False),
            tversky_alpha=config['training']['loss'].get('tversky_alpha', 0.3),
            tversky_beta=config['training']['loss'].get('tversky_beta', 0.7),
            tversky_weight=config['training']['loss'].get('tversky_weight', 1.0),
            label_smoothing=config['training']['loss'].get('label_smoothing', 0.0),
            ohem_ratio=config['training']['loss'].get('ohem_ratio', 0.0)
        ).to(self.device)

        print(f"\n[APRB] Adaptive Precision-Recall Balance enabled:")
        print(f"  - Adaptive Dice Loss: {config['training']['loss'].get('use_adaptive_dice', True)}")
        print(f"  - Online P-R Monitor: {config['training']['loss'].get('use_pr_monitor', True)}")

        # ⭐ FCCR: 特征一致性对比正则化损失
        fccr_config = config['training']['loss'].get('fccr', {})
        self.use_fccr = fccr_config.get('enable', False)
        if self.use_fccr:
            self.fccr_loss = FCCRLoss(
                pull_weight=fccr_config.get('pull_weight', 1.0),
                push_weight=fccr_config.get('push_weight', 1.0),
                push_margin=fccr_config.get('push_margin', -0.3),
                pull_threshold=fccr_config.get('pull_threshold', 0.5),
                warmup_epochs=fccr_config.get('warmup_epochs', 5),
            ).to(self.device)
            self.fccr_weight = fccr_config.get('weight', 0.1)
            print(f"\n[FCCR] Feature Consistency Contrastive Regularization enabled:")
            print(f"  - Weight: {self.fccr_weight}")
            print(f"  - Pull weight: {fccr_config.get('pull_weight', 1.0)}")
            print(f"  - Push weight: {fccr_config.get('push_weight', 1.0)}")
            print(f"  - Push margin: {fccr_config.get('push_margin', -0.3)}")
            print(f"  - Pull threshold: {fccr_config.get('pull_threshold', 0.5)}")
            print(f"  - Warmup epochs: {fccr_config.get('warmup_epochs', 5)}")

        # 创建优化器
        self.optimizer = self._build_optimizer(config['training']['optimizer'])

        # 创建学习率调度器
        self.scheduler = self._build_scheduler(config['training']['scheduler'])

        # 混合精度训练 (EAAI效率优化)
        self.use_amp = config['device'].get('use_amp', True) and torch.cuda.is_available()
        # AMP dtype: bf16 不会像 fp16 那样溢出 NaN, 且无需 GradScaler (Blackwell/Ampere+)
        _amp_dtype = str(config['device'].get('amp_dtype', 'fp16')).lower()
        self.amp_dtype = torch.bfloat16 if _amp_dtype in ('bf16', 'bfloat16') else torch.float16
        # bf16 动态范围等同 FP32, 不需要 loss scaling; 仅 fp16 使用 GradScaler
        self.scaler = GradScaler(device='cuda') if (self.use_amp and self.amp_dtype == torch.float16) else None
        if self.use_amp:
            print(f"\n[AMP] Mixed Precision Training Enabled (dtype={self.amp_dtype})")
            print(f"  - Expected speedup: 30-50%")
            print(f"  - Memory saving: ~40%")
            print(f"  - GradScaler: {'on (fp16)' if self.scaler is not None else 'off (bf16, not needed)'}")
        else:
            if not torch.cuda.is_available():
                print(f"\n[AMP] Disabled (CPU mode)")

        # ⭐ 梯度累积 (Gradient Accumulation)
        # effective_batch_size = batch_size × accumulation_steps
        # 更大的有效batch → 更稳定的梯度, 尤其对解冻backbone微调至关重要
        self.accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        if self.accumulation_steps > 1:
            effective_batch = config['data']['batch_size'] * self.accumulation_steps
            print(f"\n[GradAccum] Gradient Accumulation: {self.accumulation_steps} steps")
            print(f"  - Physical batch: {config['data']['batch_size']}")
            print(f"  - Effective batch: {effective_batch}")

        # 推理阈值 (低于0.5可提升召回率，适用于Precision >> Recall的情况)
        self.inference_threshold = config['model'].get('inference_threshold', 0.5)
        print(f"\n[Threshold] Inference threshold: {self.inference_threshold}")
        if self.inference_threshold < 0.5:
            print(f"  ↑ 低阈值模式: 预期 Recall↑ Precision↓")

        # ⭐ EMA (Exponential Moving Average) - 训练稳定性 + 泛化提升
        ema_config = config['training'].get('ema', {})
        self.use_ema = ema_config.get('enable', False)
        if self.use_ema:
            ema_decay = ema_config.get('decay', 0.9998)
            ema_warmup = ema_config.get('warmup_steps', 2000)
            self.ema = ModelEMA(self.model, decay=ema_decay, warmup_steps=ema_warmup)
            print(f"\n[EMA] Exponential Moving Average enabled:")
            print(f"  - Decay: {ema_decay}")
            print(f"  - Warmup steps: {ema_warmup}")
            print(f"  - Expected improvement: +0.5~1.0% IoU")
        else:
            self.ema = None

        # ⭐ SWA (Stochastic Weight Averaging) - 宽损失盆地 → 跨区域泛化
        swa_config = config['training'].get('swa', {})
        self.use_swa = swa_config.get('enable', False)
        if self.use_swa:
            self.swa_start = swa_config.get('start_epoch', 100)
            self.swa_lr = swa_config.get('lr', 1e-5)
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)
            self.swa_n = 0  # SWA平均次数计数
            print(f"\n[SWA] Stochastic Weight Averaging enabled:")
            print(f"  - Start epoch: {self.swa_start}")
            print(f"  - SWA LR: {self.swa_lr:.2e}")
            print(f"  - Wider minima → better cross-region generalization")
        else:
            self.swa_model = None

        # ⭐ 2-Phase LR Schedule (借鉴ChangeDINO)
        two_phase_config = config['training'].get('two_phase', {})
        self.use_two_phase = two_phase_config.get('enable', False)
        if self.use_two_phase:
            total_epochs = config['training']['num_epochs']
            self.phase2_start = int(total_epochs * (1 - two_phase_config.get('phase2_ratio', 0.1)))
            self.phase2_lr_scale = two_phase_config.get('phase2_lr_scale', 0.2)
            print(f"\n[2-Phase] 2-Phase LR Schedule enabled:")
            print(f"  - Phase1: epoch 1-{self.phase2_start} (normal cosine)")
            print(f"  - Phase2: epoch {self.phase2_start+1}-{total_epochs} (LR * {self.phase2_lr_scale})")

        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self._swa_testing = False

        # 日志
        self._setup_logging(config['logging'])

    def _setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            device = f"cuda:{self.config['device']['gpu_ids'][0]}"
            print(f"Using GPU: {device}")
        else:
            device = "cpu"
            print("Using CPU")
        return device

    @torch.no_grad()
    def _update_swa_bn(self):
        """更新SWA模型的BatchNorm统计量 (兼容dict-based dataloader)"""
        momenta = {}
        for module in self.swa_model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum

        if not momenta:
            print("[SWA] No BatchNorm layers found, skipping BN update")
            return

        was_training = self.swa_model.training
        self.swa_model.train()
        for module in momenta:
            module.momentum = None
            module.num_batches_tracked *= 0

        for i, batch in enumerate(self.train_loader):
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            self.swa_model(img1, img2)
            if (i + 1) % 100 == 0:
                print(f"  [SWA BN] {i+1}/{len(self.train_loader)} batches processed")

        for module in momenta:
            module.momentum = momenta[module]
        self.swa_model.train(was_training)
        print(f"[SWA] BatchNorm updated with {len(self.train_loader)} batches")

    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

        if self.config['device']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_optimizer(self, opt_config):
        """构建优化器 (支持LoRA/backbone分组学习率)"""
        base_lr = opt_config['lr']

        if self.has_lora:
            # ⭐ LoRA分组: LoRA参数用较低学习率 (骨干网络需要温和的适配)
            lora_lr = opt_config.get('lora_lr', base_lr * 0.5)
            lora_param_ids = set(id(p) for p in self.model.dino_extractor.get_lora_params())

            lora_params = []
            other_params = []
            for p in self.model.parameters():
                if not p.requires_grad:
                    continue
                if id(p) in lora_param_ids:
                    lora_params.append(p)
                else:
                    other_params.append(p)

            param_groups = [
                {'params': other_params, 'lr': base_lr},
                {'params': lora_params, 'lr': lora_lr, 'weight_decay': 0.0}  # LoRA通常不加weight_decay
            ]
            print(f"\n[Optimizer] Parameter groups:")
            print(f"  - Main params: lr={base_lr:.2e}, {len(other_params)} tensors")
            print(f"  - LoRA params: lr={lora_lr:.2e}, {len(lora_params)} tensors (no weight decay)")
        elif self.has_unfreeze:
            # ⭐ 骨干网络解冻分组: 解冻的backbone用极低学习率
            # 关键: backbone_lr << main_lr, 确保权重变化缓慢, 下游模块能逐步适配
            backbone_lr = opt_config.get('backbone_lr', base_lr * 0.01)
            backbone_param_ids = set(id(p) for p in self.model.dino_extractor.get_backbone_params())

            backbone_params = []
            other_params = []
            for p in self.model.parameters():
                if not p.requires_grad:
                    continue
                if id(p) in backbone_param_ids:
                    backbone_params.append(p)
                else:
                    other_params.append(p)

            param_groups = [
                {'params': other_params, 'lr': base_lr},
                {'params': backbone_params, 'lr': backbone_lr}
            ]
            print(f"\n[Optimizer] Parameter groups (backbone unfreezing):")
            print(f"  - Head/decoder params: lr={base_lr:.2e}, {len(other_params)} tensors")
            print(f"  - Backbone params: lr={backbone_lr:.2e}, {len(backbone_params)} tensors")
            print(f"  - LR ratio: backbone/main = {backbone_lr/base_lr:.4f}")
        else:
            param_groups = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_config['type'] == 'AdamW':
            optimizer = optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=opt_config['weight_decay'],
                betas=opt_config['betas']
            )
        elif opt_config['type'] == 'SGD':
            optimizer = optim.SGD(
                param_groups,
                lr=base_lr,
                momentum=0.9,
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")

        return optimizer

    def _build_scheduler(self, sched_config):
        """构建学习率调度器 (支持warmup)"""
        warmup_epochs = sched_config.get('warmup_epochs', 0)

        if sched_config['type'] == 'CosineAnnealingLR':
            # 主调度器: CosineAnnealing (从warmup结束后开始)
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(sched_config['T_max'] - warmup_epochs, 1),
                eta_min=sched_config['eta_min']
            )
        elif sched_config['type'] == 'StepLR':
            main_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_config['type']}")

        # ⭐ 添加线性warmup (如果配置了warmup_epochs > 0)
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,   # 初始LR = 0.01 × base_lr
                end_factor=1.0,      # warmup结束时 LR = base_lr
                total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
            print(f"\n[Scheduler] CosineAnnealing with {warmup_epochs}-epoch linear warmup")
            print(f"  - Warmup: LR {0.01 * self.optimizer.param_groups[0]['lr']:.2e} → {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  - Then cosine decay to {sched_config['eta_min']:.2e}")
        else:
            scheduler = main_scheduler

        return scheduler

    def _build_dataloaders(self, data_config):
        """
        构建数据加载器 - 使用注册表支持多种数据集
        """
        train_loader, val_loader = build_dataloaders(
            data_config=data_config,
            batch_size=data_config.get('batch_size', 8),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )
        
        print(f"Train samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
        print(f"Val samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")
        
        return train_loader, val_loader

    def _setup_logging(self, log_config):
        """设置日志"""
        # 使用数据集名称创建子目录
        dataset_name = self.config['data']['dataset'].lower()
        
        self.log_dir = os.path.join(log_config['log_dir'], dataset_name)
        self.save_dir = os.path.join(log_config['save_dir'], dataset_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        if log_config['tensorboard']:
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

        self.print_freq = log_config['print_freq']
        
        print(f"Log directory: {self.log_dir}")
        print(f"Checkpoint directory: {self.save_dir}")

    def train_epoch(self, epoch):
        """训练一个epoch (支持梯度累积)"""
        self.model.train()

        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']

        # 训练指标
        loss_meter = AverageMeter()

        # 累积混淆矩阵用于计算训练集指标
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        # 如果没有数据加载器，返回模拟损失
        if self.train_loader is None:
            print("WARNING: No train_loader available, skipping training.")
            return {'total_loss': 0.0, 'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        # ⭐ 梯度累积: 在accumulation_steps步内累积梯度, 然后统一更新
        accum_steps = self.accumulation_steps
        self.optimizer.zero_grad()  # 在epoch开始时清零梯度

        # 训练循环 (简化日志，不显示进度条)
        for i, batch in enumerate(self.train_loader):
            # 解析batch
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            label = batch['label'].to(self.device)

            # 前向传播 - 混合精度优化
            if self.use_amp:
                with autocast(device_type='cuda', dtype=self.amp_dtype):
                    predictions = self.model(img1, img2, return_intermediate=True)
                    loss, loss_dict = self.criterion(predictions, label, current_epoch=epoch)
                    # ⭐ FCCR: 特征一致性对比正则化
                    if self.use_fccr and isinstance(predictions, dict) and 'pyramid1' in predictions:
                        fccr_loss, fccr_dict = self.fccr_loss(
                            predictions['pyramid1'], predictions['pyramid2'],
                            label, current_epoch=epoch
                        )
                        loss = loss + self.fccr_weight * fccr_loss
                        loss_dict.update(fccr_dict)
                    loss = loss / accum_steps  # 梯度累积: 损失除以累积步数

                # 检查loss是否为NaN或Inf
                if not torch.isfinite(loss):
                    print(f"WARNING: Loss is {loss.item()} at batch {i}, skipping this batch")
                    continue

                # 反向传播 (AMP) - 累积梯度
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()  # bf16: 无需 loss scaling

                # 每accum_steps步更新一次参数
                if (i + 1) % accum_steps == 0 or (i + 1) == len(self.train_loader):
                    if self.scaler is not None:
                        # 梯度裁剪 (需先 unscale)
                        if self.config['training']['grad_clip']['enable']:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['grad_clip']['max_norm']
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # bf16: 直接裁剪并更新
                        if self.config['training']['grad_clip']['enable']:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['grad_clip']['max_norm']
                            )
                        self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                # 标准训练
                predictions = self.model(img1, img2, return_intermediate=True)
                loss, loss_dict = self.criterion(predictions, label, current_epoch=epoch)
                # ⭐ FCCR: 特征一致性对比正则化
                if self.use_fccr and isinstance(predictions, dict) and 'pyramid1' in predictions:
                    fccr_loss, fccr_dict = self.fccr_loss(
                        predictions['pyramid1'], predictions['pyramid2'],
                        label, current_epoch=epoch
                    )
                    loss = loss + self.fccr_weight * fccr_loss
                    loss_dict.update(fccr_dict)
                loss_for_backward = loss / accum_steps  # 梯度累积: 损失除以累积步数

                # 反向传播 - 累积梯度
                loss_for_backward.backward()

                # 每accum_steps步更新一次参数
                if (i + 1) % accum_steps == 0 or (i + 1) == len(self.train_loader):
                    # 梯度裁剪
                    if self.config['training']['grad_clip']['enable']:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training']['grad_clip']['max_norm']
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # ⭐ EMA更新 (每个训练step后)
            if self.ema is not None:
                self.ema.update(self.model)

            # 更新指标 (记录未缩放的loss)
            loss_meter.update(loss.item() * accum_steps if self.use_amp else loss.item(), img1.size(0))

            # 计算训练集混淆矩阵 (直接从predictions计算，不调用predict避免改变模型状态)
            with torch.no_grad():
                # predictions是logits [B, 2, H, W]，需要softmax + argmax
                if isinstance(predictions, dict):
                    logits = predictions['refined']  # 使用refined输出
                else:
                    logits = predictions

                # 使用softmax获取概率，然后argmax得到类别预测
                probs = torch.softmax(logits, dim=1)  # [B, 2, H, W]
                pred_mask = torch.argmax(probs, dim=1)  # [B, H, W]，值为0或1

                tp = ((pred_mask == 1) & (label == 1)).sum().item()
                fp = ((pred_mask == 1) & (label == 0)).sum().item()
                fn = ((pred_mask == 0) & (label == 1)).sum().item()
                tn = ((pred_mask == 0) & (label == 0)).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

            # TensorBoard记录
            if self.writer is not None:
                global_step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                for key, val in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', val, global_step)

        # 计算训练集全局指标
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)

        return {
            'total_loss': loss_meter.avg,
            'iou': iou,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    @torch.no_grad()
    def validate(self, epoch):
        """验证 (使用EMA模型参数, 如果启用)"""
        # ⭐ EMA: 验证时使用EMA参数 (更稳定的参数估计)
        if self.ema is not None:
            self.ema.apply(self.model)

        self.model.eval()

        # 如果没有验证集
        if self.val_loader is None:
            if self.ema is not None:
                self.ema.restore(self.model)
            print("WARNING: No val_loader available, skipping validation.")
            return {'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'oa': 0.0}

        # 累积预测结果用于计算全局指标
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        # 验证循环 (简化日志，不显示进度条)
        for batch in self.val_loader:
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            label = batch['label'].to(self.device)

            # 前向传播
            pred_mask, prob_map = self.model.predict(img1, img2, threshold=self.inference_threshold)

            # 累积混淆矩阵
            tp = ((pred_mask == 1) & (label == 1)).sum().item()
            fp = ((pred_mask == 1) & (label == 0)).sum().item()
            fn = ((pred_mask == 0) & (label == 1)).sum().item()
            tn = ((pred_mask == 0) & (label == 0)).sum().item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        # 计算全局指标
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)  # 前景IoU
        oa = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-6)

        # ⭐ EMA: 验证完成后恢复原始模型参数
        if self.ema is not None:
            self.ema.restore(self.model)

        # TensorBoard记录
        if self.writer is not None:
            self.writer.add_scalar('Val/IoU', iou, epoch)
            self.writer.add_scalar('Val/F1', f1, epoch)
            self.writer.add_scalar('Val/Precision', precision, epoch)
            self.writer.add_scalar('Val/Recall', recall, epoch)
            self.writer.add_scalar('Val/OA', oa, epoch)

        return {
            'iou': iou,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'oa': oa
        }

    def _compute_iou(self, pred, target):
        """计算IoU"""
        intersection = ((pred == 1) & (target == 1)).sum().float()
        union = ((pred == 1) | (target == 1)).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.item()

    def _compute_f1(self, pred, target):
        """计算F1"""
        tp = ((pred == 1) & (target == 1)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        return f1.item()

    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint (含EMA状态)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }

        # ⭐ 保存EMA状态
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        # 保存最新checkpoint
        latest_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳checkpoint (使用EMA参数, 因为EMA参数是用于推理的)
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            if self.ema is not None:
                # ⭐ 保存EMA参数作为best模型 (EMA参数泛化更好)
                self.ema.apply(self.model)
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),  # EMA参数
                    'best_metric': self.best_metric,
                    'config': self.config,
                    'is_ema': True
                }
                torch.save(best_checkpoint, best_path)
                self.ema.restore(self.model)
            else:
                torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")

        # 定期保存
        if epoch % self.config['training']['save_interval'] == 0:
            epoch_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, ckpt_path):
        """从checkpoint恢复model / optimizer / scheduler / EMA / epoch / best_metric.
        用法: 训练脚本加 --resume latest 或 --resume <path>，会接着上次 epoch 继续。"""
        print(f"\n[Resume] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if self.ema is not None and 'ema_state_dict' in ckpt:
            self.ema.load_state_dict(ckpt['ema_state_dict'])
        self.current_epoch = int(ckpt.get('epoch', 0))
        self.best_metric = float(ckpt.get('best_metric', 0.0))
        print(f"[Resume] Resumed at epoch {self.current_epoch}, best_metric={self.best_metric:.4f}")

    def _search_optimal_threshold(self):
        """在验证集上搜索IoU最优的推理阈值

        原理: 固定阈值0.5不一定是IoU最优点.
        当模型P>>R时, 降低阈值可提升recall, 从而提升IoU.
        在val set上网格搜索最优阈值, 用于test评估.

        Returns:
            best_threshold (float): IoU最优阈值
            best_iou (float): 对应的val IoU
        """
        print("\n" + "=" * 60)
        print("Searching Optimal Inference Threshold on Validation Set")
        print("=" * 60)

        self.model.eval()

        # 第1步: 收集所有val样本的概率图和标签
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                label = batch['label']  # keep on CPU to save GPU memory

                logits = self.model(img1, img2)
                prob_map = torch.softmax(logits, dim=1)[:, 1].cpu()  # [B, H, W]

                all_probs.append(prob_map)
                all_labels.append(label)

        all_probs = torch.cat(all_probs, dim=0)    # [N, H, W]
        all_labels = torch.cat(all_labels, dim=0)   # [N, H, W]

        # 第2步: 网格搜索阈值
        thresholds = [t / 100.0 for t in range(25, 71)]  # 0.25 ~ 0.70
        best_iou = 0.0
        best_threshold = 0.5
        results = []

        for th in thresholds:
            pred = (all_probs >= th).long()
            tp = ((pred == 1) & (all_labels == 1)).sum().item()
            fp = ((pred == 1) & (all_labels == 0)).sum().item()
            fn = ((pred == 0) & (all_labels == 1)).sum().item()
            iou = tp / (tp + fp + fn + 1e-6)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            results.append((th, iou, precision, recall))

            if iou > best_iou:
                best_iou = iou
                best_threshold = th

        # 打印搜索结果
        print(f"\nThreshold search results (top 10):")
        results.sort(key=lambda x: x[1], reverse=True)
        for th, iou, p, r in results[:10]:
            marker = " ← BEST" if th == best_threshold else ""
            print(f"  th={th:.2f}: IoU={iou:.4f} P={p:.4f} R={r:.4f}{marker}")

        print(f"\n[Threshold] Optimal: {best_threshold:.2f} (Val IoU: {best_iou:.4f})")
        print(f"[Threshold] Default 0.50 vs Optimal {best_threshold:.2f}: "
              f"IoU {[r for r in results if abs(r[0]-0.5)<0.001][0][1]:.4f} → {best_iou:.4f}")
        print("=" * 60)

        return best_threshold, best_iou

    def _evaluate_on_loader(self, test_loader, use_tta=False):
        """在数据加载器上评估模型 (支持TTA)

        Args:
            test_loader: 测试数据加载器
            use_tta: 是否使用Test-Time Augmentation

        Returns:
            metrics dict: iou, f1, precision, recall, oa
        """
        self.model.eval()
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        with torch.no_grad():
            for batch in test_loader:
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                label = batch['label'].to(self.device)

                if use_tta:
                    pred_mask, prob_map = self.model.predict_tta(
                        img1, img2, threshold=self.inference_threshold
                    )
                else:
                    pred_mask, prob_map = self.model.predict(
                        img1, img2, threshold=self.inference_threshold
                    )

                tp = ((pred_mask == 1) & (label == 1)).sum().item()
                fp = ((pred_mask == 1) & (label == 0)).sum().item()
                fn = ((pred_mask == 0) & (label == 1)).sum().item()
                tn = ((pred_mask == 0) & (label == 0)).sum().item()

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        oa = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-6)

        return {
            'iou': iou, 'f1': f1,
            'precision': precision, 'recall': recall, 'oa': oa
        }

    def _save_test_visualizations(self, test_loader, vis_dir, max_vis=20):
        """保存测试集可视化结果 (聚焦错误样本)

        生成: T1图像 | T2图像 | GT | 预测 | 错误图(FP=红, FN=蓝)
        优先保存IoU最低的样本, 用于分析失败模式.
        """
        import numpy as np
        from PIL import Image

        os.makedirs(vis_dir, exist_ok=True)
        self.model.eval()

        # 收集所有样本的per-sample IoU
        sample_results = []
        with torch.no_grad():
            for batch in test_loader:
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                label = batch['label'].to(self.device)
                filenames = batch['filename']

                pred_mask, prob_map = self.model.predict(
                    img1, img2, threshold=self.inference_threshold
                )

                for i in range(img1.shape[0]):
                    pred_i = pred_mask[i]  # [H, W]
                    label_i = label[i]     # [H, W]
                    tp = ((pred_i == 1) & (label_i == 1)).sum().item()
                    fp = ((pred_i == 1) & (label_i == 0)).sum().item()
                    fn = ((pred_i == 0) & (label_i == 1)).sum().item()

                    # 无变化样本(GT全0)且模型也预测无变化: IoU=1.0 (正确)
                    if tp + fp + fn == 0:
                        iou_i = 1.0
                    else:
                        iou_i = tp / (tp + fp + fn + 1e-6)

                    sample_results.append({
                        'filename': filenames[i],
                        'iou': iou_i,
                        'tp': tp, 'fp': fp, 'fn': fn,
                        'has_change': (label_i == 1).any().item(),
                        'img1': img1[i].cpu(),
                        'img2': img2[i].cpu(),
                        'label': label_i.cpu(),
                        'pred': pred_i.cpu(),
                        'prob': prob_map[i].cpu() if prob_map is not None else None,
                    })

        # 只保留有变化的样本 (GT中存在change pixels), 按IoU升序排列
        change_samples = [r for r in sample_results if r['has_change']]
        nochange_samples = [r for r in sample_results if not r['has_change']]
        change_samples.sort(key=lambda x: x['iou'])

        # 统计无变化样本中的FP情况
        nochange_fp = sum(r['fp'] for r in nochange_samples)
        print(f"\n[Visualization] {len(change_samples)} change samples, "
              f"{len(nochange_samples)} no-change samples "
              f"(FP in no-change: {nochange_fp:,} pixels)")

        sample_results = change_samples

        # ImageNet denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        saved = 0
        for idx, res in enumerate(sample_results):
            if saved >= max_vis:
                break

            # Denormalize images
            img1_vis = (res['img1'] * std + mean).clamp(0, 1)
            img2_vis = (res['img2'] * std + mean).clamp(0, 1)
            img1_np = (img1_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img2_np = (img2_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # GT and Pred masks
            label_np = (res['label'].numpy() * 255).astype(np.uint8)
            pred_np = (res['pred'].numpy() * 255).astype(np.uint8)

            # Error map: FP=红(predicted change but no change), FN=蓝(missed change)
            H, W = label_np.shape
            error_map = np.zeros((H, W, 3), dtype=np.uint8)
            fp_mask = (res['pred'].numpy() == 1) & (res['label'].numpy() == 0)
            fn_mask = (res['pred'].numpy() == 0) & (res['label'].numpy() == 1)
            tp_mask = (res['pred'].numpy() == 1) & (res['label'].numpy() == 1)
            error_map[tp_mask] = [0, 255, 0]    # TP = 绿
            error_map[fp_mask] = [255, 0, 0]    # FP = 红
            error_map[fn_mask] = [0, 0, 255]    # FN = 蓝

            # 拼接: T1 | T2 | GT | Pred | Error
            gt_rgb = np.stack([label_np]*3, axis=-1)
            pred_rgb = np.stack([pred_np]*3, axis=-1)
            canvas = np.concatenate([img1_np, img2_np, gt_rgb, pred_rgb, error_map], axis=1)

            fname = os.path.splitext(res['filename'])[0]
            iou_str = f"{res['iou']:.3f}"
            save_path = os.path.join(vis_dir, f"rank{idx:03d}_iou{iou_str}_{fname}.png")
            Image.fromarray(canvas).save(save_path)
            saved += 1

        # 统计报告 (仅统计有变化的样本)
        if sample_results:
            ious = [r['iou'] for r in sample_results]
            total_fp = sum(r['fp'] for r in sample_results)
            total_fn = sum(r['fn'] for r in sample_results)
            print(f"[Visualization] Saved {saved} worst change-samples to {vis_dir}")
            print(f"  Worst IoU: {ious[0]:.4f} ({sample_results[0]['filename']})")
            print(f"  Median IoU: {ious[len(ious)//2]:.4f}")
            print(f"  Top-20 worst avg IoU: {sum(ious[:min(20,len(ious))])/min(20,len(ious)):.4f}")
            print(f"  Total FP pixels: {total_fp:,}, Total FN pixels: {total_fn:,}")
            print(f"  FP/FN ratio: {total_fp/(total_fn+1e-6):.2f}")
            print(f"  Color code: Green=TP, Red=FP, Blue=FN")

    def test(self):
        """测试集评估（训练完成后调用, 包含可视化）"""
        is_swa = getattr(self, '_swa_testing', False)

        print("\n" + "=" * 60)
        if is_swa:
            print("Testing on Test Set (SWA-averaged model)")
        else:
            print("Testing on Test Set")
        print("=" * 60)

        if not is_swa:
            # 加载最佳模型 (SWA模式跳过, 因为已经加载了SWA权重)
            best_model_path = os.path.join(self.save_dir, 'best.pth')
            if not os.path.exists(best_model_path):
                print(f"Best model not found at {best_model_path}, skipping test.")
                return None

            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            is_ema = checkpoint.get('is_ema', False)
            print(f"Loaded best model from epoch {checkpoint['epoch']}" +
                  (" (EMA parameters)" if is_ema else ""))

        # 构建测试数据加载器
        data_config = self.config['data'].copy()
        data_config['list_file'] = data_config.get('test_list', 'test.txt')

        from dinobcd.datasets import ChangeDetectionDataset
        test_dataset = ChangeDetectionDataset(
            dataset_name=data_config['dataset'],
            data_root=data_config['data_root'],
            list_file=data_config['list_file'],
            image_size=data_config['image_size'],
            is_train=False,
            augmentation=None
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config.get('batch_size', 8),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True)
        )

        print(f"Test samples: {len(test_dataset)}, batches: {len(test_loader)}")

        # ==================== 验证集阈值搜索 ====================
        try:
            optimal_th, val_iou = self._search_optimal_threshold()
        except RuntimeError as e:
            print(f"[Threshold] Optimal-threshold search skipped ({e}); using fixed 0.50.")
            optimal_th, val_iou = self.inference_threshold, 0.0
            torch.cuda.empty_cache()

        # ==================== TTA配置 ====================
        use_tta = self.config.get('evaluation', {}).get('use_tta', False)
        if use_tta:
            print("[TTA] Test-Time Augmentation enabled (4x geometric transforms)")

        # ==================== 标准评估 (threshold=0.5) ====================
        test_metrics = self._evaluate_on_loader(test_loader, use_tta=use_tta)

        tta_tag = " + TTA" if use_tta else ""
        print("\n" + "=" * 60)
        print(f"Test Set Results (threshold=0.50{tta_tag}):")
        print(f"  IoU:       {test_metrics['iou']:.4f}")
        print(f"  F1:        {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  OA:        {test_metrics['oa']:.4f}")
        print("=" * 60)

        # ==================== 最优阈值评估 ====================
        if abs(optimal_th - self.inference_threshold) > 0.01:
            old_th = self.inference_threshold
            self.inference_threshold = optimal_th
            test_metrics_opt = self._evaluate_on_loader(test_loader, use_tta=use_tta)
            self.inference_threshold = old_th  # restore

            print("\n" + "=" * 60)
            print(f"Test Set Results (optimal threshold={optimal_th:.2f}):")
            print(f"  IoU:       {test_metrics_opt['iou']:.4f}")
            print(f"  F1:        {test_metrics_opt['f1']:.4f}")
            print(f"  Precision: {test_metrics_opt['precision']:.4f}")
            print(f"  Recall:    {test_metrics_opt['recall']:.4f}")
            print(f"  OA:        {test_metrics_opt['oa']:.4f}")
            iou_delta = test_metrics_opt['iou'] - test_metrics['iou']
            print(f"  IoU improvement: {iou_delta:+.4f} (vs threshold=0.50)")
            print("=" * 60)

            # 使用最优阈值进行可视化和返回
            self.inference_threshold = optimal_th
            test_metrics = test_metrics_opt  # 返回最优阈值的结果

        # ==================== 可视化 ====================
        eval_config = self.config.get('evaluation', {})
        if eval_config.get('visualize', False):
            vis_num = eval_config.get('vis_num', 20)
            vis_dir = os.path.join(self.log_dir, 'test_vis')
            self._save_test_visualizations(test_loader, vis_dir, max_vis=vis_num)

        return test_metrics

    def train(self):
        """完整训练流程 (支持早停)"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print(f"Dataset: {self.config['data']['dataset']}")
        print("=" * 60)

        num_epochs = self.config['training']['num_epochs']

        # ⭐ 早停机制
        es_config = self.config['training'].get('early_stopping', {})
        use_early_stopping = es_config.get('enable', False)
        patience = es_config.get('patience', 25)
        epochs_no_improve = 0

        if use_early_stopping:
            print(f"[EarlyStopping] Enabled: patience={patience}, metric=IoU")

        for epoch in range(self.current_epoch, num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)

            # 验证
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_metrics = self.validate(epoch)

                # 保存最佳模型 (基于IoU)
                is_best = val_metrics['iou'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['iou']
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # 格式化输出
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                best_mark = " *" if is_best else ""
                log_msg = (f"{timestamp} - INFO - Epoch {epoch+1:3d}/{num_epochs} | "
                          f"Train: Loss={train_metrics['total_loss']:.4f} "
                          f"IoU={train_metrics['iou']:.4f} "
                          f"F1={train_metrics['f1']:.4f} "
                          f"Pre={train_metrics['precision']:.4f} "
                          f"Rec={train_metrics['recall']:.4f} | "
                          f"Val: IoU={val_metrics['iou']:.4f} "
                          f"F1={val_metrics['f1']:.4f} "
                          f"Pre={val_metrics['precision']:.4f} "
                          f"Rec={val_metrics['recall']:.4f}"
                          f"{best_mark}")
                print(log_msg)

                self.save_checkpoint(epoch + 1, is_best)

                # ⭐ 早停检查
                if use_early_stopping and epochs_no_improve >= patience:
                    print(f"\n[EarlyStopping] No improvement for {patience} epochs, stopping.")
                    print(f"[EarlyStopping] Best IoU: {self.best_metric:.4f}")
                    break
            else:
                # 非验证轮次只打印训练指标
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
                log_msg = (f"{timestamp} - INFO - Epoch {epoch+1:3d}/{num_epochs} | "
                          f"Train: Loss={train_metrics['total_loss']:.4f} "
                          f"IoU={train_metrics['iou']:.4f} "
                          f"F1={train_metrics['f1']:.4f} "
                          f"Pre={train_metrics['precision']:.4f} "
                          f"Rec={train_metrics['recall']:.4f}")
                print(log_msg)

            # 更新学习率
            if self.use_swa and (epoch + 1) >= self.swa_start:
                # SWA阶段: 使用固定LR + 累积权重平均
                if self.swa_n == 0:
                    # ⭐ 重建 swa_model/swa_scheduler 以匹配"最终的" model 与 optimizer.
                    # 子类 (v25/v36/v37/v38) 会在 base __init__ 之后才添加模块 (如 BTOT decoder)
                    # 并重建 optimizer, 使 base __init__ 时创建的 SWA 对象指向过时的 model/optimizer.
                    # 在 SWA 真正开始的这一刻重建, 保证权重平均覆盖全部当前可训练模块, 且 SWALR 绑定活跃 optimizer.
                    self.swa_model = AveragedModel(self.model)
                    self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                self.swa_n += 1
                if self.swa_n == 1:
                    print(f"\n[SWA] Started at epoch {epoch + 1}, LR={self.swa_lr:.2e} (swa_model/scheduler rebuilt for current model+optimizer)")
            elif self.use_two_phase and (epoch + 1) >= self.phase2_start:
                # 2-Phase阶段: 用固定小LR微调
                if (epoch + 1) == self.phase2_start:
                    # 进入Phase2时，记录当前LR并缩放
                    for pg in self.optimizer.param_groups:
                        pg['phase2_base_lr'] = pg['lr']
                        pg['lr'] = pg['lr'] * self.phase2_lr_scale
                    phase2_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                    print(f"\n[2-Phase] Entering Phase2 at epoch {epoch + 2}, LRs={[f'{lr:.2e}' for lr in phase2_lrs]}")
                # Phase2不再step scheduler，保持固定LR
            else:
                self.scheduler.step()

        print("\n" + "=" * 60)
        print("Training Completed!")
        print(f"Best IoU: {self.best_metric:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        print("=" * 60)

        # 训练完成后自动测试 (best checkpoint)
        self.test()

        # ⭐ SWA: 更新BN统计量并测试SWA模型
        if self.use_swa and self.swa_n > 0:
            print(f"\n{'=' * 60}")
            print(f"[SWA] Evaluating SWA-averaged model ({self.swa_n} models averaged)")
            print(f"{'=' * 60}")
            print(f"[SWA] Updating BatchNorm statistics...")
            self._update_swa_bn()
            # 保存SWA模型
            swa_path = os.path.join(self.save_dir, 'swa_best.pth')
            torch.save(self.swa_model.module.state_dict(), swa_path)
            print(f"[SWA] Model saved to {swa_path}")
            # 用SWA模型替换当前模型并测试
            self.model.load_state_dict(self.swa_model.module.state_dict())
            self._swa_testing = True
            self.test()
            self._swa_testing = False


def main():
    # 从统一配置中获取数据集列表
    from dinobcd.datasets import DATASET_CONFIGS, list_available_datasets
    
    available_datasets = list_available_datasets()
    preset_names = [name.lower().replace('_', '') for name in available_datasets]
    
    parser = argparse.ArgumentParser(description='Train DINOBCD')
    parser.add_argument('--config', type=str, default='dinobcd/configs/default_config.yaml',
                        help='Path to config file')
    
    # 数据集命令行参数 - 动态从DATASET_CONFIGS获取
    parser.add_argument('--dataset', type=str, default=None,
                        choices=available_datasets,
                        help=f'Dataset name: {available_datasets}')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root directory (overrides config)')
    
    # 数据集预设 - 一键切换
    parser.add_argument('--preset', type=str, default=None,
                        help=f'Dataset preset: {preset_names}')
    
    # 其他常用参数
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--grad_accum', type=int, default=None,
                        help='Gradient accumulation steps (overrides config); effective batch = batch_size * grad_accum')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name for logging')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--focal_alpha', type=float, default=None,
                        help='Focal loss alpha for change class (default 0.75)')
    parser.add_argument('--auto_alpha', action='store_true',
                        help='Enable DACB: auto-compute focal_alpha from dataset')
    parser.add_argument('--edge_loss_weight', type=float, default=None,
                        help='Edge loss weight (overrides config)')
    parser.add_argument('--dice_weight', type=float, default=None,
                        help='Dice loss weight (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint. Pass "latest" to load <save_dir>/latest.pth, or an explicit .pth path.')

    # ⭐ 骨干网络解冻参数 (数据集特定调优)
    parser.add_argument('--backbone_lr', type=float, default=None,
                        help='Backbone learning rate for unfrozen blocks')
    parser.add_argument('--num_unfreeze_blocks', type=int, default=None,
                        help='Number of DINOv3 blocks to unfreeze (from end)')

    # ⭐ Tversky Loss参数 (P/R平衡调优)
    parser.add_argument('--tversky_alpha', type=float, default=None,
                        help='Tversky FP penalty weight (higher → more precision)')
    parser.add_argument('--tversky_beta', type=float, default=None,
                        help='Tversky FN penalty weight (higher → more recall)')

    # ⭐ 模块消融开关 (用于ablation study)
    parser.add_argument('--no_bti', action='store_true',
                        help='Ablation: disable BTI (Bidirectional Temporal Interaction)')
    parser.add_argument('--no_hfa', action='store_true',
                        help='Ablation: disable HFA (Hierarchical Feature Aggregation)')
    parser.add_argument('--no_msca', action='store_true',
                        help='Ablation: disable MSCA, use default DualDifference')
    parser.add_argument('--no_msfa', action='store_true',
                        help='Ablation: disable MSFA (Multi-Scale Feature Aggregation)')
    parser.add_argument('--no_swa', action='store_true',
                        help='Ablation: disable SWA (Stochastic Weight Averaging)')
    parser.add_argument('--no_tta', action='store_true',
                        help='Ablation: disable TTA at test time')
    parser.add_argument('--deep_supervision', action='store_true',
                        help='Enable Deep Supervision on FPN layers')
    parser.add_argument('--sgce', action='store_true',
                        help='Enable SGCE (Semantic-Guided Change Enhancement)')
    parser.add_argument('--hbca', action='store_true',
                        help='Enable HBCA (Hierarchical Bi-temporal Cross-Attention)')
    parser.add_argument('--btot', action='store_true',
                        help='Enable BTOT (Bi-Temporal Optimal Transport) diff module, 替换 HBCA/MSCA')
    parser.add_argument('--ssm', action='store_true',
                        help='Enable SSM (state-space / selective-scan) interaction operator, 受控对比 baseline')
    parser.add_argument('--hbca_window_size', type=int, default=None,
                        help='HBCA window size (default 8, ablation: 4/16)')
    parser.add_argument('--ot_window', type=int, default=None,
                        help='BTOT window size (default 8, ablation: 4/16)')
    parser.add_argument('--ot_iters', type=int, default=None,
                        help='BTOT Sinkhorn iterations (default 5, ablation: 1/3)')
    parser.add_argument('--ot_no_dustbin', action='store_true',
                        help='BTOT ablation: balanced OT without dustbin (evidence = transport-cost residual only)')
    parser.add_argument('--ot_fixed_bin', action='store_true',
                        help='BTOT ablation: dustbin score fixed at 0 instead of learnable')
    parser.add_argument('--ot_cost_only', action='store_true',
                        help='BTOT ablation: keep the dustbin plan but zero the unmatchable-mass channel')
    parser.add_argument('--no_gamma_gate', action='store_true',
                        help='Ablation: disable gamma-gated residual in HBCA (use direct addition)')
    parser.add_argument('--no_cross_scale', action='store_true',
                        help='Ablation: disable cross-scale context propagation in HBCA')
    parser.add_argument('--two_phase', action='store_true',
                        help='Enable 2-Phase LR schedule (90%% cosine + 10%% fine-tune)')
    parser.add_argument('--nochange_ratio', type=float, default=None,
                        help='Downsample no-change training samples (e.g. 0.5 = keep 50%%)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Override dataloader workers')
    parser.add_argument('--swa_start', type=int, default=None,
                        help='Override SWA start epoch (e.g. for short schedules)')
    parser.add_argument('--ohem_ratio', type=float, default=None,
                        help='Enable OHEM: keep hardest K%% pixels (e.g. 0.7)')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 动态生成预设映射 (从 DATASET_CONFIGS 自动生成)
    DATASET_PRESETS = {}
    for name, cfg in DATASET_CONFIGS.items():
        preset_key = name.lower().replace('_', '')  # LEVIR_CD -> levircd
        DATASET_PRESETS[preset_key] = {
            'dataset': name,
            'data_root': cfg['data_root'],
        }
        # 添加简短别名
        if name == 'LEVIR_CD':
            DATASET_PRESETS['levir'] = DATASET_PRESETS[preset_key]
        elif name == 'LEVIR_CD_PLUS':
            DATASET_PRESETS['levirplus'] = DATASET_PRESETS[preset_key]
        elif name == 'WHU_CD':
            DATASET_PRESETS['whu'] = DATASET_PRESETS[preset_key]
        elif name == 'S2LOOKING':
            DATASET_PRESETS['s2looking'] = DATASET_PRESETS[preset_key]
    
    if args.preset:
        preset_key = args.preset.lower().replace('_', '').replace('-', '')
        if preset_key not in DATASET_PRESETS:
            print(f"Unknown preset: {args.preset}")
            print(f"Available presets: {list(DATASET_PRESETS.keys())}")
            return
        preset = DATASET_PRESETS[preset_key]
        config['data']['dataset'] = preset['dataset']
        config['data']['data_root'] = preset['data_root']
        print(f"Using preset: {args.preset}")
        print(f"  Dataset: {preset['dataset']}")
        print(f"  Data root: {preset['data_root']}")
    
    # 命令行参数覆盖配置文件
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.data_root:
        config['data']['data_root'] = args.data_root
    if args.gpu is not None:
        config['device']['gpu_ids'] = [args.gpu]
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.grad_accum:
        config['training']['gradient_accumulation_steps'] = args.grad_accum
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers
    if args.swa_start is not None:
        config.setdefault('training', {}).setdefault('swa', {})['start_epoch'] = args.swa_start
        print(f"[SWA] start_epoch overridden to {args.swa_start}")
    if args.exp_name:
        config['experiment']['name'] = args.exp_name
        config['logging']['log_dir'] = f"logs/{args.exp_name}"
        config['logging']['save_dir'] = f"checkpoints/{args.exp_name}"
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    if args.focal_alpha:
        config['training']['loss']['focal_alpha'] = args.focal_alpha
    if args.auto_alpha:
        config['training']['loss']['auto_alpha'] = True
    if args.edge_loss_weight:
        config['training']['loss']['edge_loss_weight'] = args.edge_loss_weight
    if args.dice_weight:
        config['training']['loss']['dice_weight'] = args.dice_weight
    # ⭐ 骨干网络解冻参数覆盖
    if args.backbone_lr is not None:
        config['training']['optimizer']['backbone_lr'] = args.backbone_lr
    if args.num_unfreeze_blocks is not None:
        if 'backbone_unfreeze' not in config['model']:
            config['model']['backbone_unfreeze'] = {'enable': True}
        config['model']['backbone_unfreeze']['num_blocks'] = args.num_unfreeze_blocks
        config['model']['backbone_unfreeze']['enable'] = True

    # ⭐ Tversky参数覆盖
    if args.tversky_alpha is not None:
        config['training']['loss']['tversky_alpha'] = args.tversky_alpha
        config['training']['loss']['use_tversky'] = True
    if args.tversky_beta is not None:
        config['training']['loss']['tversky_beta'] = args.tversky_beta
        config['training']['loss']['tversky_alpha'] = 1.0 - args.tversky_beta
        config['training']['loss']['use_tversky'] = True
        print(f"[Loss] Tversky alpha={1.0-args.tversky_beta:.1f}, beta={args.tversky_beta}")

    # ⭐ 模块消融覆盖
    if args.no_bti:
        # 禁用BTI: 回退到独立提取 (通过禁用HFA中的temporal interaction)
        # BTI是HFA的前置模块, 单独禁用需要特殊处理
        config['model']['use_bti'] = False
        print("[Ablation] BTI disabled")
    if args.no_hfa:
        config['model']['use_hfa'] = False
        config['model']['dino_extract_layers'] = [5, 11, 17, 23]  # 回退到4层提取
        print("[Ablation] HFA disabled, using 4-layer extraction")
    if args.no_msca:
        config['model']['use_msca'] = False
        print("[Ablation] MSCA disabled, using default DualDifference")
    if args.no_msfa:
        config['model']['use_msfa'] = False
        print("[Ablation] MSFA disabled, using P1-only output")
    if args.no_swa:
        config['training']['swa']['enable'] = False
        print("[Ablation] SWA disabled")
    if args.no_tta:
        if 'evaluation' not in config:
            config['evaluation'] = {}
        config['evaluation']['use_tta'] = False
        print("[Ablation] TTA disabled")
    if args.deep_supervision:
        config['model']['use_deep_supervision'] = True
        print("[Model] Deep Supervision enabled via CLI")
    if args.sgce:
        config['model']['use_sgce'] = True
        print("[Model] SGCE enabled via CLI")
    if args.hbca:
        config['model']['use_hbca'] = True
        config['model']['use_msca'] = False  # HBCA替代MSCA
        print("[Model] HBCA enabled via CLI (replacing MSCA)")
    if args.btot:
        config['model']['use_btot'] = True
        config['model']['use_hbca'] = False
        config['model']['use_msca'] = False  # BTOT替代MSCA/HBCA
        print("[Model] BTOT enabled via CLI (replacing MSCA/HBCA)")
    if args.ot_window is not None or args.ot_iters is not None:
        bc = config['model'].get('btot_config') or {}
        if args.ot_window is not None:
            bc['window'] = args.ot_window
        if args.ot_iters is not None:
            bc['n_iters'] = args.ot_iters
        config['model']['btot_config'] = bc
        print(f"[BTOT] config override: window={bc.get('window', 8)}, n_iters={bc.get('n_iters', 5)}")
    if args.ot_no_dustbin or args.ot_fixed_bin or args.ot_cost_only:
        bc = config['model'].get('btot_config') or {}
        if args.ot_no_dustbin:
            bc['use_dustbin'] = False
        if args.ot_fixed_bin:
            bc['learn_bin'] = False
        if args.ot_cost_only:
            bc['cost_only'] = True
        config['model']['btot_config'] = bc
        print(f"[BTOT] ablation: use_dustbin={bc.get('use_dustbin', True)}, "
              f"learn_bin={bc.get('learn_bin', True)}, cost_only={bc.get('cost_only', False)}")
    if args.ssm:
        config['model']['use_ssm'] = True
        config['model']['use_btot'] = False
        config['model']['use_hbca'] = False
        config['model']['use_msca'] = False  # SSM替代MSCA/HBCA/BTOT
        print("[Model] SSM enabled via CLI (replacing MSCA/HBCA/BTOT)")
    if args.hbca_window_size is not None:
        if 'hbca_config' not in config['model']:
            config['model']['hbca_config'] = {}
        config['model']['hbca_config']['window_size'] = args.hbca_window_size
        print(f"[HBCA] Window size overridden to {args.hbca_window_size}")
    if args.no_gamma_gate:
        if 'hbca_config' not in config['model']:
            config['model']['hbca_config'] = {}
        config['model']['hbca_config']['use_gamma_gate'] = False
        print("[Ablation] HBCA gamma-gated residual disabled (direct addition)")
    if args.no_cross_scale:
        if 'hbca_config' not in config['model']:
            config['model']['hbca_config'] = {}
        config['model']['hbca_config']['use_cross_scale'] = False
        print("[Ablation] HBCA cross-scale context propagation disabled")
    if args.two_phase:
        if 'two_phase' not in config['training']:
            config['training']['two_phase'] = {}
        config['training']['two_phase']['enable'] = True
        print("[Training] 2-Phase LR schedule enabled via CLI")
    if args.nochange_ratio is not None:
        config['data']['nochange_ratio'] = args.nochange_ratio
        print(f"[Data] No-change downsample ratio: {args.nochange_ratio}")
    if args.ohem_ratio is not None:
        config['training']['loss']['ohem_ratio'] = args.ohem_ratio
        print(f"[Loss] OHEM ratio: {args.ohem_ratio}")

    if args.seed is not None:
        config['experiment']['seed'] = args.seed

        # 自动为不同seed创建独立的保存目录
        if args.exp_name:
            # 如果指定了exp_name，添加seed后缀
            config['logging']['log_dir'] = f"logs/{args.exp_name}_seed{args.seed}"
            config['logging']['save_dir'] = f"checkpoints/{args.exp_name}_seed{args.seed}"
        else:
            # 否则使用默认目录+dataset+seed
            dataset_name = config['data']['dataset'].lower()
            base_log_dir = config['logging']['log_dir']
            base_save_dir = config['logging']['save_dir']
            config['logging']['log_dir'] = f"{base_log_dir}_{dataset_name}_seed{args.seed}"
            config['logging']['save_dir'] = f"{base_save_dir}_{dataset_name}_seed{args.seed}"

        print(f"Using random seed: {args.seed}")
        print(f"Log directory: {config['logging']['log_dir']}")
        print(f"Checkpoint directory: {config['logging']['save_dir']}")

    # 创建训练器
    trainer = Trainer(config)

    # ⭐ 可选 resume
    if args.resume is not None:
        if args.resume == 'latest':
            # trainer.save_dir 已经包含 dataset 子目录 (e.g. .../whu_cd)
            ckpt_path = os.path.join(trainer.save_dir, 'latest.pth')
            if not os.path.isfile(ckpt_path):
                # fallback: 取最大 epoch_N.pth
                import glob, re
                cands = glob.glob(os.path.join(trainer.save_dir, 'epoch_*.pth'))
                if cands:
                    def _ep(p):
                        m = re.search(r'epoch_(\d+)\.pth$', p)
                        return int(m.group(1)) if m else -1
                    ckpt_path = max(cands, key=_ep)
                    print(f"[Resume] latest.pth not found, fell back to {ckpt_path}")
        else:
            ckpt_path = args.resume
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--resume: checkpoint not found at {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()

