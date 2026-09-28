"""
DINOBCD - Unified Change Detection Dataset
统一的变化检测数据集类 - 通过字典配置支持多种数据集

使用方法:
1. 在 DATASET_CONFIGS 中添加新数据集配置
2. 运行时通过 --preset 或 --dataset 切换
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from typing import Dict, List, Optional, Tuple


# ==================== 数据集配置字典 ====================
# 在这里添加新数据集，无需创建新的类
DATASET_CONFIGS: Dict[str, Dict] = {
    'LEVIR_CD': {
        'name': 'LEVIR-CD',
        'data_root': 'data_dir/LEVIR-CD',
        'img1_dir': 'A',
        'img2_dir': 'B',
        'label_dir': 'label',
        'extensions': ['.png', '.jpg', '.tif'],
    },
    'LEVIR_CD_PLUS': {
        'name': 'LEVIR-CD+',
        'data_root': 'data_dir/LEVIR-CD_plus',
        'img1_dir': 'A',
        'img2_dir': 'B',
        'label_dir': 'label',
        'extensions': ['.png', '.jpg', '.tif'],
    },
    'WHU_CD': {
        'name': 'WHU-CD',
        'data_root': 'data_dir/WHU-CD',
        'img1_dir': 'A',
        'img2_dir': 'B',
        'label_dir': 'label',
        'extensions': ['.png', '.jpg', '.tif'],
    },
    'S2LOOKING': {
        'name': 'S2Looking',
        'data_root': 'data_dir/S2Looking',
        'img1_dir': 'Image1',  # S2Looking使用不同的目录名
        'img2_dir': 'Image2',
        'label_dir': 'label',
        'extensions': ['.png', '.jpg', '.tif'],
        # 备用目录名，如果主目录不存在则使用备用
        'fallback_img1_dir': 'A',
        'fallback_img2_dir': 'B',
    },
    # ============ 在这里添加新数据集 ============
    # 'NEW_DATASET': {
    #     'name': 'New Dataset',
    #     'data_root': 'data_dir/new_dataset',
    #     'img1_dir': 'A',
    #     'img2_dir': 'B',
    #     'label_dir': 'label',
    #     'extensions': ['.png', '.jpg', '.tif'],
    # },
}


def get_dataset_config(dataset_name: str) -> Dict:
    """获取数据集配置"""
    name = dataset_name.upper().replace('-', '_').replace(' ', '_')
    if name not in DATASET_CONFIGS:
        available = list(DATASET_CONFIGS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    return DATASET_CONFIGS[name].copy()


def list_available_datasets() -> List[str]:
    """列出所有可用数据集"""
    return list(DATASET_CONFIGS.keys())


class ChangeDetectionDataset(Dataset):
    """
    通用变化检测数据集
    
    支持所有在 DATASET_CONFIGS 中配置的数据集
    """
    
    def __init__(
        self,
        dataset_name: str = 'LEVIR_CD',
        data_root: Optional[str] = None,
        list_file: str = 'train.txt',
        image_size: int = 256,
        is_train: bool = True,
        augmentation: Optional[dict] = None,
        nochange_ratio: float = 1.0
    ):
        """
        Args:
            dataset_name: 数据集名称 (LEVIR_CD, WHU_CD, S2LOOKING, etc.)
            data_root: 数据目录，如果为None则使用配置中的默认路径
            list_file: 样本列表文件
            image_size: 图像尺寸
            is_train: 是否训练模式
            augmentation: 数据增强配置
            nochange_ratio: 保留无变化样本的比例 (0.0-1.0), 1.0=全部保留
        """
        # 获取数据集配置
        config = get_dataset_config(dataset_name)

        self.dataset_name = dataset_name
        self.data_root = data_root or config['data_root']
        self.image_size = image_size
        self.is_train = is_train
        self.augmentation = augmentation or {}

        # 确定目录结构
        self.img1_dir = config['img1_dir']
        self.img2_dir = config['img2_dir']
        self.label_dir = config['label_dir']
        self.extensions = config['extensions']

        # 检查备用目录
        if not os.path.exists(os.path.join(self.data_root, self.img1_dir)):
            fallback1 = config.get('fallback_img1_dir')
            fallback2 = config.get('fallback_img2_dir')
            if fallback1 and os.path.exists(os.path.join(self.data_root, fallback1)):
                self.img1_dir = fallback1
                self.img2_dir = fallback2

        # 读取样本列表
        self.samples = self._load_samples(list_file)

        # ⭐ 无变化样本降采样: 减少负样本比例, 缓解模型过度保守
        if is_train and nochange_ratio < 1.0:
            self.samples = self._downsample_nochange(nochange_ratio)

        print(f"[{config['name']}] Loaded {len(self.samples)} samples (is_train={is_train})")
        
        # 图像归一化
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _downsample_nochange(self, ratio: float) -> List[str]:
        """降采样无变化样本, 保留所有有变化样本"""
        import numpy as np
        from PIL import Image as PILImage

        change_samples = []
        nochange_samples = []
        for fname in self.samples:
            label_path = os.path.join(self.data_root, self.label_dir, fname)
            if os.path.exists(label_path):
                label = np.array(PILImage.open(label_path))
                if (label > 0).any():
                    change_samples.append(fname)
                else:
                    nochange_samples.append(fname)
            else:
                change_samples.append(fname)  # 找不到label的保留

        n_keep = max(1, int(len(nochange_samples) * ratio))
        rng = np.random.RandomState(42)
        kept_nochange = list(rng.choice(nochange_samples, size=n_keep, replace=False))

        result = change_samples + kept_nochange
        rng.shuffle(result)
        print(f"[NoChange Downsample] {len(change_samples)} change + "
              f"{n_keep}/{len(nochange_samples)} no-change = {len(result)} total "
              f"(ratio={ratio:.2f})")
        return result

    def _load_samples(self, list_file: str) -> List[str]:
        """加载样本列表"""
        samples = []
        
        # 尝试多个可能的列表文件路径
        possible_paths = [
            list_file,  # 绝对路径或相对路径
            os.path.join(self.data_root, 'list', os.path.basename(list_file)),
            os.path.join(self.data_root, list_file),
        ]
        
        list_path = None
        for path in possible_paths:
            if os.path.exists(path):
                list_path = path
                break
        
        if list_path:
            with open(list_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(line)
        else:
            # 自动扫描目录
            img1_full_dir = os.path.join(self.data_root, self.img1_dir)
            if os.path.exists(img1_full_dir):
                for fname in sorted(os.listdir(img1_full_dir)):
                    if any(fname.endswith(ext) for ext in self.extensions):
                        samples.append(fname)
            else:
                raise FileNotFoundError(
                    f"Cannot find sample list or image directory for {self.dataset_name}. "
                    f"Tried: {possible_paths} and {img1_full_dir}"
                )
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename = self.samples[idx]
        
        # 构建路径
        img1_path = os.path.join(self.data_root, self.img1_dir, filename)
        img2_path = os.path.join(self.data_root, self.img2_dir, filename)
        label_path = os.path.join(self.data_root, self.label_dir, filename)
        
        # 读取图像
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = Image.open(label_path).convert('L')
        
        # Resize
        if self.image_size:
            img1 = img1.resize((self.image_size, self.image_size), Image.BILINEAR)
            img2 = img2.resize((self.image_size, self.image_size), Image.BILINEAR)
            label = label.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # 训练时数据增强
        if self.is_train and self.augmentation:
            img1, img2, label = self._augment(img1, img2, label)
        
        # 转换为tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        label = torch.from_numpy(np.array(label)).long()
        
        # 标签二值化
        label = (label > 0).long()
        
        # 归一化
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        
        return {
            'img1': img1,
            'img2': img2,
            'label': label,
            'filename': filename
        }
    
    def _augment(self, img1, img2, label):
        """变化检测专用数据增强 - 优化版"""
        aug_config = self.augmentation.get('train', {})

        # ⭐ 时序互换: 50%概率交换T1/T2 (变化标签不变, 增强时序对称性)
        if aug_config.get('temporal_swap', False) and random.random() > 0.5:
            img1, img2 = img2, img1

        # 随机水平翻转
        if aug_config.get('random_flip', False) and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
            label = TF.hflip(label)

        # 随机垂直翻转
        if aug_config.get('random_flip', False) and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)
            label = TF.vflip(label)

        # 随机旋转 (90度倍数，适合遥感图像)
        if aug_config.get('random_rotate', False):
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img1 = TF.rotate(img1, angle)
                img2 = TF.rotate(img2, angle)
                label = TF.rotate(label, angle)

        # 随机缩放 (模拟不同分辨率/高度拍摄, 增强泛化)
        scale_range = aug_config.get('random_scale', None)
        if scale_range and random.random() > 0.5:
            if isinstance(scale_range, (list, tuple)) and len(scale_range) == 2:
                scale = random.uniform(scale_range[0], scale_range[1])
                w, h = img1.size
                new_w, new_h = int(w * scale), int(h * scale)
                img1 = img1.resize((new_w, new_h), Image.BILINEAR)
                img2 = img2.resize((new_w, new_h), Image.BILINEAR)
                label = label.resize((new_w, new_h), Image.NEAREST)
                # 裁剪或填充回原始大小
                if scale > 1.0:
                    # 随机裁剪
                    left = random.randint(0, new_w - w)
                    top = random.randint(0, new_h - h)
                    img1 = img1.crop((left, top, left + w, top + h))
                    img2 = img2.crop((left, top, left + w, top + h))
                    label = label.crop((left, top, left + w, top + h))
                else:
                    # 填充回原始大小 (边缘填0)
                    padded1 = Image.new('RGB', (w, h), (0, 0, 0))
                    padded2 = Image.new('RGB', (w, h), (0, 0, 0))
                    padded_label = Image.new('L', (w, h), 0)
                    left = (w - new_w) // 2
                    top = (h - new_h) // 2
                    padded1.paste(img1, (left, top))
                    padded2.paste(img2, (left, top))
                    padded_label.paste(label, (left, top))
                    img1, img2, label = padded1, padded2, padded_label

        # 随机高斯噪声 (增强鲁棒性)
        noise_sigma = aug_config.get('gaussian_noise', 0.0)
        if noise_sigma > 0 and random.random() > 0.5:
            img1_np = np.array(img1).astype(np.float32) / 255.0
            img2_np = np.array(img2).astype(np.float32) / 255.0
            noise1 = np.random.normal(0, noise_sigma, img1_np.shape).astype(np.float32)
            noise2 = np.random.normal(0, noise_sigma, img2_np.shape).astype(np.float32)
            img1_np = np.clip(img1_np + noise1, 0, 1)
            img2_np = np.clip(img2_np + noise2, 0, 1)
            img1 = Image.fromarray((img1_np * 255).astype(np.uint8))
            img2 = Image.fromarray((img2_np * 255).astype(np.uint8))

        # 独立光照增强 (模拟不同时间/季节拍摄的真实场景)
        # 这是变化检测的关键：T1和T2应该有不同的光照条件
        color_jitter = aug_config.get('color_jitter', {})
        if color_jitter:
            brightness = color_jitter.get('brightness', 0)
            contrast = color_jitter.get('contrast', 0)
            saturation = color_jitter.get('saturation', 0)
            hue = color_jitter.get('hue', 0)

            if any([brightness, contrast, saturation, hue]):
                # 默认使用独立的color jitter增强泛化能力
                if aug_config.get('independent_color_jitter', True):
                    # 为img1和img2分别应用不同的随机光照变换
                    color_transform1 = T.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue
                    )
                    color_transform2 = T.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue
                    )
                    img1 = color_transform1(img1)
                    img2 = color_transform2(img2)
                else:
                    # 使用相同的color jitter (保守策略)
                    color_transform = T.ColorJitter(
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue
                    )
                    img1 = color_transform(img1)
                    img2 = color_transform(img2)

        return img1, img2, label


# ==================== 数据加载器构建函数 ====================

def build_dataloaders(
    data_config: dict,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    根据配置构建数据加载器
    
    Args:
        data_config: 数据配置字典
        batch_size: 批次大小
        num_workers: 数据加载线程数
        pin_memory: 是否使用pinned memory
    
    Returns:
        train_loader, val_loader
    """
    dataset_name = data_config.get('dataset', 'LEVIR_CD')
    data_root = data_config.get('data_root', None)
    image_size = data_config.get('image_size', 256)
    augmentation = data_config.get('augmentation', {})
    train_list = data_config.get('train_list', 'train.txt')
    val_list = data_config.get('val_list', 'val.txt')
    nochange_ratio = data_config.get('nochange_ratio', 1.0)

    # 创建训练集
    train_dataset = ChangeDetectionDataset(
        dataset_name=dataset_name,
        data_root=data_root,
        list_file=train_list,
        image_size=image_size,
        is_train=True,
        augmentation=augmentation,
        nochange_ratio=nochange_ratio
    )
    
    # 创建验证集
    val_dataset = ChangeDetectionDataset(
        dataset_name=dataset_name,
        data_root=data_root,
        list_file=val_list,
        image_size=image_size,
        is_train=False,
        augmentation=None
    )
    
    # 创建DataLoader
    # num_workers>0 时启用 persistent_workers + prefetch, 消除每个 epoch 重启 worker 的停顿,
    # 让 GPU 更不易因数据等待而空转 (吃满 GPU).
    _extra = {}
    if num_workers and num_workers > 0:
        _extra = {"persistent_workers": True, "prefetch_factor": 4}
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **_extra,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **_extra,
    )

    return train_loader, val_loader


# ==================== 便捷函数 ====================

def add_dataset(
    name: str,
    data_root: str,
    img1_dir: str = 'A',
    img2_dir: str = 'B',
    label_dir: str = 'label',
    extensions: List[str] = None
):
    """
    动态添加新数据集配置
    
    Args:
        name: 数据集名称 (如 'MY_DATASET')
        data_root: 数据根目录
        img1_dir: T1图像目录名
        img2_dir: T2图像目录名
        label_dir: 标签目录名
        extensions: 支持的文件扩展名
    """
    DATASET_CONFIGS[name.upper()] = {
        'name': name,
        'data_root': data_root,
        'img1_dir': img1_dir,
        'img2_dir': img2_dir,
        'label_dir': label_dir,
        'extensions': extensions or ['.png', '.jpg', '.tif'],
    }
    print(f"Added dataset: {name}")


if __name__ == "__main__":
    # 测试代码
    print("Available datasets:", list_available_datasets())
    
    # 测试加载
    for name in ['LEVIR_CD', 'WHU_CD', 'S2LOOKING']:
        config = get_dataset_config(name)
        print(f"\n{name}: {config}")
