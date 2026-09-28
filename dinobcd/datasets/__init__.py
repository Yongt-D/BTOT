"""
DINOBCD Datasets Module

使用方法:
1. 在 unified_dataset.py 的 DATASET_CONFIGS 中添加新数据集配置
2. 运行时通过 --preset 或 --dataset 切换

支持的数据集: LEVIR_CD, LEVIR_CD_PLUS, WHU_CD, S2LOOKING
"""

from .unified_dataset import (
    ChangeDetectionDataset,
    DATASET_CONFIGS,
    build_dataloaders,
    get_dataset_config,
    list_available_datasets,
    add_dataset,
)

__all__ = [
    'ChangeDetectionDataset',
    'DATASET_CONFIGS',
    'build_dataloaders',
    'get_dataset_config',
    'list_available_datasets',
    'add_dataset',
]
