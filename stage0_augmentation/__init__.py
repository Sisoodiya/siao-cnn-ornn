"""
Stage 0: Data Augmentation Module

Contains TimeGAN-based synthetic data generation
and quality validation utilities for expanding limited NPP datasets.
"""

from .time_gan_augmentor import TimeGANAugmentor
from .quality_metrics import (
    plot_pca_comparison,
    calculate_discriminative_score,
    calculate_predictive_score,
    validate_synthetic_data
)
from .mixup import (
    MixupAugmentor,
    CutmixAugmentor,
    mixup_criterion,
    blend_real_synthetic
)

__all__ = [
    'TimeGANAugmentor',
    'plot_pca_comparison',
    'calculate_discriminative_score', 
    'calculate_predictive_score',
    'validate_synthetic_data',
    'MixupAugmentor',
    'CutmixAugmentor',
    'mixup_criterion',
    'blend_real_synthetic'
]

