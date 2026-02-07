"""
Mixup Data Augmentation for Time-Series

Implements mixup augmentation technique to improve model generalization
by creating convex combinations of training examples.

Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018)
"""

import numpy as np
import torch
from typing import Tuple, Optional


class MixupAugmentor:
    """
    Mixup augmentation for time-series data.
    
    Creates virtual training examples by linearly interpolating
    between pairs of training samples and their labels.
    """
    
    def __init__(
        self,
        alpha: float = 0.2,
        prob: float = 0.5,
        same_class_only: bool = False
    ):
        """
        Initialize MixupAugmentor.
        
        Args:
            alpha: Beta distribution parameter (higher = more mixing)
            prob: Probability of applying mixup to each batch
            same_class_only: If True, only mix samples from same class
        """
        self.alpha = alpha
        self.prob = prob
        self.same_class_only = same_class_only
    
    def __call__(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation.
        
        Args:
            X: Input data [batch, seq_len, features] or [batch, channels, seq_len]
            y: Labels [batch] (class indices)
            
        Returns:
            X_mixed: Mixed inputs
            y_a: Original labels
            y_b: Mixed labels  
            lam: Mixing coefficient
        """
        if np.random.random() > self.prob:
            return X, y, y, 1.0
        
        batch_size = X.size(0)
        
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Create random permutation for mixing pairs
        if self.same_class_only:
            # Shuffle within each class
            indices = self._same_class_shuffle(y)
        else:
            indices = torch.randperm(batch_size, device=X.device)
        
        # Mix inputs
        X_mixed = lam * X + (1 - lam) * X[indices]
        
        # Return original and shuffled labels (for mixup loss)
        y_a = y
        y_b = y[indices]
        
        return X_mixed, y_a, y_b, lam
    
    def _same_class_shuffle(self, y: torch.Tensor) -> torch.Tensor:
        """Shuffle indices within each class."""
        indices = torch.arange(len(y), device=y.device)
        
        for c in torch.unique(y):
            mask = y == c
            class_indices = indices[mask]
            
            # Shuffle class indices
            perm = torch.randperm(len(class_indices), device=y.device)
            indices[mask] = class_indices[perm]
        
        return indices


def mixup_criterion(
    criterion: torch.nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute mixup loss.
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a: Original labels
        y_b: Mixed labels
        lam: Mixing coefficient
        
    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutmixAugmentor:
    """
    CutMix augmentation adapted for 1D time-series.
    
    Cuts and pastes segments between training samples.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5
    ):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        if np.random.random() > self.prob:
            return X, y, y, 1.0
        
        batch_size = X.size(0)
        seq_len = X.size(1)
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Calculate cut boundaries
        cut_ratio = np.sqrt(1.0 - lam)
        cut_len = int(seq_len * cut_ratio)
        
        # Random cut start position
        cut_start = np.random.randint(0, seq_len - cut_len + 1)
        cut_end = cut_start + cut_len
        
        # Create random permutation
        indices = torch.randperm(batch_size, device=X.device)
        
        # Apply cut
        X_mixed = X.clone()
        X_mixed[:, cut_start:cut_end, :] = X[indices, cut_start:cut_end, :]
        
        # Adjust lambda based on actual cut ratio
        lam = 1 - cut_len / seq_len
        
        return X_mixed, y, y[indices], lam


def apply_augmentation_batch(
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    augmentor: Optional[MixupAugmentor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply augmentation to a batch.
    
    Convenience function for training loop integration.
    """
    if augmentor is None:
        return X_batch, y_batch, y_batch, 1.0
    
    return augmentor(X_batch, y_batch)


# =============================================================================
# Synthetic Data Blending
# =============================================================================

def blend_real_synthetic(
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_synth: np.ndarray, 
    y_synth: np.ndarray,
    synth_ratio: float = 0.3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend real and synthetic data with controlled ratio.
    
    Args:
        X_real: Real data samples
        y_real: Real labels
        X_synth: Synthetic data samples
        y_synth: Synthetic labels
        synth_ratio: Target ratio of synthetic data (0.0-1.0)
        seed: Random seed
        
    Returns:
        X_blended, y_blended: Combined dataset
    """
    np.random.seed(seed)
    
    n_real = len(X_real)
    n_synth_target = int(n_real * synth_ratio / (1 - synth_ratio))
    n_synth_use = min(n_synth_target, len(X_synth))
    
    if n_synth_use < len(X_synth):
        # Subsample synthetic data
        indices = np.random.choice(len(X_synth), n_synth_use, replace=False)
        X_synth_use = X_synth[indices]
        y_synth_use = y_synth[indices]
    else:
        X_synth_use = X_synth
        y_synth_use = y_synth
    
    # Combine
    X_blended = np.concatenate([X_real, X_synth_use], axis=0)
    y_blended = np.concatenate([y_real, y_synth_use], axis=0)
    
    # Shuffle
    perm = np.random.permutation(len(X_blended))
    
    return X_blended[perm], y_blended[perm]
