"""
IP-200 Reactor Data Ingestion Pipeline

A scalable Python pipeline for ingesting, preprocessing, and structuring
RELAP5 simulation data for nuclear reactor operating class classification.

Supports 6 operating classes:
1. Steady State (100% power)
2. Transient / Power Change (60%-80%)
3. Pressurizer PORV Stuck Open (100%)
4. Steam Generator Tube Rupture (10%)
5. Feedwater Line Break (50%)
6. Reactor Coolant Pump Failure (1 out of 4)

Author: Nuclear Systems Data Engineer
"""

import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

CLASS_MAP = {
    'steady_state': 0,
    'transient_power_change': 1,
    'porv_stuck_open': 2,
    'sg_tube_rupture': 3,
    'feedwater_break': 4,
    'rcp_failure': 5
}

CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# Filename patterns for class detection
CLASS_PATTERNS = {
    'steady_state': [
        r'steady\s*state',
        r'ss\d+',
    ],
    'transient_power_change': [
        r'power\s*change',
        r'pp\d+p',
        r'ip200-pp',
        r'transient',
    ],
    'porv_stuck_open': [
        r'pzrv',
        r'porv',
        r'pressurizer.*porv',
    ],
    'sg_tube_rupture': [
        r'sgtr',
        r'sg\s*tube\s*rupture',
        r'steam.*generator.*tube',
    ],
    'feedwater_break': [
        r'fwb',
        r'feedwater.*break',
        r'feed\s*water',
    ],
    'rcp_failure': [
        r'rcp',
        r'pump\s*failure',
        r'reactor.*coolant.*pump',
    ],
}


# =============================================================================
# Data Pipeline Class
# =============================================================================

class IP200DataPipeline:
    """
    Main data ingestion pipeline for IP-200 reactor simulation data.
    
    Handles:
    - Multi-format loading (Excel, CSV, MAT)
    - Class label inference from filenames
    - Missing value handling
    - Outlier smoothing
    - Feature normalization
    - Tensor output generation
    """
    
    def __init__(
        self,
        data_dir: str = 'data/',
        time_column: str = 'time000000000',
        exclude_columns: Optional[List[str]] = None,
        normalization: str = 'zscore',  # 'zscore', 'minmax', or 'none'
        outlier_window: int = 5,
        handle_missing: str = 'interpolate',  # 'interpolate', 'ffill', 'drop'
    ):
        """
        Initialize the data pipeline.
        
        Args:
            data_dir: Path to directory containing data files
            time_column: Name of the time column
            exclude_columns: Columns to exclude from features
            normalization: Normalization method ('zscore', 'minmax', 'none')
            outlier_window: Window size for rolling median smoothing
            handle_missing: Missing value strategy
        """
        self.data_dir = Path(data_dir)
        self.time_column = time_column
        self.exclude_columns = exclude_columns or ['condition', 'class', 'label']
        self.normalization = normalization
        self.outlier_window = outlier_window
        self.handle_missing = handle_missing
        
        # Storage for normalization parameters (for inference)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.feature_columns: List[str] = []
        
        logger.info(f"Initialized IP200DataPipeline with data_dir={data_dir}")
    
    # -------------------------------------------------------------------------
    # File Loading
    # -------------------------------------------------------------------------
    
    def _load_single_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single data file (Excel, CSV, or MAT)."""
        suffix = filepath.suffix.lower()
        
        try:
            if suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath, engine='openpyxl')
            elif suffix == '.csv':
                df = pd.read_csv(filepath)
            elif suffix == '.mat':
                # MAT file support (requires scipy)
                from scipy.io import loadmat
                mat_data = loadmat(str(filepath))
                # Convert first non-meta key to DataFrame
                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if data_keys:
                    df = pd.DataFrame(mat_data[data_keys[0]])
                else:
                    raise ValueError(f"No data found in MAT file: {filepath}")
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            logger.debug(f"Loaded {filepath.name}: shape={df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
    
    def _infer_class_from_filename(self, filename: str) -> str:
        """Infer operating class from filename using patterns."""
        filename_lower = filename.lower()
        
        for class_name, patterns in CLASS_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename_lower):
                    return class_name
        
        logger.warning(f"Could not infer class from filename: {filename}")
        return 'unknown'
    
    def load_all_data(
        self,
        subdirs: Optional[List[str]] = None,
        file_pattern: str = '*.xlsx'
    ) -> Tuple[List[pd.DataFrame], List[int]]:
        """
        Load all data files from the data directory.
        
        Args:
            subdirs: Specific subdirectories to include
            file_pattern: Glob pattern for files
        
        Returns:
            Tuple of (list of DataFrames, list of class labels)
        """
        all_dfs = []
        all_labels = []
        
        # Collect files from main directory and subdirectories
        search_dirs = [self.data_dir]
        if subdirs:
            search_dirs.extend([self.data_dir / sd for sd in subdirs])
        else:
            # Include all subdirectories
            search_dirs.extend([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            for filepath in search_dir.glob(file_pattern):
                # Skip processed/output files
                if 'processed' in str(filepath).lower():
                    continue
                if 'predicted' in filepath.name.lower():
                    continue
                    
                try:
                    df = self._load_single_file(filepath)
                    class_name = self._infer_class_from_filename(filepath.name)
                    
                    if class_name == 'unknown':
                        continue
                    
                    class_label = CLASS_MAP[class_name]
                    all_dfs.append(df)
                    all_labels.append(class_label)
                    
                    logger.info(f"Loaded {filepath.name} -> class={class_name} ({class_label})")
                    
                except Exception as e:
                    logger.warning(f"Skipping {filepath}: {e}")
        
        logger.info(f"Loaded {len(all_dfs)} files total")
        self._log_class_distribution(all_labels)
        
        return all_dfs, all_labels
    
    def _log_class_distribution(self, labels: List[int]) -> None:
        """Log the distribution of classes."""
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Class distribution:")
        for label, count in zip(unique, counts):
            class_name = CLASS_NAMES.get(label, 'unknown')
            logger.info(f"  {class_name} ({label}): {count} samples")
    
    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        missing_count = df.isna().sum().sum()
        
        if missing_count == 0:
            return df
        
        logger.info(f"Handling {missing_count} missing values using '{self.handle_missing}'")
        
        if self.handle_missing == 'interpolate':
            # Linear interpolation for time-series
            df = df.interpolate(method='linear', limit_direction='both')
            # Fill any remaining NaNs at edges
            df = df.ffill().bfill()
        elif self.handle_missing == 'ffill':
            df = df.ffill().bfill()
        elif self.handle_missing == 'drop':
            df = df.dropna()
        
        remaining = df.isna().sum().sum()
        if remaining > 0:
            logger.warning(f"Still {remaining} missing values after handling")
        
        return df
    
    def _smooth_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply rolling median smoothing to reduce outliers."""
        if self.outlier_window <= 1:
            return df
        
        logger.info(f"Smoothing outliers with rolling median (window={self.outlier_window})")
        
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col] = df[col].rolling(
                    window=self.outlier_window,
                    center=True,
                    min_periods=1
                ).median()
        
        return df
    
    def _normalize(self, df: pd.DataFrame, columns: List[str], fit: bool = True) -> pd.DataFrame:
        """Apply normalization to feature columns."""
        if self.normalization == 'none':
            return df
        
        logger.info(f"Applying {self.normalization} normalization")
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if fit:
                # Compute and store statistics
                if self.normalization == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    std = std if std > 0 else 1.0
                    self.feature_stats[col] = {'mean': mean, 'std': std}
                elif self.normalization == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    range_val = max_val - min_val
                    range_val = range_val if range_val > 0 else 1.0
                    self.feature_stats[col] = {'min': min_val, 'range': range_val}
            
            # Apply normalization
            stats = self.feature_stats.get(col, {})
            if self.normalization == 'zscore':
                df[col] = (df[col] - stats.get('mean', 0)) / stats.get('std', 1)
            elif self.normalization == 'minmax':
                df[col] = (df[col] - stats.get('min', 0)) / stats.get('range', 1)
        
        return df
    
    def preprocess_dataframe(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply full preprocessing pipeline to a DataFrame."""
        # Identify feature columns
        feature_cols = [
            c for c in df.columns 
            if c not in self.exclude_columns and c != self.time_column
        ]
        
        if fit and not self.feature_columns:
            self.feature_columns = feature_cols
            logger.info(f"Detected {len(feature_cols)} feature columns")
        
        # Apply preprocessing steps
        df = self._handle_missing_values(df)
        df = self._smooth_outliers(df, feature_cols)
        df = self._normalize(df, feature_cols, fit=fit)
        
        return df
    
    # -------------------------------------------------------------------------
    # Tensor Creation
    # -------------------------------------------------------------------------
    
    def create_tensors(
        self,
        dataframes: List[pd.DataFrame],
        labels: List[int],
        pad_to_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert list of DataFrames to tensor format.
        
        Args:
            dataframes: List of preprocessed DataFrames
            labels: List of class labels
            pad_to_length: Pad/truncate all samples to this length
        
        Returns:
            X: numpy array of shape [samples, time_steps, features]
            y: numpy array of shape [samples]
        """
        if not dataframes:
            raise ValueError("No dataframes provided")
        
        # Determine feature columns from first DataFrame
        sample_df = dataframes[0]
        feature_cols = [
            c for c in sample_df.columns 
            if c not in self.exclude_columns and c != self.time_column
        ]
        num_features = len(feature_cols)
        
        # Determine max time steps if not specified
        if pad_to_length is None:
            pad_to_length = max(len(df) for df in dataframes)
        
        logger.info(f"Creating tensors: {len(dataframes)} samples, {pad_to_length} time steps, {num_features} features")
        
        # Initialize output arrays
        X = np.zeros((len(dataframes), pad_to_length, num_features), dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        
        # Fill tensor
        for i, df in enumerate(dataframes):
            # Extract features
            features = df[feature_cols].values
            
            # Pad or truncate
            actual_length = min(len(features), pad_to_length)
            X[i, :actual_length, :] = features[:actual_length]
        
        # Validate shapes
        self._validate_tensor_shapes(X, y)
        
        return X, y
    
    def _validate_tensor_shapes(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate output tensor shapes."""
        assert len(X.shape) == 3, f"X must be 3D, got shape {X.shape}"
        assert len(y.shape) == 1, f"y must be 1D, got shape {y.shape}"
        assert X.shape[0] == y.shape[0], f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}"
        
        # Check for NaN/Inf
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0:
            logger.warning(f"X contains {nan_count} NaN values")
        if inf_count > 0:
            logger.warning(f"X contains {inf_count} Inf values")
        
        # Check class labels
        unique_labels = set(np.unique(y))
        valid_labels = set(CLASS_MAP.values())
        if not unique_labels.issubset(valid_labels):
            logger.warning(f"Invalid labels found: {unique_labels - valid_labels}")
        
        logger.info(f" Shape validation passed: X={X.shape}, y={y.shape}")
    
    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------
    
        return X, y
    
    def run(
        self,
        subdirs: Optional[List[str]] = None,
        file_pattern: str = '*.xlsx',
        pad_to_length: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: str = 'data/processed/'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the complete data ingestion pipeline.
        
        Args:
            subdirs: Specific subdirectories to process
            file_pattern: Glob pattern for data files
            pad_to_length: Pad all samples to this length
            use_cache: Whether to use cached .npy files
            cache_dir: Directory for cache files
        
        Returns:
            X: numpy array [samples, time_steps, features]
            y: numpy array [samples]
        """
        logger.info("=" * 60)
        logger.info("Starting IP-200 Data Ingestion Pipeline")
        logger.info("=" * 60)
        
        # Check cache
        cache_path = Path(cache_dir)
        x_cache = cache_path / 'X.npy'
        y_cache = cache_path / 'y.npy'
        
        if use_cache and x_cache.exists() and y_cache.exists():
            logger.info(f"Loading cached data from {cache_dir}")
            try:
                X = np.load(x_cache)
                y = np.load(y_cache)
                logger.info(f"Loaded cached tensors: X={X.shape}, y={y.shape}")
                return X, y
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Proceeding with full ingestion.")
        
        # Step 1: Load all data
        dataframes, labels = self.load_all_data(subdirs, file_pattern)
        
        if not dataframes:
            raise ValueError("No data files found!")
        
        # Step 2: Preprocess each DataFrame
        logger.info("Preprocessing data...")
        processed_dfs = []
        for i, df in enumerate(dataframes):
            fit = (i == 0)  # Only fit stats on first sample
            processed_df = self.preprocess_dataframe(df, fit=fit)
            processed_dfs.append(processed_df)
        
        # Step 3: Create output tensors
        X, y = self.create_tensors(processed_dfs, labels, pad_to_length)
        
        # Save cache
        if use_cache:
            self.save_tensors(X, y, output_dir=cache_dir)
            
        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info(f"  Output X shape: {X.shape}")
        logger.info(f"  Output y shape: {y.shape}")
        logger.info(f"  Feature columns: {len(self.feature_columns)}")
        logger.info("=" * 60)
        
        return X, y
    
    def save_tensors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str = 'data/processed/'
    ) -> None:
        """Save processed tensors to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'X.npy', X)
        np.save(output_path / 'y.npy', y)
        
        # Save feature column names
        with open(output_path / 'feature_columns.txt', 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        logger.info(f"Saved tensors to {output_path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def load_processed_data(data_dir: str = 'data/processed/') -> Tuple[np.ndarray, np.ndarray]:
    """Load previously processed tensors."""
    data_path = Path(data_dir)
    X = np.load(data_path / 'X.npy')
    y = np.load(data_path / 'y.npy')
    return X, y


def get_class_name(label: int) -> str:
    """Get class name from label."""
    return CLASS_NAMES.get(label, 'unknown')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Example usage
    pipeline = IP200DataPipeline(
        data_dir='data/',
        normalization='zscore',
        outlier_window=5,
        handle_missing='interpolate'
    )
    
    X, y = pipeline.run()
    
    print(f"\nFinal Output:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Features: {len(pipeline.feature_columns)}")
