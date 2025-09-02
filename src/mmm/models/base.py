"""
Base model interface for MMM backends.

Defines common interface for Meridian and PyMC models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
import re


@dataclass
class FitResult:
    """Container for model fitting results."""
    backend: str
    posteriors: Any  # Backend-specific posterior samples
    diagnostics: Dict[str, Any]
    meta: Dict[str, Any]
    fit_time: float
    
    def __post_init__(self):
        """Validate fit result."""
        # Get valid backends from config or use defaults
        valid_backends = ['meridian', 'pymc']  # These are the supported backends
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Supported: {valid_backends}")


def assemble_features(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Assemble features for MMM training with strict leakage prevention.
    
    This is the ONLY code path used by both backends to ensure consistency.
    
    Args:
        df: Input DataFrame with all engineered features
        config: Configuration object
        
    Returns:
        Tuple of (X, y, meta) where:
        - X: Feature matrix (float32, deterministic order)
        - y: Target variable
        - meta: Metadata with feature_names, channels, dates, target_name
    """
    logger = logging.getLogger(__name__)
    
    # =============================================================================
    # 1. TARGET VARIABLE EXTRACTION
    # =============================================================================
    target_col = getattr(config.data, 'revenue_col', 'revenue')
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    y = df[target_col].copy()
    logger.info(f"Target variable: {target_col} (n_observations={len(y)})")
    
    # =============================================================================
    # 2. FEATURE SELECTION WITH STRICT LEAKAGE GUARDS
    # =============================================================================
    included_features = []
    excluded_features = []
    
    # Media features: *_adstocked_saturated only
    media_pattern = r'.*_adstocked_saturated$'
    media_features = [col for col in df.columns if re.match(media_pattern, col)]
    included_features.extend(media_features)
    logger.info(f"Media features: {len(media_features)} columns")
    
    # Seasonality features: fourier_*, dow_*, holiday_*, trend (if owned by seasonality)
    seasonality_patterns = [r'^fourier_', r'^dow_', r'^holiday_', r'^trend$']
    seasonality_features = []
    for pattern in seasonality_patterns:
        matches = [col for col in df.columns if re.match(pattern, col)]
        seasonality_features.extend(matches)
    included_features.extend(seasonality_features)
    logger.info(f"Seasonality features: {len(seasonality_features)} columns")
    
    # Baseline features: macro_*, external_*
    baseline_patterns = [r'^macro_', r'^external_']
    baseline_features = []
    for pattern in baseline_patterns:
        matches = [col for col in df.columns if re.match(pattern, col)]
        baseline_features.extend(matches)
    included_features.extend(baseline_features)
    logger.info(f"Baseline features: {len(baseline_features)} columns")
    
    # Custom features: promo_* only if not target-derived
    custom_features = []
    promo_features = [col for col in df.columns if col.startswith('promo_')]
    
    # Check for target-derived promo features
    target_derived_patterns = [r'promo_.*revenue.*', r'promo_.*sales.*', r'promo_.*purchases.*']
    target_derived_promos = []
    for pattern in target_derived_patterns:
        matches = [col for col in promo_features if re.search(pattern, col, re.IGNORECASE)]
        target_derived_promos.extend(matches)
    
    # Handle target-derived features
    allow_target_derived = getattr(config.model, 'allow_target_derived', False)
    if target_derived_promos:
        if not allow_target_derived:
            logger.error(f"Target-derived promo features detected: {target_derived_promos}")
            logger.error("Set config.model.allow_target_derived=True to proceed")
            raise ValueError("Target-derived features found - potential data leakage")
        else:
            logger.warning(f"Including target-derived features: {target_derived_promos}")
    
    # Include non-target-derived promo features
    safe_promo_features = [col for col in promo_features if col not in target_derived_promos]
    custom_features.extend(safe_promo_features)
    included_features.extend(custom_features)
    logger.info(f"Custom features: {len(custom_features)} columns (excluded {len(target_derived_promos)} target-derived)")
    
    # =============================================================================
    # 3. SCHEMA VALIDATION BEFORE FIT
    # =============================================================================
    
    # Verify channel mapping consistency
    channel_map = getattr(config.data, 'channel_map', {})
    if channel_map:
        missing_channels = []
        for channel_name, spend_col in channel_map.items():
            expected_col = f"{spend_col}_adstocked_saturated"
            if expected_col not in media_features:
                missing_channels.append((channel_name, expected_col))
        
        if missing_channels:
            error_msg = "Missing adstocked_saturated columns for channels:\n"
            for channel, col in missing_channels:
                error_msg += f"  - {channel}: expected '{col}'\n"
            error_msg += "Available media features: " + ", ".join(media_features)
            logger.error(error_msg)
            raise ValueError("Channel mapping validation failed")
    
    # =============================================================================
    # 4. FEATURE MATRIX CONSTRUCTION
    # =============================================================================
    
    # Remove duplicates and maintain order
    included_features = list(dict.fromkeys(included_features))  # Preserves order
    
    # Exclude target and metadata columns
    metadata_cols = [target_col, 'date', 'time', 'DATE_DAY', 'ORDINAL_DATE']
    included_features = [col for col in included_features if col not in metadata_cols]
    
    # Extract feature matrix
    missing_features = [col for col in included_features if col not in df.columns]
    if missing_features:
        logger.error(f"Missing features in DataFrame: {missing_features}")
        raise ValueError("Feature validation failed")
    
    X = df[included_features].copy()
    
    # =============================================================================
    # 5. DATA QUALITY ENFORCEMENT
    # =============================================================================
    
    # Check for NaNs
    nan_counts = X.isnull().sum()
    nan_features = nan_counts[nan_counts > 0]
    if len(nan_features) > 0:
        logger.error("NaN values found in features:")
        for feat, count in nan_features.head(10).items():
            logger.error(f"  - {feat}: {count} NaNs")
        raise ValueError("NaN values detected in feature matrix")
    
    # Check for NaNs in target
    if y.isnull().sum() > 0:
        logger.error(f"NaN values in target variable: {y.isnull().sum()}")
        raise ValueError("NaN values detected in target variable")
    
    # =============================================================================
    # 6. DETERMINISTIC DTYPE & ORDER
    # =============================================================================
    
    # Cast to float32 for memory efficiency
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Freeze column order
    feature_names = X.columns.tolist()
    
    # =============================================================================
    # 7. QUALITY WARNINGS (Nice-to-have)
    # =============================================================================
    
    # Check for zero variance features
    variances = X.var()
    zero_var_features = variances[variances < 1e-8].index.tolist()
    if zero_var_features:
        logger.warning(f"Near-zero variance features: {zero_var_features}")
    
    # Check for perfect collinearity (using correlation matrix)
    if len(X.columns) > 1:
        try:
            corr_matrix = X.corr().abs()
            # Set diagonal to 0 to ignore self-correlation
            np.fill_diagonal(corr_matrix.values, 0)
            
            # Get correlation threshold from config
            features_config = config.get('features', {})
            baseline_config = features_config.get('baseline', {})
            quality_control = baseline_config.get('quality_control', {})
            max_correlation_threshold = quality_control.get('max_correlation_threshold', 0.99)
            
            high_corr = np.where(corr_matrix > max_correlation_threshold)
            if len(high_corr[0]) > 0:
                corr_pairs = [(X.columns[i], X.columns[j]) for i, j in zip(high_corr[0], high_corr[1])]
                logger.warning(f"High correlation pairs (>{max_correlation_threshold}): {corr_pairs[:5]}")  # Show first 5
        except Exception as e:
            logger.debug(f"Could not compute correlation matrix: {e}")
    
    # =============================================================================
    # 8. METADATA CONSTRUCTION
    # =============================================================================
    
    # Extract channel names from media features
    channels = []
    for media_col in media_features:
        # Remove _adstocked_saturated suffix to get channel name
        channel_name = media_col.replace('_adstocked_saturated', '')
        channels.append(channel_name)
    
    # Extract dates if available
    date_cols = [col for col in df.columns if col.lower() in ['date', 'time', 'date_day']]
    dates = df[date_cols[0]].copy() if date_cols else None
    
    meta = {
        'n_observations': len(X),
        'feature_names': feature_names,  # Frozen order
        'channels': channels,
        'target_name': target_col,
        'dates': dates,
        'media_features': media_features,
        'seasonality_features': seasonality_features,
        'baseline_features': baseline_features,
        'custom_features': custom_features
    }
    
    logger.info(f"Assembled features: {len(feature_names)} total")
    logger.info(f"  - Media: {len(media_features)}")
    logger.info(f"  - Seasonality: {len(seasonality_features)}")
    logger.info(f"  - Baseline: {len(baseline_features)}")
    logger.info(f"  - Custom: {len(custom_features)}")
    
    return X, y, meta


class BaseMMM(ABC):
    """Base class for MMM model implementations."""
    
    def __init__(self, config):
        """Initialize model with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_fitted = False
        self.fit_result: Optional[FitResult] = None
        self.feature_names_: Optional[List[str]] = None  # Frozen feature order
    
    def fit_from_dataframe(self, df: pd.DataFrame) -> FitResult:
        """
        Fit model from engineered features DataFrame.
        
        This is the primary interface - uses assemble_features for consistency.
        
        Args:
            df: DataFrame with all engineered features
            
        Returns:
            FitResult: Model fitting results
        """
        # Use the centralized feature assembly
        X, y, meta = assemble_features(df, self.config)
        
        # Store feature order for consistent predictions
        self.feature_names_ = meta['feature_names']
        
        # Call backend-specific fit
        return self.fit(X, y, meta)
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align features to training order and types.
        
        Args:
            X: Input feature matrix
            
        Returns:
            pd.DataFrame: Aligned feature matrix
        """
        if self.feature_names_ is None:
            raise ValueError("Model not fitted - feature order not available")
        
        # Reorder columns to match training, fill missing with 0
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0.0)
        
        # Cast to float32 for consistency
        X_aligned = X_aligned.astype(np.float32)
        
        return X_aligned
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, meta: Dict[str, Any]) -> FitResult:
        """
        Fit the MMM model.
        
        Args:
            X: Feature matrix (adstocked and saturated media variables + controls)
            y: Target variable (revenue/conversions)
            meta: Metadata including channel names, dates, etc.
            
        Returns:
            FitResult: Model fitting results
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from fitted model.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        pass
    
    @abstractmethod
    def get_media_contribution(self) -> pd.DataFrame:
        """
        Calculate media contribution for each channel.
        
        Returns:
            pd.DataFrame: Media contribution by channel and time period
        """
        pass
    
    @abstractmethod
    def get_roi_curves(self, budget_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate ROI curves for budget optimization.
        
        Args:
            budget_range: Range of budget values to evaluate
            
        Returns:
            Dict: ROI curves by channel
        """
        pass
    
    @abstractmethod
    def save_artifacts(self, run_dir: str) -> None:
        """
        Save model artifacts to directory.
        
        Args:
            run_dir: Directory to save artifacts
        """
        pass
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting diagnostics")
        
        return self.fit_result.diagnostics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        diagnostics = self.get_diagnostics()
        
        summary = {
            'backend': self.fit_result.backend,
            'fit_time': self.fit_result.fit_time,
            'n_parameters': diagnostics.get('n_parameters', 0),
            'convergence': {
                'max_rhat': diagnostics.get('max_rhat', None),
                'min_ess': diagnostics.get('min_ess', None),
                'converged': diagnostics.get('converged', False)
            }
        }
        
        return summary


def validate_model_inputs(X: pd.DataFrame, y: pd.Series, meta: Dict[str, Any]) -> None:
    """
    Validate model inputs.
    
    Args:
        X: Feature matrix
        y: Target variable
        meta: Metadata
    """
    if len(X) != len(y):
        raise ValueError(f"Feature matrix and target have different lengths: {len(X)} vs {len(y)}")
    
    if X.isnull().any().any():
        raise ValueError("Feature matrix contains null values")
    
    if y.isnull().any():
        raise ValueError("Target variable contains null values")
    
    if not isinstance(meta, dict):
        raise ValueError("Meta must be a dictionary")
    
    required_meta_keys = ['channels', 'date_col']
    missing_keys = [key for key in required_meta_keys if key not in meta]
    if missing_keys:
        raise ValueError(f"Missing required meta keys: {missing_keys}")


def prepare_model_data(df: pd.DataFrame, config) -> tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Prepare data for model fitting.
    
    Args:
        df: Input dataframe with features and target
        config: Configuration object
        
    Returns:
        tuple: (X, y, meta)
    """
    # Extract target variable
    y = df[config.data.revenue_col]
    
    # Extract features (spend channels + seasonality + controls)
    feature_cols = []
    
    # Add media channels
    for channel in config.data.channel_map.keys():
        feature_cols.append(f"{channel}_adstocked_saturated")
    
    # Add seasonality features
    seasonality_cols = [col for col in df.columns if any(pattern in col for pattern in 
                       ['sin_', 'cos_', 'trend', 'holiday', 'is_'])]
    feature_cols.extend(seasonality_cols)
    
    # Filter to existing columns
    existing_cols = [col for col in feature_cols if col in df.columns]
    X = df[existing_cols]
    
    # Create metadata
    meta = {
        'channels': list(config.data.channel_map.keys()),
        'date_col': config.data.date_col,
        'dates': df[config.data.date_col] if config.data.date_col in df.columns else None,
        'feature_names': existing_cols,
        'n_observations': len(df)
    }
    
    return X, y, meta


if __name__ == "__main__":
    # Test the base model interface
    import pandas as pd
    import numpy as np
    
    # Create sample data
    n_obs = 100
    X = pd.DataFrame({
        'google_search_adstocked_saturated': np.random.randn(n_obs),
        'meta_facebook_adstocked_saturated': np.random.randn(n_obs),
        'trend': np.linspace(0, 1, n_obs),
        'sin_annual_1': np.sin(2 * np.pi * np.arange(n_obs) / 52)
    })
    
    y = pd.Series(np.random.lognormal(10, 1, n_obs))
    
    meta = {
        'channels': ['google_search', 'meta_facebook'],
        'date_col': 'date',
        'feature_names': X.columns.tolist()
    }
    
    # Test validation
    try:
        validate_model_inputs(X, y, meta)
        print("Model input validation passed")
    except Exception as e:
        print(f"Validation error: {e}")
    
    print("Base model module test completed")
