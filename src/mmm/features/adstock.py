"""
Adstock transformation module.

Implements geometric and Weibull adstock transformations for media carryover effects.
Features: Parameter validation, vectorized operations, flexible normalization options.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Literal
import logging
from scipy import signal
from scipy.stats import weibull_min


def _validate_parameters(adstock_type: str, decay: float = None, shape: float = None, 
                        scale: float = None, lambda_param: float = None, max_lag: int = None,
                        hill_alpha: float = None, hill_gamma: float = None) -> None:
    """
    Validate adstock parameters with helpful error messages.
    
    Args:
        adstock_type: Type of adstock transformation
        decay: Geometric decay parameter
        shape: Weibull shape parameter
        scale: Weibull scale parameter
        lambda_param: Alternative geometric decay parameter
        max_lag: Maximum lag periods
        hill_alpha: Hill alpha parameter
        hill_gamma: Hill gamma parameter
        
    Raises:
        ValueError: If parameters are invalid
    """
    if adstock_type == "geometric":
        effective_decay = decay if decay is not None else lambda_param
        if effective_decay is None:
            raise ValueError("Geometric adstock requires either 'decay' or 'lambda_param'")
        if not (0 <= effective_decay < 1):
            raise ValueError(f"Geometric decay must be 0 ≤ decay < 1, got {effective_decay}. "
                           "Values ≥ 1 cause divergence.")
    
    elif adstock_type == "weibull":
        if shape is None or scale is None:
            raise ValueError("Weibull adstock requires both 'shape' and 'scale' parameters")
        if shape <= 0:
            raise ValueError(f"Weibull shape must be > 0, got {shape}")
        if scale <= 0:
            raise ValueError(f"Weibull scale must be > 0, got {scale}")
        if max_lag is not None and max_lag < 0:
            raise ValueError(f"max_lag must be ≥ 0, got {max_lag}")
    
    # Hill transformation parameters
    if hill_alpha is not None and hill_alpha <= 0:
        raise ValueError(f"Hill alpha must be > 0, got {hill_alpha}")
    if hill_gamma is not None and hill_gamma <= 0:
        raise ValueError(f"Hill gamma must be > 0, got {hill_gamma}")


def _handle_missing_values(spend: Union[pd.Series, np.ndarray], 
                          fill_method: str = "zero") -> np.ndarray:
    """
    Handle NaN values in spend data.
    
    Args:
        spend: Media spend series
        fill_method: Method to handle NaNs ("zero", "raise")
        
    Returns:
        np.ndarray: Cleaned spend array
        
    Raises:
        ValueError: If NaNs found and fill_method is "raise"
    """
    spend_array = np.array(spend, dtype=float)
    
    if np.any(np.isnan(spend_array)):
        if fill_method == "raise":
            raise ValueError("NaN values found in spend data. Clean data before applying adstock.")
        elif fill_method == "zero":
            spend_array = np.nan_to_num(spend_array, nan=0.0)
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")
    
    return spend_array


def _normalize_adstock(adstocked: np.ndarray, original: np.ndarray, 
                      method: Literal["sum", "none", "mean", "max"] = "sum") -> np.ndarray:
    """
    Normalize adstocked values according to specified method.
    
    Args:
        adstocked: Adstocked values
        original: Original spend values
        method: Normalization method
            - "sum": Scale to match sum of original (preserves total spend)
            - "none": No normalization (pure carryover increases mass)
            - "mean": Scale to match mean of original 
            - "max": Scale to match max of original (peak-preserving)
        
    Returns:
        np.ndarray: Normalized adstocked values
    """
    if method == "none":
        return adstocked
    
    # Guard against zero division
    if np.sum(adstocked) == 0:
        logging.warning("Adstocked series sums to zero. Returning original values.")
        return original.copy()
    
    if method == "sum":
        return adstocked * (np.sum(original) / np.sum(adstocked))
    elif method == "mean":
        return adstocked * (np.mean(original) / np.mean(adstocked))
    elif method == "max":
        orig_max = np.max(original)
        if orig_max == 0:
            return adstocked
        return adstocked * (orig_max / np.max(adstocked))
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def apply_adstock(
    spend: Union[pd.Series, np.ndarray], 
    decay: float = None,
    adstock_type: str = "geometric",
    shape: float = None,
    scale: float = None,
    lambda_param: float = None,
    max_lag: int = 8,
    normalizing: Literal["sum", "none", "mean", "max"] = "sum",
    weibull_mode: Literal["pdf", "cdf_diff"] = "pdf",
    fill_na_method: str = "zero"
) -> np.ndarray:
    """
    Apply adstock (media carryover effects) to a media spend time series.
    
    Supports multiple adstock transformation types:
    - Geometric: Exponential decay with tunable carryover parameter
    - Weibull: PDF-based or CDF-based saturation curves for richer dynamics
    
    Args:
        spend: Media spend time series (pandas Series or numpy array)
        decay: Geometric decay parameter (0 ≤ decay < 1). Alternative to lambda_param.
        adstock_type: Type of adstock ("geometric", "weibull")
        shape: Weibull shape parameter (> 0, controls curve steepness)
        scale: Weibull scale parameter (> 0, controls lag timing)
        lambda_param: Alternative geometric decay parameter (legacy)
        max_lag: Maximum carryover periods for Weibull (default: 8)
        normalizing: Post-transformation normalization method:
            - "sum": Scale to preserve total spend mass (default)
            - "none": Pure carryover (increases total mass)
            - "mean": Scale to preserve mean spend level
            - "max": Scale to preserve peak spend level
        weibull_mode: Weibull kernel semantics:
            - "pdf": Probability density function (continuous decay)
            - "cdf_diff": CDF differences (step-wise decay)
        fill_na_method: NaN handling ("zero" to fill, "raise" to error)
        
    Returns:
        np.ndarray: Adstocked media spend values
        
    Raises:
        ValueError: If parameters are invalid or data contains issues
        
    Examples:
        # Standard geometric adstock with 30% carryover
        adstocked = apply_adstock(spend, decay=0.3, adstock_type="geometric")
        
        # Weibull adstock with custom saturation curve
        adstocked = apply_adstock(spend, adstock_type="weibull", shape=2.0, scale=3.5)
        
        # Pure carryover without normalization
        adstocked = apply_adstock(spend, decay=0.5, normalizing="none")
    """
    # Parameter validation
    _validate_parameters(
        adstock_type=adstock_type,
        decay=decay, 
        shape=shape,
        scale=scale,
        lambda_param=lambda_param,
        max_lag=max_lag
    )
    
    # Handle input data cleaning
    spend_array = _handle_missing_values(spend, fill_method=fill_na_method)
    original_spend = spend_array.copy()
    
    # Log transformation parameters
    logging.info(f"Applying {adstock_type} adstock transformation")
    if adstock_type == "geometric":
        effective_decay = decay if decay is not None else lambda_param
        logging.info(f"  Geometric decay: {effective_decay}")
    elif adstock_type == "weibull":
        logging.info(f"  Weibull shape: {shape}, scale: {scale}, max_lag: {max_lag}, mode: {weibull_mode}")
    logging.info(f"  Normalization: {normalizing}")
    
    # Apply transformation
    if adstock_type == "geometric":
        adstocked = geometric_adstock(spend_array, decay=decay, lambda_param=lambda_param)
    elif adstock_type == "weibull":
        adstocked = weibull_adstock(spend_array, shape=shape, scale=scale, 
                                   max_lag=max_lag, mode=weibull_mode)
    else:
        raise ValueError(f"Unknown adstock_type: {adstock_type}")
    
    # Apply normalization
    normalized = _normalize_adstock(adstocked, original_spend, method=normalizing)
    
    logging.info(f"Adstock transformation complete. "
                f"Original sum: {np.sum(original_spend):.2f}, "
                f"Final sum: {np.sum(normalized):.2f}")
    
    return normalized


def apply_adstock_hill(
    spend: Union[pd.Series, np.ndarray],
    adstock_params: Dict[str, Any],
    max_lag: int = 8
) -> np.ndarray:
    """
    Apply Hill-transformed adstock (advanced convolution function).
    
    Args:
        spend: Media spend series
        adstock_params: Dictionary with adstock parameters
        max_lag: Maximum lag periods
        
    Returns:
        np.ndarray: Hill-adstocked spend values
    """
    # First apply standard adstock
    adstocked = apply_adstock(
        spend=spend,
        decay=adstock_params.get('decay'),
        adstock_type=adstock_params.get('type', 'geometric'),
        shape=adstock_params.get('shape'),
        scale=adstock_params.get('scale'),
        lambda_param=adstock_params.get('lambda'),
        max_lag=max_lag
    )
    
    # Then apply Hill transformation for saturation
    alpha = adstock_params.get('hill_alpha', 0.5)
    gamma = adstock_params.get('hill_gamma', 1.0)
    
    return adstocked**alpha / (gamma**alpha + adstocked**alpha)


def geometric_adstock(spend: Union[pd.Series, np.ndarray], 
                     decay: float = None, lambda_param: float = None) -> np.ndarray:
    """
    Apply geometric adstock transformation using vectorized operations.
    
    Uses scipy.signal.lfilter for efficient computation of the recursive formula:
    adstocked[t] = spend[t] + decay * adstocked[t-1]
    
    Args:
        spend: Media spend series
        decay: Decay parameter (0 ≤ decay < 1, higher means longer carryover)
        lambda_param: Alternative decay parameter name (for compatibility)
        
    Returns:
        np.ndarray: Adstocked spend values
        
    Raises:
        ValueError: If decay parameter is invalid
    """
    # Parameter handling
    effective_decay = decay if decay is not None else lambda_param
    if effective_decay is None:
        raise ValueError("geometric_adstock requires either 'decay' or 'lambda_param'")
    
    spend_array = np.array(spend, dtype=float)
    
    # Edge case: zero decay means no carryover
    if effective_decay == 0:
        return spend_array
    
    # Use scipy.signal.lfilter for vectorized geometric adstock
    # The recursive relation: y[n] = x[n] + decay * y[n-1]
    # In filter terms: a = [1, -decay], b = [1]
    a_coeffs = [1, -effective_decay]  # Denominator coefficients
    b_coeffs = [1]                    # Numerator coefficients
    
    # lfilter applies: sum(b_coeffs * x[n-m]) = sum(a_coeffs * y[n-m])
    # For our case: y[n] - decay*y[n-1] = x[n] -> y[n] = x[n] + decay*y[n-1]
    adstocked = signal.lfilter(b_coeffs, a_coeffs, spend_array)
    
    return adstocked


def weibull_adstock(spend: Union[pd.Series, np.ndarray], shape: float, scale: float, 
                   max_lag: int = 8, mode: Literal["pdf", "cdf_diff"] = "pdf") -> np.ndarray:
    """
    Apply Weibull adstock transformation with vectorized convolution.
    
    Creates a Weibull-based carryover kernel for richer adstock dynamics beyond 
    simple exponential decay. Supports both PDF and CDF-difference semantics.
    
    Args:
        spend: Media spend series
        shape: Weibull shape parameter (> 0, controls curve steepness)
               - shape < 1: Decreasing hazard (front-loaded carryover)
               - shape = 1: Exponential decay (equivalent to geometric)
               - shape > 1: Increasing then decreasing hazard (bell-shaped)
        scale: Weibull scale parameter (> 0, controls carryover timing)
               Higher scale = longer carryover duration
        max_lag: Maximum lag periods to consider (default: 8)
        mode: Kernel construction method:
              - "pdf": Use Weibull probability density function (continuous)
              - "cdf_diff": Use CDF differences (step-wise decay)
        
    Returns:
        np.ndarray: Adstocked spend values
        
    Notes:
        Weibull PDF: f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
        where k=shape, λ=scale
    """
    spend_array = np.array(spend, dtype=float)
    
    # Generate lag periods (1 to max_lag)
    lags = np.arange(1, max_lag + 1, dtype=float)
    
    if mode == "pdf":
        # Use Weibull PDF for smooth carryover weights
        weights = weibull_min.pdf(lags, c=shape, scale=scale)
    elif mode == "cdf_diff":
        # Use CDF differences for step-wise carryover
        cdf_vals = weibull_min.cdf(lags, c=shape, scale=scale)
        weights = np.diff(np.concatenate([[0], cdf_vals]))
    else:
        raise ValueError(f"Unknown Weibull mode: {mode}")
    
    # Normalize weights to sum to 1 (preserve mass)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        logging.warning("Weibull weights sum to zero. Using uniform weights.")
        weights = np.ones_like(weights) / len(weights)
    
    # Create full convolution kernel [current_effect, lag1_weight, lag2_weight, ...]
    # Current period has weight 1, lags have Weibull weights
    kernel = np.concatenate([[1.0], weights])
    
    # Apply convolution using numpy's efficient implementation
    # mode='same' keeps original length, 'valid' would truncate
    adstocked = np.convolve(spend_array, kernel, mode='same')
    
    return adstocked


def get_adstock_params(config, channel: str) -> Dict[str, Any]:
    """
    Get adstock parameters for a specific channel.
    
    Args:
        config: Configuration object
        channel: Channel name
        
    Returns:
        Dict: Adstock parameters
    """
    adstock_config = config.features.adstock
    
    # Default parameters
    params = {
        'type': adstock_config.get('default_type', 'geometric'),
        'decay': adstock_config.get('default_decay', 0.9)
    }
    
    # Platform-specific overrides from enhanced config
    platform_overrides = adstock_config.get('platform_overrides', {})
    if channel in platform_overrides:
        override_params = platform_overrides[channel]
        params.update(override_params)
    
    # Advanced parameters
    advanced_params = adstock_config.get('advanced_params', {})
    params.update({
        'max_lag': advanced_params.get('max_lag', 8),
        'normalizing': advanced_params.get('normalizing', True),
        'mode': advanced_params.get('mode', 'multiplicative'),
        'convolve_func': advanced_params.get('convolve_func', 'adstock_hill')
    })
    
    return params


def apply_platform_adstock(spend_data: pd.DataFrame, config) -> pd.DataFrame:
    """
    Apply platform-specific adstock transformations to all channels.
    
    Args:
        spend_data: DataFrame with spend columns
        config: Configuration object
        
    Returns:
        pd.DataFrame: Data with adstocked columns added
    """
    logger = logging.getLogger(__name__)
    result_df = spend_data.copy()
    
    # Get channel mapping
    channel_map = config.data.channel_map
    
    for channel_name, spend_col in channel_map.items():
        if spend_col in spend_data.columns:
            # Get platform-specific parameters
            params = get_adstock_params(config, channel_name)
            
            logger.info(f"Applying {params['type']} adstock to {channel_name} (decay={params.get('decay', 'N/A')})")
            
            # Apply adstock transformation
            if params['convolve_func'] == 'adstock_hill':
                adstocked_values = apply_adstock_hill(
                    spend_data[spend_col], 
                    params, 
                    params['max_lag']
                )
            else:
                adstocked_values = apply_adstock(
                    spend=spend_data[spend_col],
                    decay=params.get('decay'),
                    adstock_type=params.get('type', 'geometric'),
                    shape=params.get('shape'),
                    scale=params.get('scale'),
                    lambda_param=params.get('lambda'),
                    max_lag=params.get('max_lag', 8)
                )
            
            # Add adstocked column
            result_df[f"{channel_name}_adstocked"] = adstocked_values
            
            # Normalize if specified
            if params.get('normalizing', True):
                result_df[f"{channel_name}_adstocked"] = (
                    result_df[f"{channel_name}_adstocked"] / 
                    result_df[f"{channel_name}_adstocked"].sum()
                ) * result_df[spend_col].sum()
    
    logger.info(f"Applied adstock transformations to {len(channel_map)} channels")
    return result_df


def process_all_features(data_df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Apply all feature engineering including digital metrics and adstock.
    
    Args:
        data_df: Input dataframe with raw data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe
    """
    logger = logging.getLogger(__name__)
    
    # Apply platform-specific adstock transformations
    result_df = apply_platform_adstock(data_df, config)
    
    # Add feature engineering metadata
    result_df.attrs['feature_engineering'] = {
        'adstock_applied': True,
        'timestamp': pd.Timestamp.now().isoformat(),
        'platform_count': len(config.data.channel_map),
        'config_version': config.project.get('name', 'unknown')
    }
    
    logger.info("Completed all feature engineering")
    return result_df


if __name__ == "__main__":
    # Test adstock functions
    import matplotlib.pyplot as plt
    
    # Create sample spend data
    spend = np.array([100, 0, 0, 0, 0, 200, 0, 0, 0, 0])
    
    # Apply different decay rates
    decays = [0.3, 0.6, 0.9]
    
    for decay in decays:
        adstocked = geometric_adstock(spend, decay)
        print(f"Decay {decay}: {adstocked}")
    
    print("Adstock module test completed")
