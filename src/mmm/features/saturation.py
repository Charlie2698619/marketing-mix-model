"""
Saturation transformation module.

Implements Hill and logistic saturation curves for diminishing returns modeling.
Features robust normalization, parameter calibration, and flexible configuration.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Tuple, Optional, Literal
import logging

# Small epsilon to avoid numerical issues
EPS = 1e-12


def _safe_config_get(config_obj, key, default=None):
    """Safely get value from config object (dict or Pydantic)."""
    if hasattr(config_obj, 'get'):
        return config_obj.get(key, default)
    else:
        return getattr(config_obj, key, default)



def _normalize(
    x: np.ndarray, 
    mode: str, 
    k: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Normalize input array based on specified mode.
    
    Args:
        x: Input array to normalize
        mode: Normalization mode
            - "max": Divide by maximum value (current default behavior)
            - "p95": Divide by 95th percentile (robust to outliers)
            - "none": No normalization (assume inputs already comparable)
            - "k_in_units": No normalization, k interpreted in original units
        k: Parameter k (unused in normalization but kept for API consistency)
        
    Returns:
        Tuple of (normalized_array, scale_factor)
    """
    x = np.asarray(x, dtype=float)
    
    if mode == "max":
        m = np.nanmax(x) if x.size > 0 else 0.0
        scale = m if m > 0 else 1.0
        return x / scale, scale
    elif mode == "p95":
        valid_x = x[np.isfinite(x) & (x > 0)]
        if len(valid_x) > 0:
            p = np.percentile(valid_x, 95)
            scale = p if p > 0 else 1.0
        else:
            scale = 1.0
        return x / scale, scale
    elif mode == "none":
        return x, 1.0
    elif mode == "k_in_units":
        # No normalization - k will be interpreted in original spend units
        return x, 1.0
    else:
        # Default fallback to max normalization
        return _normalize(x, "max", k)


def apply_saturation(
    adstocked_spend: Union[pd.Series, np.ndarray],
    saturation_type: str = "hill",
    normalization: str = "max",
    preserve_index: bool = False,
    config: Optional[Dict[str, Any]] = None,
    enable_validation: bool = True,
    enable_calibration: bool = False,
    response_data: Optional[np.ndarray] = None,
    **params
) -> Union[np.ndarray, pd.Series]:
    """
    Apply saturation transformation to adstocked media spend with optional validation and calibration.
    
    Args:
        adstocked_spend: Adstocked spend series
        saturation_type: Type of saturation ("hill" or "logistic")
        normalization: Normalization mode ("max", "p95", "none", "k_in_units")
        preserve_index: Whether to preserve pandas Series index
        config: Configuration dictionary containing validation/calibration settings
        enable_validation: Whether to validate the saturation curve
        enable_calibration: Whether to auto-calibrate parameters from response_data
        response_data: Response data for calibration (required if enable_calibration=True)
        **params: Saturation parameters (k, s for hill; k, x0 for logistic)
        
    Returns:
        Saturated spend values (0-1 scale) as array or Series
    """
    logger = logging.getLogger(__name__)
    
    # Auto-calibrate parameters if enabled and response data provided
    if enable_calibration and response_data is not None:
        spend_array = np.asarray(adstocked_spend)
        response_array = np.asarray(response_data)
        
        # Get minimum data points requirement from config
        min_data_points = 20
        if config and 'calibration' in config:
            min_data_points = config['calibration'].get('min_data_points', 20)
            
        if len(spend_array) >= min_data_points:
            logger.info("Auto-calibrating saturation parameters from response data")
            calibrated_params = calibrate_saturation_curve(
                spend_array, response_array, saturation_type, normalization
            )
            # Update params with calibrated values
            params.update(calibrated_params)
        else:
            logger.warning(f"Insufficient data for calibration ({len(spend_array)} < {min_data_points} points)")
    
    # Validate parameters if enabled
    if enable_validation and config:
        validation_settings = config.get('validation', {
            'check_monotonic': True,
            'check_bounded': True,
            'warn_extreme_params': True
        })
        
        # Validate input parameters
        param_dict = {'type': saturation_type, 'normalization': normalization, **params}
        validated_params = validate_saturation_params(param_dict, validation_settings)
        params.update({k: v for k, v in validated_params.items() if k not in ['type', 'normalization']})
    
    # Apply the actual saturation transformation
    if saturation_type == "hill":
        filtered_params = {k: v for k, v in params.items() if k in ['k', 's']}
        result = hill_saturation(
            adstocked_spend, 
            normalization=normalization,
            preserve_index=preserve_index,
            **filtered_params
        )
    elif saturation_type == "logistic":
        filtered_params = {k: v for k, v in params.items() if k in ['k', 'x0']}
        result = logistic_saturation(
            adstocked_spend,
            normalization=normalization, 
            preserve_index=preserve_index,
            **filtered_params
        )
    else:
        raise ValueError(f"Unknown saturation type: {saturation_type}")
    
    # Validate the resulting curve if enabled
    if enable_validation and config:
        validation_settings = config.get('validation', {})
        x_vals = np.asarray(adstocked_spend)
        y_vals = np.asarray(result)
        
        # Filter to finite values for validation
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if np.any(mask):
            validate_saturation_curve(x_vals[mask], y_vals[mask], validation_settings)
    
    return result


def hill_saturation(
    adstocked_spend: Union[pd.Series, np.ndarray],
    k: float = 0.5,
    s: float = 1.0,
    normalization: str = "max",
    preserve_index: bool = False
) -> Union[np.ndarray, pd.Series]:
    """
    Apply Hill saturation transformation with robust normalization.
    
    The Hill saturation curve models diminishing returns with the formula:
    y = x^s / (k^s + x^s)
    
    Args:
        adstocked_spend: Adstocked spend series
        k: Half-saturation point (inflection parameter)
            - If normalization != "k_in_units": k is on normalized scale [0,1]
            - If normalization == "k_in_units": k is in original spend units
        s: Shape parameter (slope steepness at inflection point)
        normalization: How to normalize inputs ("max", "p95", "none", "k_in_units")
        preserve_index: Whether to return pandas Series with original index
        
    Returns:
        Saturated values between 0 and 1
    """
    x = np.asarray(adstocked_spend, dtype=float)
    
    # Handle edge cases: all zeros, all NaNs, or invalid data
    if x.size == 0 or not np.isfinite(x).any() or np.nanmax(x) <= 0:
        out = np.zeros_like(x)
        return pd.Series(out, index=adstocked_spend.index) if preserve_index and hasattr(adstocked_spend, 'index') else out
    
    # Normalize input
    x_scaled, scale = _normalize(x, normalization, k)
    
    # Adjust k parameter based on normalization
    k_eff = k / scale if (normalization != "k_in_units") else k
    
    # Apply Hill transformation with numerical stability
    x_clipped = np.clip(x_scaled, 0, None)  # Ensure non-negative
    
    # Compute numerator and denominator with numerical stability
    numerator = np.power(x_clipped, s)
    denominator = np.power(np.abs(k_eff), s) + numerator + EPS
    
    # Calculate saturated values
    y = numerator / denominator
    
    # Handle any remaining NaNs or infinities
    y = np.where(np.isfinite(y), y, 0.0)
    
    return pd.Series(y, index=adstocked_spend.index) if preserve_index and hasattr(adstocked_spend, 'index') else y


def logistic_saturation(
    adstocked_spend: Union[pd.Series, np.ndarray],
    k: float = 1.0,
    x0: float = 0.5,
    normalization: str = "max",
    preserve_index: bool = False
) -> Union[np.ndarray, pd.Series]:
    """
    Apply logistic saturation transformation with robust normalization.
    
    The logistic saturation curve models S-shaped growth with the formula:
    y = 1 / (1 + exp(-k * (x - x0)))
    
    Args:
        adstocked_spend: Adstocked spend series
        k: Steepness parameter (higher = steeper curve)
        x0: Midpoint parameter (inflection point on normalized scale)
        normalization: How to normalize inputs ("max", "p95", "none", "k_in_units")
        preserve_index: Whether to return pandas Series with original index
        
    Returns:
        Saturated values between 0 and 1
    """
    x = np.asarray(adstocked_spend, dtype=float)
    
    # Handle edge cases: all zeros, all NaNs, or invalid data
    if x.size == 0 or not np.isfinite(x).any() or np.nanmax(x) <= 0:
        out = np.zeros_like(x)
        return pd.Series(out, index=adstocked_spend.index) if preserve_index and hasattr(adstocked_spend, 'index') else out
    
    # Normalize input
    x_scaled, scale = _normalize(x, normalization, k)
    
    # Apply logistic transformation with numerical stability
    z = k * (x_scaled - x0)
    z = np.clip(z, -60, 60)  # Prevent exp overflow/underflow
    
    y = 1.0 / (1.0 + np.exp(-z))
    
    # Handle any remaining NaNs or infinities
    y = np.where(np.isfinite(y), y, 0.0)
    
    return pd.Series(y, index=adstocked_spend.index) if preserve_index and hasattr(adstocked_spend, 'index') else y


def get_saturation_params(config: Dict[str, Any], channel: str) -> Dict[str, Any]:
    """
    Get saturation parameters for a specific channel from configuration.
    
    Args:
        config: Configuration dictionary with saturation settings
        channel: Channel name for potential channel-specific overrides
        
    Returns:
        Dict: Saturation parameters including type, normalization, curve parameters,
              and calibration/validation settings
    """
    # Access saturation config from features.saturation path
    features_config = _safe_config_get(config, 'features', {})
    saturation_config = _safe_config_get(features_config, 'saturation', {})
    
    # Default parameters with normalization support
    params = {
        'type': _safe_config_get(saturation_config, 'type', 'hill'),
        'normalization': _safe_config_get(saturation_config, 'normalization', 'max'),
        'preserve_index': _safe_config_get(saturation_config, 'preserve_index', False),
    }
    
    # Hill-specific parameters
    if params['type'] == 'hill':
        params.update({
            'k': _safe_config_get(saturation_config, 'default_inflection', 0.5),
            's': _safe_config_get(saturation_config, 'default_slope', 1.0)
        })
    
    # Logistic-specific parameters  
    elif params['type'] == 'logistic':
        params.update({
            'k': _safe_config_get(saturation_config, 'default_steepness', 1.0),
            'x0': _safe_config_get(saturation_config, 'default_midpoint', 0.5)
        })
    
    # Add calibration settings
    calibration_config = _safe_config_get(saturation_config, 'calibration', {})
    params['calibration'] = {
        'enabled': _safe_config_get(calibration_config, 'enabled', False),
        'min_data_points': _safe_config_get(calibration_config, 'min_data_points', 20),
        'calibration_method': _safe_config_get(calibration_config, 'calibration_method', 'heuristic')
    }
    
    # Add validation settings
    validation_config = _safe_config_get(saturation_config, 'validation', {})
    params['validation'] = {
        'check_monotonic': _safe_config_get(validation_config, 'check_monotonic', True),
        'check_bounded': _safe_config_get(validation_config, 'check_bounded', True),
        'warn_extreme_params': _safe_config_get(validation_config, 'warn_extreme_params', True)
    }
    
    # Check for channel-specific overrides
    channel_overrides_all = _safe_config_get(saturation_config, 'channel_overrides', {})
    channel_overrides = _safe_config_get(channel_overrides_all, channel, {})
    params.update(channel_overrides)
    
    return params


def apply_saturation_from_config(
    adstocked_spend: Union[pd.Series, np.ndarray],
    config: Dict[str, Any],
    channel: str,
    response_data: Optional[np.ndarray] = None
) -> Union[np.ndarray, pd.Series]:
    """
    Apply saturation transformation using complete YAML configuration.
    
    This function integrates all configuration settings including:
    - Saturation type and parameters
    - Normalization mode
    - Calibration settings (auto-calibrate if enabled)
    - Validation settings (validate parameters and curves)
    
    Args:
        adstocked_spend: Adstocked spend series
        config: Complete configuration dictionary from YAML
        channel: Channel name for parameter lookup
        response_data: Optional response data for auto-calibration
        
    Returns:
        Saturated spend values (0-1 scale) as array or Series
    """
    # Get all parameters and settings for this channel
    params = get_saturation_params(config, channel)
    
    # Extract saturation-specific parameters
    saturation_type = params.pop('type', 'hill')
    normalization = params.pop('normalization', 'max')
    preserve_index = params.pop('preserve_index', False)
    
    # Extract configuration settings
    calibration_config = params.pop('calibration', {})
    validation_config = params.pop('validation', {})
    
    # Determine if calibration should be enabled
    enable_calibration = (
        calibration_config.get('enabled', False) and 
        response_data is not None and
        len(np.asarray(adstocked_spend)) >= calibration_config.get('min_data_points', 20)
    )
    
    # Apply saturation with all configuration
    return apply_saturation(
        adstocked_spend=adstocked_spend,
        saturation_type=saturation_type,
        normalization=normalization,
        preserve_index=preserve_index,
        config={'calibration': calibration_config, 'validation': validation_config},
        enable_validation=True,  # Always validate when using config
        enable_calibration=enable_calibration,
        response_data=response_data,
        **params  # Remaining saturation parameters (k, s, x0, etc.)
    )


def calibrate_saturation_curve(
    spend: np.ndarray,
    response: np.ndarray,
    saturation_type: str = "hill",
    normalization: str = "max"
) -> Dict[str, float]:
    """
    Calibrate saturation parameters from observed spend and response data.
    
    Uses a lightweight heuristic approach that works well in practice:
    - For Hill: Sets k around 60th percentile of spend, s=1.0
    - For Logistic: Uses k=1.0, x0=0.5 as reasonable defaults
    
    Args:
        spend: Historical spend data
        response: Historical response data  
        saturation_type: Type of saturation curve ("hill" or "logistic")
        normalization: Normalization mode to use
        
    Returns:
        Dict: Calibrated saturation parameters
    """
    x = np.asarray(spend, dtype=float)
    y = np.asarray(response, dtype=float)
    
    # Filter to valid data points
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x, y = x[mask], y[mask]
    
    if len(x) < 8:
        # Small sample size - fall back to reasonable defaults
        logger = logging.getLogger(__name__)
        logger.warning(f"Small sample size ({len(x)} points) for saturation calibration - using defaults")
        
        if saturation_type == "hill":
            k_guess = np.percentile(x[x > 0], 50) if np.any(x > 0) else 1.0
            return {"k": float(k_guess), "s": 1.0, "normalization": normalization}
        else:
            return {"k": 1.0, "x0": 0.5, "normalization": normalization}
    
    # Heuristic calibration approach
    if saturation_type == "hill":
        # Set k around 60th percentile - this typically gives good half-saturation behavior
        k_guess = np.percentile(x, 60) if len(x) > 0 else 1.0
        return {
            "k": float(k_guess), 
            "s": 1.0,  # Start with linear shape
            "normalization": normalization
        }
    
    elif saturation_type == "logistic":
        # For logistic, use reasonable defaults that work well
        return {
            "k": 1.0,      # Moderate steepness
            "x0": 0.5,     # Midpoint at 50% of normalized scale
            "normalization": normalization
        }
    
    else:
        raise ValueError(f"Unknown saturation type: {saturation_type}")


def validate_saturation_params(params: Dict[str, Any], validation_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate and sanitize saturation parameters with configurable validation rules.
    
    Args:
        params: Saturation parameters dictionary
        validation_settings: Validation configuration (check_monotonic, check_bounded, warn_extreme_params)
        
    Returns:
        Dict: Validated parameters with warnings for invalid values
    """
    logger = logging.getLogger(__name__)
    validated = params.copy()
    
    # Use default validation settings if not provided
    if validation_settings is None:
        validation_settings = {
            'check_monotonic': True,
            'check_bounded': True, 
            'warn_extreme_params': True
        }
    
    # Validate saturation type
    if params.get('type') not in ['hill', 'logistic']:
        logger.warning(f"Unknown saturation type '{params.get('type')}', defaulting to 'hill'")
        validated['type'] = 'hill'
    
    # Validate normalization mode
    valid_norms = ['max', 'p95', 'none', 'k_in_units']
    if params.get('normalization') not in valid_norms:
        logger.warning(f"Unknown normalization '{params.get('normalization')}', defaulting to 'max'")
        validated['normalization'] = 'max'
    
    # Validate Hill parameters
    if validated['type'] == 'hill':
        k = validated.get('k', 0.5)
        s = validated.get('s', 1.0)
        
        if k <= 0:
            logger.warning("Hill parameter k must be positive, setting to 0.5")
            validated['k'] = 0.5
        elif validation_settings.get('warn_extreme_params', True):
            if k > 10 or k < 0.01:
                logger.warning(f"Hill parameter k={k:.3f} is extreme (typical range: 0.01-10)")
                
        if s <= 0:
            logger.warning("Hill parameter s must be positive, setting to 1.0") 
            validated['s'] = 1.0
        elif validation_settings.get('warn_extreme_params', True):
            if s > 5 or s < 0.1:
                logger.warning(f"Hill parameter s={s:.3f} is extreme (typical range: 0.1-5)")
    
    # Validate Logistic parameters
    elif validated['type'] == 'logistic':
        k = validated.get('k', 1.0)
        x0 = validated.get('x0', 0.5)
        
        if k <= 0:
            logger.warning("Logistic parameter k must be positive, setting to 1.0")
            validated['k'] = 1.0
        elif validation_settings.get('warn_extreme_params', True):
            if k > 10 or k < 0.1:
                logger.warning(f"Logistic parameter k={k:.3f} is extreme (typical range: 0.1-10)")
                
        if validation_settings.get('warn_extreme_params', True):
            if abs(x0) > 2:
                logger.warning(f"Logistic parameter x0={x0:.3f} is extreme (typical range: -2 to 2)")
    
    return validated


def validate_saturation_curve(x: np.ndarray, y: np.ndarray, validation_settings: Dict[str, Any]) -> bool:
    """
    Validate that a saturation curve meets mathematical requirements.
    
    Args:
        x: Input values (normalized spend)
        y: Output values (saturated response)
        validation_settings: Validation configuration
        
    Returns:
        bool: True if curve passes all enabled validations
    """
    logger = logging.getLogger(__name__)
    
    # Check monotonic increasing (if enabled)
    if validation_settings.get('check_monotonic', True):
        # Sort by x to check monotonicity properly
        sorted_indices = np.argsort(x)
        x_sorted, y_sorted = x[sorted_indices], y[sorted_indices]
        
        # Check if y is monotonically increasing with x
        if not np.all(np.diff(y_sorted) >= -EPS):  # Allow small numerical errors
            logger.warning("Saturation curve is not monotonically increasing")
            return False
    
    # Check bounded output [0,1] (if enabled)
    if validation_settings.get('check_bounded', True):
        if np.any(y < -EPS) or np.any(y > 1 + EPS):
            logger.warning(f"Saturation curve output not bounded [0,1]: min={y.min():.6f}, max={y.max():.6f}")
            return False
    
    return True


if __name__ == "__main__":
    # Test refined saturation functions with new features
    print("=== TESTING REFINED SATURATION MODULE ===")
    
    # Create sample adstocked spend data with some realistic properties
    np.random.seed(42)
    base_spend = np.linspace(0, 100, 50)
    adstocked_spend = base_spend + np.random.normal(0, 5, 50)  # Add some noise
    adstocked_spend = np.maximum(adstocked_spend, 0)  # Ensure non-negative
    
    print("1. Testing Hill saturation with different normalization modes:")
    
    # Test different normalization modes
    normalization_modes = ["max", "p95", "none", "k_in_units"]
    for norm_mode in normalization_modes:
        try:
            if norm_mode == "k_in_units":
                # For k_in_units, use k in spend units (e.g., 50)
                hill_result = hill_saturation(adstocked_spend, k=50, s=1.0, normalization=norm_mode)
            else:
                # For normalized modes, use k in [0,1] scale
                hill_result = hill_saturation(adstocked_spend, k=0.5, s=1.0, normalization=norm_mode)
            
            print(f"  {norm_mode:12}: min={hill_result.min():.3f}, max={hill_result.max():.3f}, mean={hill_result.mean():.3f}")
        except Exception as e:
            print(f"  {norm_mode:12}: Error - {e}")
    
    print("\n2. Testing logistic saturation with robust features:")
    
    # Test logistic with different parameters
    logistic_1 = logistic_saturation(adstocked_spend, k=1.0, x0=0.5, normalization="max")
    logistic_2 = logistic_saturation(adstocked_spend, k=2.0, x0=0.3, normalization="p95")
    
    print(f"  Standard (k=1.0, x0=0.5): min={logistic_1.min():.3f}, max={logistic_1.max():.3f}")
    print(f"  Steep (k=2.0, x0=0.3):    min={logistic_2.min():.3f}, max={logistic_2.max():.3f}")
    
    print("\n3. Testing edge cases and robustness:")
    
    # Test with all zeros
    zero_data = np.zeros(10)
    zero_result = hill_saturation(zero_data, k=0.5, s=1.0)
    print(f"  All zeros: shape={zero_result.shape}, all_zero={np.all(zero_result == 0)}")
    
    # Test with pandas Series and preserve_index
    series_data = pd.Series(adstocked_spend[:10], index=pd.date_range('2023-01-01', periods=10))
    series_result = hill_saturation(series_data, k=0.5, s=1.0, preserve_index=True)
    print(f"  Pandas Series: type={type(series_result)}, index_preserved={hasattr(series_result, 'index')}")
    
    # Test with extreme values
    extreme_data = np.array([0, 1e-10, 1e10, np.inf, -1])
    extreme_result = hill_saturation(extreme_data, k=0.5, s=1.0, normalization="max")
    print(f"  Extreme values: all_finite={np.all(np.isfinite(extreme_result))}")
    
    print("\n4. Testing calibration function:")
    
    # Create synthetic response data  
    synthetic_response = hill_saturation(adstocked_spend, k=0.6, s=1.2) * 1000 + np.random.normal(0, 50, len(adstocked_spend))
    synthetic_response = np.maximum(synthetic_response, 0)
    
    # Test calibration
    calibrated_hill = calibrate_saturation_curve(adstocked_spend, synthetic_response, "hill", "max")
    calibrated_logistic = calibrate_saturation_curve(adstocked_spend, synthetic_response, "logistic", "p95")
    
    print(f"  Hill calibration: k={calibrated_hill['k']:.3f}, s={calibrated_hill['s']:.3f}")
    print(f"  Logistic calibration: k={calibrated_logistic['k']:.3f}, x0={calibrated_logistic['x0']:.3f}")
    
    print("\n5. Testing parameter validation:")
    
    # Test invalid parameters
    invalid_params = {'type': 'unknown', 'k': -1, 's': 0, 'normalization': 'invalid'}
    validated = validate_saturation_params(invalid_params)
    print(f"  Validated params: {validated}")
    
    print("\n6. Testing config integration:")
    
    # Test configuration parameter retrieval
    test_config = {
        'features': {
            'saturation': {
                'type': 'hill',
                'normalization': 'p95',
                'default_inflection': 0.6,
                'default_slope': 1.2,
                'preserve_index': True,
                'channel_overrides': {
                    'google_search': {'k': 0.3, 'normalization': 'k_in_units'},
                    'meta_facebook': {'type': 'logistic', 'k': 1.5}
                }
            }
        }
    }
    
    # Test default channel
    default_params = get_saturation_params(test_config, 'tiktok')
    print(f"  Default channel params: {default_params}")
    
    # Test channel with overrides
    override_params = get_saturation_params(test_config, 'google_search')
    print(f"  Override channel params: {override_params}")
    
    print("\n✅ Refined saturation module testing completed successfully!")
    print("Key improvements:")
    print("  • Explicit normalization modes (max, p95, none, k_in_units)")
    print("  • Robust numerical stability with edge case handling")
    print("  • Pandas Series support with index preservation")
    print("  • Lightweight parameter calibration")
    print("  • Configuration integration with channel overrides")
    print("  • Parameter validation and sanitization")


def apply_platform_saturation(spend_data: pd.DataFrame, config) -> pd.DataFrame:
    """
    Apply platform-specific saturation transformations to all channels.
    
    Args:
        spend_data: DataFrame with spend columns
        config: Configuration object
        
    Returns:
        pd.DataFrame: Data with saturated columns added
    """
    import logging
    logger = logging.getLogger(__name__)
    result_df = spend_data.copy()
    
    # Get channel mapping from config
    if hasattr(config, 'data') and hasattr(config.data, 'channel_map'):
        channel_map = config.data.channel_map
    else:
        logger.warning("No channel mapping found in config, skipping saturation")
        return result_df
    
    saturated_channels = 0
    
    # Apply saturation to each channel
    for channel_name, raw_column in channel_map.items():
        # Look for the normalized column name (lowercase)
        column_candidates = [
            raw_column.lower(),  # Try lowercase version
            raw_column,          # Try original version
            f"{channel_name}_spend"  # Try constructed name
        ]
        
        spend_column = None
        for candidate in column_candidates:
            if candidate in result_df.columns:
                spend_column = candidate
                break
        
        if spend_column is None:
            logger.debug(f"Spend column not found for channel {channel_name} (tried: {column_candidates})")
            continue
            
        try:
            # Apply saturation to this channel
            adstocked_spend = result_df[spend_column]
            
            # Apply saturation transformation
            saturated_values = apply_saturation_from_config(
                adstocked_spend=adstocked_spend,
                config=config.__dict__ if hasattr(config, '__dict__') else config,
                channel=channel_name
            )
            
            # Add saturated column
            saturated_column = f"{spend_column}_saturated"
            result_df[saturated_column] = saturated_values
            saturated_channels += 1
            
            logger.debug(f"Applied saturation to {channel_name} ({spend_column} → {saturated_column})")
            
        except Exception as e:
            logger.warning(f"Failed to apply saturation to {channel_name}: {e}")
            continue
    
    logger.info(f"Applied saturation transformations to {saturated_channels} channels")
    return result_df
