"""
Creative fatigue modeling module.

Implements creative fatigue effects where ad effectiveness declines over time
without creative refreshes. Features robust refresh detection, parameter validation,
and configurable fatigue floors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple
import logging
from datetime import datetime, timedelta

# Small epsilon for numerical stability
EPS = 1e-12


def validate_fatigue_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize creative fatigue parameters.
    
    Args:
        config: Creative fatigue configuration dictionary
        
    Returns:
        Dict: Validated parameters with warnings for invalid values
    """
    logger = logging.getLogger(__name__)
    validated = config.copy()
    
    # Validate half_life
    half_life = validated.get('half_life', 14)
    if half_life <= 0:
        logger.warning("Half life must be positive, setting to 14 days")
        validated['half_life'] = 14
    elif half_life > 365:
        logger.warning(f"Half life {half_life} days is very long, consider shorter periods")
    
    # Validate fatigue_floor
    fatigue_floor = validated.get('fatigue_floor', 0.2)
    if fatigue_floor < 0 or fatigue_floor >= 1:
        logger.warning("Fatigue floor must be in [0, 1), setting to 0.2")
        validated['fatigue_floor'] = 0.2
    elif fatigue_floor < 0.1:
        logger.warning(f"Very low fatigue floor ({fatigue_floor:.1%}) may cause numerical issues")
    
    # Validate refresh detection parameters
    refresh_config = validated.get('refresh_detection', {})
    min_gap = refresh_config.get('min_gap_days', 7)
    if min_gap < 1:
        logger.warning("Minimum gap between refreshes must be >= 1 day, setting to 7")
        refresh_config['min_gap_days'] = 7
    
    threshold = refresh_config.get('spend_change_threshold', 0.5)
    if threshold <= 0 or threshold > 5.0:
        logger.warning("Spend change threshold should be in (0, 5], setting to 0.5")
        refresh_config['spend_change_threshold'] = 0.5
    
    validated['refresh_detection'] = refresh_config
    
    # Validate external matching parameters
    external_config = validated.get('external_matching', {})
    max_distance = external_config.get('max_distance_days', 5)
    if max_distance < 0:
        logger.warning("Max distance for external matching must be >= 0, setting to 5")
        external_config['max_distance_days'] = 5
    
    validated['external_matching'] = external_config
    
    return validated


def get_fatigue_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate creative fatigue configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dict: Validated creative fatigue configuration
    """
    fatigue_config = config.get('features', {}).get('creative_fatigue', {})
    return validate_fatigue_params(fatigue_config)


def apply_creative_fatigue(
    spend_data: pd.DataFrame,
    creative_refresh_signals: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply creative fatigue transformation to media spend with full configuration support.
    
    Args:
        spend_data: DataFrame with spend columns and dates
        creative_refresh_signals: Optional DataFrame indicating creative refreshes
        config: Configuration dictionary
        
    Returns:
        pd.DataFrame: Data with creative fatigue applied
    """
    logger = logging.getLogger(__name__)
    
    if config is None:
        logger.warning("No configuration provided, using defaults")
        config = {'features': {'creative_fatigue': {'enabled': False}}}
    
    fatigue_config = get_fatigue_config(config)
    
    if not fatigue_config.get('enabled', False):
        logger.info("Creative fatigue modeling disabled")
        return spend_data.copy()
    
    logger.info("Applying creative fatigue modeling with enhanced parameters")
    
    result_df = spend_data.copy()
    
    # Ensure consistent datetime handling
    date_col = config.get('data', {}).get('date_col', 'date')
    if date_col not in spend_data.columns:
        logger.error(f"Date column '{date_col}' not found in spend_data")
        return spend_data.copy()
    
    dates = pd.to_datetime(spend_data[date_col])
    
    # Get channel mapping
    channel_map = config.get('data', {}).get('channel_map', {})
    
    applied_count = 0
    for channel_name, spend_col in channel_map.items():
        if spend_col in spend_data.columns:
            try:
                # Apply fatigue to this channel
                fatigued_spend = calculate_creative_fatigue_vectorized(
                    spend_series=spend_data[spend_col],
                    dates=dates,
                    config=fatigue_config,
                    creative_refreshes=creative_refresh_signals
                )
                
                result_df[f"{channel_name}_fatigue_adjusted"] = fatigued_spend
                applied_count += 1
                logger.debug(f"Applied creative fatigue to {channel_name}")
                
            except Exception as e:
                logger.error(f"Failed to apply fatigue to {channel_name}: {e}")
    
    logger.info(f"Applied creative fatigue to {applied_count}/{len(channel_map)} channels")
    return result_df


def calculate_creative_fatigue_vectorized(
    spend_series: pd.Series,
    dates: pd.DatetimeIndex,
    config: Dict[str, Any],
    creative_refreshes: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Calculate creative fatigue effects using stable vectorized operations.
    
    Args:
        spend_series: Media spend values
        dates: Corresponding dates as DatetimeIndex
        config: Fatigue configuration dictionary
        creative_refreshes: External refresh signals
        
    Returns:
        pd.Series: Fatigue-adjusted spend values
    """
    logger = logging.getLogger(__name__)
    
    if len(spend_series) == 0:
        logger.warning("Empty spend series provided")
        return spend_series.copy()
    
    # Ensure dates are DatetimeIndex
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    
    # Validate series lengths match
    if len(spend_series) != len(dates):
        raise ValueError(f"Spend series length ({len(spend_series)}) doesn't match dates length ({len(dates)})")
    
    # Extract parameters
    half_life = config.get('half_life', 14)
    fatigue_floor = config.get('fatigue_floor', 0.2)
    refresh_signal = config.get('refresh_signal', 'weekly_creative_change')
    
    # Calculate decay rate from half-life
    decay_rate = np.log(0.5) / half_life
    
    # Detect creative refresh points with debouncing
    refresh_points = detect_creative_refreshes_robust(
        spend_series, dates, refresh_signal, config, creative_refreshes
    )
    
    # Calculate stable vectorized fatigue
    fatigue_multiplier = calculate_vectorized_decay(
        dates, refresh_points, decay_rate, fatigue_floor
    )
    
    # Apply fatigue to spend (multiplicative effect)
    fatigued_spend = spend_series * fatigue_multiplier
    
    # Validate output and ensure floor is respected for non-zero spend
    if not np.all(np.isfinite(fatigued_spend)):
        logger.warning("Non-finite values in fatigued spend, replacing with original")
        fatigued_spend = np.where(np.isfinite(fatigued_spend), fatigued_spend, spend_series)
    
    # Ensure fatigue floor is respected for positive spend (zero spend should remain zero)
    positive_spend_mask = spend_series > EPS
    if positive_spend_mask.any():
        min_fatigued = spend_series * fatigue_floor
        fatigued_spend = np.where(
            positive_spend_mask & (fatigued_spend < min_fatigued),
            min_fatigued,
            fatigued_spend
        )
    
    return pd.Series(fatigued_spend, index=spend_series.index)


def calculate_vectorized_decay(
    dates: pd.DatetimeIndex,
    refresh_points: np.ndarray,
    decay_rate: float,
    fatigue_floor: float
) -> np.ndarray:
    """
    Calculate fatigue decay using stable vectorized operations.
    
    Args:
        dates: DatetimeIndex of dates
        refresh_points: Boolean array of refresh points
        decay_rate: Exponential decay rate (negative)
        fatigue_floor: Minimum fatigue multiplier
        
    Returns:
        np.ndarray: Fatigue multiplier values
    """
    n = len(dates)
    fatigue_multiplier = np.ones(n)
    
    # Find refresh indices
    refresh_indices = np.where(refresh_points)[0]
    
    if len(refresh_indices) == 0:
        # No refreshes detected - apply decay from start
        days_elapsed = (dates - dates[0]).days
        fatigue_multiplier = np.exp(decay_rate * days_elapsed)
    else:
        # Apply decay between refresh points
        for i in range(n):
            # Find the most recent refresh before or at this point
            recent_refresh_idx = refresh_indices[refresh_indices <= i]
            
            if len(recent_refresh_idx) > 0:
                # Calculate days since most recent refresh
                last_refresh_idx = recent_refresh_idx[-1]
                days_since_refresh = (dates[i] - dates[last_refresh_idx]).days
                fatigue_multiplier[i] = np.exp(decay_rate * days_since_refresh)
            else:
                # Before first refresh - decay from start
                days_elapsed = (dates[i] - dates[0]).days
                fatigue_multiplier[i] = np.exp(decay_rate * days_elapsed)
    
    # Apply fatigue floor
    fatigue_multiplier = np.maximum(fatigue_multiplier, fatigue_floor)
    
    return fatigue_multiplier


def detect_creative_refreshes_robust(
    spend_series: pd.Series,
    dates: pd.DatetimeIndex,
    refresh_signal: str,
    config: Dict[str, Any],
    creative_refreshes: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Detect creative refresh points with robust debouncing and validation.
    
    Args:
        spend_series: Media spend values
        dates: DatetimeIndex of dates
        refresh_signal: Type of refresh detection
        config: Configuration with detection parameters
        creative_refreshes: External refresh data
        
    Returns:
        np.ndarray: Boolean array indicating refresh points
    """
    logger = logging.getLogger(__name__)
    refresh_points = np.zeros(len(spend_series), dtype=bool)
    
    try:
        if creative_refreshes is not None:
            # Use external refresh signals with tolerance matching
            refresh_points = detect_external_refreshes_with_tolerance(
                dates, creative_refreshes, config.get('external_matching', {})
            )
        
        elif refresh_signal == "weekly_creative_change":
            # Detect spend pattern changes with debouncing
            refresh_points = detect_spend_pattern_changes_debounced(
                spend_series, dates, config.get('refresh_detection', {})
            )
        
        elif refresh_signal == "monthly_creative_change":
            # Monthly pattern changes
            detection_config = config.get('refresh_detection', {})
            detection_config['rolling_window_days'] = 30
            refresh_points = detect_spend_pattern_changes_debounced(
                spend_series, dates, detection_config
            )
        
        elif refresh_signal == "campaign_launch":
            # Campaign launches with validation
            refresh_points = detect_campaign_launches_robust(
                spend_series, config.get('refresh_detection', {})
            )
        
        else:
            # Default: periodic refreshes
            refresh_points = detect_periodic_refreshes(dates, period_days=30)
        
        # Validate refresh detection
        refresh_points = validate_refresh_detection(
            refresh_points, dates, config.get('validation', {}), refresh_signal
        )
    
    except Exception as e:
        logger.error(f"Error in refresh detection: {e}")
        # Fallback to periodic refreshes
        refresh_points = detect_periodic_refreshes(dates, period_days=30)
    
    return refresh_points


def detect_external_refreshes_with_tolerance(
    dates: pd.DatetimeIndex,
    creative_refreshes: pd.DataFrame,
    external_config: Dict[str, Any]
) -> np.ndarray:
    """
    Use external creative refresh signals with distance tolerance.
    
    Args:
        dates: DatetimeIndex of dates
        creative_refreshes: DataFrame with refresh dates
        external_config: External matching configuration
        
    Returns:
        np.ndarray: Boolean refresh indicators
    """
    logger = logging.getLogger(__name__)
    refresh_points = np.zeros(len(dates), dtype=bool)
    
    max_distance_days = external_config.get('max_distance_days', 5)
    require_exact = external_config.get('require_exact_match', False)
    
    if 'refresh_date' not in creative_refreshes.columns:
        logger.warning("No 'refresh_date' column in external refresh data")
        return refresh_points
    
    refresh_dates = pd.to_datetime(creative_refreshes['refresh_date'])
    matched_count = 0
    
    for refresh_date in refresh_dates:
        if require_exact:
            # Exact match required
            exact_matches = dates == refresh_date
            if exact_matches.any():
                refresh_points[exact_matches] = True
                matched_count += 1
        else:
            # Find closest date within tolerance
            time_diff = np.abs((dates - refresh_date).days)
            closest_idx = np.argmin(time_diff)
            
            if time_diff[closest_idx] <= max_distance_days:
                refresh_points[closest_idx] = True
                matched_count += 1
            else:
                logger.debug(f"External refresh date {refresh_date} too far from any data point (closest: {time_diff[closest_idx]} days)")
    
    logger.info(f"Matched {matched_count}/{len(refresh_dates)} external refresh signals")
    return refresh_points


def detect_spend_pattern_changes_debounced(
    spend_series: pd.Series,
    dates: pd.DatetimeIndex,
    detection_config: Dict[str, Any]
) -> np.ndarray:
    """
    Detect significant changes in spending patterns with debouncing.
    
    Args:
        spend_series: Media spend values
        dates: DatetimeIndex of dates
        detection_config: Detection configuration parameters
        
    Returns:
        np.ndarray: Boolean refresh indicators
    """
    window = detection_config.get('rolling_window_days', 7)
    threshold = detection_config.get('spend_change_threshold', 0.5)
    min_gap_days = detection_config.get('min_gap_days', 7)
    
    refresh_points = np.zeros(len(spend_series), dtype=bool)
    
    # Guard against zero/very low spend
    spend_with_eps = spend_series + EPS
    
    # Calculate rolling mean with minimum periods
    rolling_mean = spend_with_eps.rolling(window=window, min_periods=1).mean()
    
    # Calculate percentage change, handling division by zero
    pct_change = rolling_mean.pct_change().fillna(0)
    
    # Identify significant changes
    significant_changes = np.abs(pct_change) > threshold
    
    # Apply debouncing - no refresh within min_gap_days of previous
    last_refresh_date = None
    
    for i, (is_significant, current_date) in enumerate(zip(significant_changes, dates)):
        if is_significant and i > 0:  # Skip first point for percentage change
            if last_refresh_date is None or (current_date - last_refresh_date).days >= min_gap_days:
                refresh_points[i] = True
                last_refresh_date = current_date
    
    # Always mark the first day as a refresh if there's spend
    if len(spend_series) > 0 and spend_series.iloc[0] > EPS:
        refresh_points[0] = True
    
    return refresh_points


def detect_campaign_launches_robust(
    spend_series: pd.Series,
    detection_config: Dict[str, Any]
) -> np.ndarray:
    """
    Detect campaign launches with validation and minimum spend threshold.
    
    Args:
        spend_series: Media spend values
        detection_config: Detection configuration parameters
        
    Returns:
        np.ndarray: Boolean refresh indicators
    """
    min_launch_spend = detection_config.get('campaign_launch_threshold', 0.1)
    min_gap_days = detection_config.get('min_gap_days', 7)
    
    refresh_points = np.zeros(len(spend_series), dtype=bool)
    
    # Identify transitions from near-zero to meaningful spend
    prev_low = spend_series.shift(1) <= min_launch_spend
    current_high = spend_series > min_launch_spend
    
    potential_launches = prev_low & current_high
    
    # Apply debouncing for launches
    last_launch_idx = -min_gap_days - 1
    
    for i, is_launch in enumerate(potential_launches):
        if is_launch and i > 0:  # Skip first point
            if (i - last_launch_idx) >= min_gap_days:
                refresh_points[i] = True
                last_launch_idx = i
    
    # Mark first meaningful spend as launch
    if len(spend_series) > 0 and spend_series.iloc[0] > min_launch_spend:
        refresh_points[0] = True
    
    return refresh_points


def detect_periodic_refreshes(
    dates: pd.DatetimeIndex,
    period_days: int = 30
) -> np.ndarray:
    """
    Assume periodic creative refreshes.
    
    Args:
        dates: DatetimeIndex of dates
        period_days: Days between refreshes
        
    Returns:
        np.ndarray: Boolean refresh indicators
    """
    refresh_points = np.zeros(len(dates), dtype=bool)
    
    if len(dates) == 0:
        return refresh_points
    
    start_date = dates[0]
    
    for i, date in enumerate(dates):
        days_since_start = (date - start_date).days
        
        # Mark refresh every period_days
        if days_since_start % period_days == 0:
            refresh_points[i] = True
    
    return refresh_points


def validate_refresh_detection(
    refresh_points: np.ndarray,
    dates: pd.DatetimeIndex,
    validation_config: Dict[str, Any],
    detection_method: str
) -> np.ndarray:
    """
    Validate refresh detection results and apply guardrails.
    
    Args:
        refresh_points: Boolean array of detected refreshes
        dates: DatetimeIndex of dates
        validation_config: Validation configuration
        detection_method: Method used for detection
        
    Returns:
        np.ndarray: Validated refresh points
    """
    logger = logging.getLogger(__name__)
    
    if len(refresh_points) == 0:
        return refresh_points
    
    refresh_rate = refresh_points.sum() / len(refresh_points)
    max_rate = validation_config.get('max_refresh_rate', 0.4)
    min_rate = validation_config.get('min_refresh_rate', 0.05)
    
    # Check for excessive refreshes
    if validation_config.get('warn_excessive_refreshes', True):
        if refresh_rate > max_rate:
            logger.warning(f"High refresh rate detected: {refresh_rate:.1%} > {max_rate:.1%} (method: {detection_method})")
            
            # Keep only the strongest refresh signals if too many
            if refresh_rate > max_rate * 1.5:  # If really excessive
                target_count = int(len(refresh_points) * max_rate)
                # Keep first and last, plus evenly spaced ones
                validated = np.zeros_like(refresh_points)
                validated[0] = True
                validated[-1] = True
                
                # Fill remaining slots evenly
                if target_count > 2:
                    step = len(refresh_points) // (target_count - 2)
                    for i in range(step, len(refresh_points) - step, step):
                        validated[i] = True
                
                logger.info(f"Reduced refresh rate from {refresh_rate:.1%} to {validated.sum()/len(validated):.1%}")
                return validated
        
        elif refresh_rate < min_rate:
            logger.warning(f"Low refresh rate detected: {refresh_rate:.1%} < {min_rate:.1%} (method: {detection_method})")
    
    # Check fatigue consistency if enabled
    if validation_config.get('check_fatigue_consistency', True):
        # Could add check that fatigue decays monotonically between refreshes
        pass
    
    return refresh_points



def calculate_fatigue_metrics(
    original_spend: pd.Series,
    fatigued_spend: pd.Series,
    dates: pd.DatetimeIndex,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics to assess creative fatigue impact.
    
    Args:
        original_spend: Original spend values
        fatigued_spend: Fatigue-adjusted spend values
        dates: DatetimeIndex of dates
        config: Configuration dictionary with detection parameters
        
    Returns:
        Dict: Fatigue impact metrics
    """
    # Guard against zero spend to avoid division issues
    total_original = original_spend.sum()
    if total_original <= EPS:
        return {
            'fatigue_impact_pct': 0.0,
            'max_fatigue_pct': 0.0,
            'fatigue_variability': 0.0,
            'refresh_count': 0,
            'avg_days_between_refreshes': 0.0,
            'fatigue_floor_reached': False
        }
    
    # Calculate average fatigue impact
    fatigue_impact = 1 - (fatigued_spend.sum() / total_original)
    
    # Calculate fatigue multiplier (avoiding division by zero)
    fatigue_multiplier = fatigued_spend / (original_spend + EPS)
    
    # Calculate maximum fatigue reached
    max_fatigue = 1 - fatigue_multiplier.min()
    
    # Calculate fatigue variability
    fatigue_std = fatigue_multiplier.std()
    
    # Count refresh events using the same detector as used for the series
    refresh_signal = config.get('refresh_signal', 'weekly_creative_change')
    refresh_points = detect_creative_refreshes_robust(
        original_spend, dates, refresh_signal, config, None
    )
    refresh_count = refresh_points.sum()
    
    # Calculate average days between refreshes
    avg_days_between = len(dates) / max(refresh_count, 1)
    
    # Check if fatigue floor was reached
    fatigue_floor = config.get('fatigue_floor', 0.2)
    floor_reached = fatigue_multiplier.min() <= (fatigue_floor + EPS)
    
    return {
        'fatigue_impact_pct': fatigue_impact * 100,
        'max_fatigue_pct': max_fatigue * 100,
        'fatigue_variability': fatigue_std,
        'refresh_count': int(refresh_count),
        'avg_days_between_refreshes': avg_days_between,
        'fatigue_floor_reached': floor_reached
    }


def apply_fatigue_to_features(
    feature_df: pd.DataFrame,
    config: Dict[str, Any],
    creative_refresh_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Apply creative fatigue to feature engineered data (recommended for MMM).
    
    This applies fatigue to adstocked features rather than raw spend,
    modeling fatigue as an effectiveness scaler on remembered exposure.
    
    Args:
        feature_df: Feature-engineered DataFrame
        config: Full configuration dictionary
        creative_refresh_data: External creative refresh signals
        
    Returns:
        pd.DataFrame: Data with fatigue effects applied
    """
    logger = logging.getLogger(__name__)
    
    fatigue_config = get_fatigue_config(config)
    
    if not fatigue_config.get('enabled', False):
        logger.info("Creative fatigue disabled, returning original features")
        return feature_df.copy()
    
    # Check apply_to_stage configuration
    apply_stage = fatigue_config.get('apply_to_stage', 'adstocked')
    if apply_stage not in ['raw', 'adstocked']:
        logger.warning(f"Unknown apply_to_stage '{apply_stage}', defaulting to 'adstocked'")
        apply_stage = 'adstocked'
    
    result_df = feature_df.copy()
    date_col = config.get('data', {}).get('date_col', 'date')
    
    if date_col not in feature_df.columns:
        logger.error(f"Date column '{date_col}' not found in feature DataFrame")
        return feature_df.copy()
    
    dates = pd.DatetimeIndex(pd.to_datetime(feature_df[date_col]))
    
    # Apply fatigue to appropriate features
    if apply_stage == 'adstocked':
        # Apply to adstocked features (recommended for MMM interpretability)
        target_cols = [col for col in feature_df.columns if col.endswith('_adstocked')]
        suffix = '_fatigue_adjusted'
    else:
        # Apply to raw spend features
        channel_map = config.get('data', {}).get('channel_map', {})
        target_cols = [col for col in feature_df.columns if col in channel_map.values()]
        suffix = '_fatigue_adjusted'
    
    applied_count = 0
    for col in target_cols:
        try:
            # Apply fatigue to this feature
            fatigued_values = calculate_creative_fatigue_vectorized(
                spend_series=feature_df[col],
                dates=dates,
                config=fatigue_config,
                creative_refreshes=creative_refresh_data
            )
            
            # Extract channel name for output column
            if col.endswith('_adstocked'):
                channel_name = col.replace('_adstocked', '')
            else:
                # Find channel name from channel map
                channel_map = config.get('data', {}).get('channel_map', {})
                channel_name = next((k for k, v in channel_map.items() if v == col), col)
            
            result_df[f"{channel_name}{suffix}"] = fatigued_values
            applied_count += 1
            
            logger.debug(f"Applied creative fatigue to {col} -> {channel_name}{suffix}")
            
        except Exception as e:
            logger.error(f"Failed to apply fatigue to {col}: {e}")
    
    logger.info(f"Applied creative fatigue to {applied_count}/{len(target_cols)} {apply_stage} features")
    
    # Calculate and log metrics if enabled
    if fatigue_config.get('metrics', {}).get('calculate_fatigue_impact', True) and applied_count > 0:
        try:
            # Calculate aggregate metrics across all channels
            original_total = sum(feature_df[col].sum() for col in target_cols if col in feature_df.columns)
            fatigued_total = sum(result_df[f"{col.replace('_adstocked', '')}{suffix}"].sum() 
                               for col in target_cols 
                               if f"{col.replace('_adstocked', '')}{suffix}" in result_df.columns)
            
            if original_total > EPS:
                overall_impact = (1 - fatigued_total / original_total) * 100
                logger.info(f"Overall creative fatigue impact: {overall_impact:.1f}% reduction in effectiveness")
        
        except Exception as e:
            logger.debug(f"Could not calculate aggregate fatigue metrics: {e}")
    
    return result_df


def apply_creative_fatigue_from_config(
    data: pd.DataFrame,
    config: Dict[str, Any],
    stage: str = "adstocked",
    creative_refresh_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Apply creative fatigue using complete YAML configuration.
    
    This is the main entry point for config-driven creative fatigue modeling.
    
    Args:
        data: DataFrame with spend/feature data
        config: Complete configuration dictionary from YAML
        stage: Whether to apply to "raw" spend or "adstocked" features
        creative_refresh_data: Optional external refresh data
        
    Returns:
        pd.DataFrame: Data with fatigue effects applied
    """
    # Override stage from config if specified
    fatigue_config = get_fatigue_config(config)
    config_stage = fatigue_config.get('apply_to_stage', stage)
    
    if config_stage == 'adstocked':
        return apply_fatigue_to_features(data, config, creative_refresh_data)
    else:
        return apply_creative_fatigue(data, creative_refresh_data, config)


if __name__ == "__main__":
    # Comprehensive test of refined creative fatigue functions
    print("=== TESTING REFINED CREATIVE FATIGUE MODULE ===")
    
    import yaml
    
    # Create realistic test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create spend with natural variability and some campaign launches
    base_spend = np.random.exponential(500, 100)
    
    # Add some campaign launches (spend jumps)
    base_spend[20:30] *= 3  # Big campaign
    base_spend[50:52] = 0   # Campaign pause
    base_spend[60:70] *= 2  # Medium campaign
    
    spend_data = pd.DataFrame({
        'date': dates,
        'google_search_spend': base_spend,
        'meta_facebook_spend': base_spend * 0.8 + np.random.normal(0, 50, 100)
    })
    
    # Test configuration
    test_config = {
        'data': {
            'date_col': 'date',
            'channel_map': {
                'google_search': 'google_search_spend',
                'meta_facebook': 'meta_facebook_spend'
            }
        },
        'features': {
            'creative_fatigue': {
                'enabled': True,
                'half_life': 14,
                'fatigue_floor': 0.2,
                'refresh_signal': 'weekly_creative_change',
                'apply_to_stage': 'adstocked',
                'refresh_detection': {
                    'min_gap_days': 7,
                    'spend_change_threshold': 0.4,
                    'rolling_window_days': 7,
                    'campaign_launch_threshold': 100
                },
                'external_matching': {
                    'max_distance_days': 3,
                    'require_exact_match': False
                },
                'validation': {
                    'check_fatigue_consistency': True,
                    'warn_excessive_refreshes': True,
                    'max_refresh_rate': 0.3,
                    'min_refresh_rate': 0.05
                },
                'metrics': {
                    'calculate_fatigue_impact': True,
                    'track_refresh_frequency': True
                }
            }
        }
    }
    
    print("1. Testing parameter validation:")
    fatigue_config = get_fatigue_config(test_config)
    print(f"   Validated config keys: {list(fatigue_config.keys())}")
    
    print("\n2. Testing robust refresh detection:")
    
    # Test different detection methods
    detection_methods = ['weekly_creative_change', 'campaign_launch', 'monthly_creative_change']
    
    for method in detection_methods:
        test_config['features']['creative_fatigue']['refresh_signal'] = method
        
        refreshes = detect_creative_refreshes_robust(
            spend_data['google_search_spend'],
            pd.DatetimeIndex(spend_data['date']),
            method,
            fatigue_config,
            None
        )
        
        refresh_rate = refreshes.sum() / len(refreshes)
        print(f"   {method}: {refreshes.sum()} refreshes ({refresh_rate:.1%} rate)")
    
    print("\n3. Testing vectorized fatigue calculation:")
    
    fatigued_spend = calculate_creative_fatigue_vectorized(
        spend_data['google_search_spend'],
        pd.DatetimeIndex(spend_data['date']),
        fatigue_config,
        None
    )
    
    print(f"   Original range: [{spend_data['google_search_spend'].min():.0f}, {spend_data['google_search_spend'].max():.0f}]")
    print(f"   Fatigued range: [{fatigued_spend.min():.0f}, {fatigued_spend.max():.0f}]")
    
    # Test fatigue floor
    fatigue_multiplier = fatigued_spend / (spend_data['google_search_spend'] + EPS)
    print(f"   Fatigue multiplier range: [{fatigue_multiplier.min():.3f}, {fatigue_multiplier.max():.3f}]")
    print(f"   Floor respected: {fatigue_multiplier.min() >= fatigue_config.get('fatigue_floor', 0.2) - EPS}")
    
    print("\n4. Testing external refresh matching:")
    
    # Create external refresh data
    external_refreshes = pd.DataFrame({
        'refresh_date': ['2023-01-15', '2023-02-10', '2023-03-05', '2023-03-25']
    })
    
    matched_refreshes = detect_external_refreshes_with_tolerance(
        pd.DatetimeIndex(spend_data['date']),
        external_refreshes,
        fatigue_config.get('external_matching', {})
    )
    
    print(f"   External refreshes provided: {len(external_refreshes)}")
    print(f"   Refreshes matched: {matched_refreshes.sum()}")
    
    print("\n5. Testing complete config-driven pipeline:")
    
    # Create adstocked features for testing
    feature_data = spend_data.copy()
    feature_data['google_search_adstocked'] = spend_data['google_search_spend'] * 0.9  # Simulate adstock
    feature_data['meta_facebook_adstocked'] = spend_data['meta_facebook_spend'] * 0.85
    
    # Apply fatigue using config
    result = apply_creative_fatigue_from_config(
        feature_data,
        test_config,
        creative_refresh_data=external_refreshes
    )
    
    fatigue_cols = [col for col in result.columns if 'fatigue_adjusted' in col]
    print(f"   Fatigue columns created: {fatigue_cols}")
    
    print("\n6. Testing comprehensive metrics:")
    
    metrics = calculate_fatigue_metrics(
        spend_data['google_search_spend'],
        fatigued_spend,
        pd.DatetimeIndex(spend_data['date']),
        fatigue_config
    )
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.2f}")
        else:
            print(f"   {metric}: {value}")
    
    print("\n✅ REFINED CREATIVE FATIGUE MODULE TESTING COMPLETED!")
    print()
    print("KEY IMPROVEMENTS IMPLEMENTED:")
    print("• Guardrails & parameter validation with informative errors")
    print("• Stable vectorized decay calculation since last refresh")
    print("• Debounced refresh detection (prevents chatter on noisy spend)")
    print("• External refresh matching with distance tolerance")
    print("• Fatigue floor prevents effectiveness from vanishing")
    print("• Metrics aligned with chosen detector and protected divisions")
    print("• Consistent DatetimeIndex handling throughout pipeline")
    print("• Configurable application to raw vs adstocked features")
    print("• Complete YAML configuration integration")
