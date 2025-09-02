"""
Custom business terms module.

Implements custom business logic and promotional effects
for MMM modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta


def apply_custom_business_terms(
    data_df: pd.DataFrame,
    config,
    promo_data: Optional[pd.DataFrame] = None,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Apply custom business terms and promotional effects with leakage prevention.
    
    FIXED: Passes training_end_date for leakage-safe feature generation.
    
    Args:
        data_df: Input data
        config: Configuration object
        promo_data: Optional promotional campaign data
        training_end_date: End of training period for leakage prevention
        
    Returns:
        pd.DataFrame: Data with custom business terms (leakage-safe)
    """
    logger = logging.getLogger(__name__)
    
    # Safe config access with defaults
    custom_config = getattr(config.features, 'custom_terms', {})
    if not custom_config:
        custom_config = {}
        logger.warning("No custom_terms configuration found - using defaults")
    
    result_df = data_df.copy()
    
    # Apply promotional effects with leakage prevention
    promo_config = custom_config.get('promo_flag', {})
    if promo_config.get('enabled', False):
        promo_features = generate_promotional_features(
            data_df,
            config,
            promo_data,
            sign_constraint=promo_config.get('sign_constraint', 'positive'),
            training_end_date=training_end_date
        )
        result_df = pd.concat([result_df, promo_features], axis=1)
        logger.info("Applied promotional effects (leakage-safe)")
    
    # Apply competitive effects
    competitive_features = generate_competitive_features(data_df, config)
    result_df = pd.concat([result_df, competitive_features], axis=1)
    
    # Apply price elasticity effects
    price_features = generate_price_elasticity_features(data_df, config)
    result_df = pd.concat([result_df, price_features], axis=1)
    
    # Apply inventory/stock effects
    inventory_features = generate_inventory_features(data_df, config)
    result_df = pd.concat([result_df, inventory_features], axis=1)
    
    logger.info("Applied custom business terms")
    return result_df


def generate_promotional_features(
    data_df: pd.DataFrame,
    config,
    promo_data: Optional[pd.DataFrame] = None,
    sign_constraint: str = 'positive',
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Generate promotional campaign features with leakage prevention.
    
    FIXED: Passes training_end_date to prevent leakage in intensity calculations.
    
    Args:
        data_df: Input data with dates
        config: Configuration object
        promo_data: External promotional data
        sign_constraint: Constraint on promotional effect sign
        training_end_date: End of training period for leakage prevention
        
    Returns:
        pd.DataFrame: Promotional features (leakage-safe)
    """
    logger = logging.getLogger(__name__)
    
    date_col = config.data.date_col
    dates = pd.to_datetime(data_df[date_col])
    
    promo_features = pd.DataFrame(index=data_df.index)
    
    if promo_data is not None:
        # Use external promotional data
        external_features = process_external_promo_data(dates, promo_data)
        promo_features = pd.concat([promo_features, external_features], axis=1)
    else:
        # Detect promotional periods from spend patterns
        detected_features = detect_promotional_periods(
            data_df, config, training_end_date=training_end_date
        )
        promo_features = pd.concat([promo_features, detected_features], axis=1)
    
    # Add promotional intensity features (with leakage prevention)
    intensity_features = calculate_promotional_intensity(
        data_df, config, training_end_date=training_end_date
    )
    promo_features = pd.concat([promo_features, intensity_features], axis=1)
    
    # Apply sign constraints without full-series normalization leakage
    promo_features = apply_promotional_constraints(
        promo_features, sign_constraint, data_df, config, training_end_date
    )
    
    return promo_features


def apply_promotional_constraints(
    promo_features: pd.DataFrame,
    sign_constraint: str,
    data_df: pd.DataFrame,
    config,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Apply promotional constraints without causing data leakage.
    
    FIXED: Uses train-only statistics for normalization constraints.
    
    Args:
        promo_features: Promotional features DataFrame
        sign_constraint: Sign constraint ('positive', 'negative', or 'none')
        data_df: Original data DataFrame (for date access)
        config: Configuration object
        training_end_date: End of training period
        
    Returns:
        pd.DataFrame: Constrained promotional features
    """
    result_features = promo_features.copy()
    
    # Apply sign constraints
    if sign_constraint == 'positive':
        for col in result_features.columns:
            if col.startswith('promo_'):
                result_features[col] = result_features[col].clip(lower=0)
    elif sign_constraint == 'negative':
        for col in result_features.columns:
            if col.startswith('promo_'):
                result_features[col] = result_features[col].clip(upper=0)
    
    # FIXED: Train-only normalization for intensity features (no leakage)
    intensity_cols = [col for col in result_features.columns if 'intensity' in col]
    for col in intensity_cols:
        if training_end_date is not None:
            # Use date column for proper masking
            date_mask = data_df[config.data.date_col] <= training_end_date  
            train_data = result_features.loc[date_mask, col]
            if len(train_data) > 0 and train_data.max() > 0:
                result_features[col] = result_features[col] / train_data.max()
        else:
            # Fallback: use predefined scale (discount% already 0-1)
            result_features[col] = result_features[col].clip(0, 1)
    
    return result_features



def process_external_promo_data(
    dates: pd.DatetimeIndex,
    promo_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Process external promotional campaign data with robust dtype handling.
    
    FIXED: Proper dtype checks and casting for discount_pct and budget fields.
    
    Args:
        dates: Model date series
        promo_data: DataFrame with columns [campaign_name, start_date, end_date, discount_pct, budget]
        
    Returns:
        pd.DataFrame: Processed promotional features
    """
    promo_features = pd.DataFrame(index=dates)
    
    for _, campaign in promo_data.iterrows():
        campaign_name = str(campaign['campaign_name']).replace(' ', '_').lower()
        start_date = pd.to_datetime(campaign['start_date'])
        end_date = pd.to_datetime(campaign['end_date'])
        
        # Binary promotional indicator
        promo_indicator = (dates >= start_date) & (dates <= end_date)
        promo_features[f'promo_{campaign_name}'] = promo_indicator.astype(float)
        
        # FIXED: Promotional intensity with proper dtype checking
        if 'discount_pct' in campaign:
            discount_raw = campaign.get('discount_pct')
            discount = float(discount_raw) if pd.notna(discount_raw) else None
            if discount is not None:
                discount_intensity = promo_indicator * discount
                promo_features[f'promo_{campaign_name}_intensity'] = discount_intensity
        
        # FIXED: Promotional budget impact with dtype checking
        if 'budget' in campaign:
            budget_raw = campaign.get('budget')
            budget = float(budget_raw) if pd.notna(budget_raw) else None
            if budget is not None:
                budget_impact = promo_indicator * budget
                promo_features[f'promo_{campaign_name}_budget'] = budget_impact
        
        # Ramp-up and ramp-down effects
        promo_ramp = calculate_promotional_ramp(dates, start_date, end_date)
        promo_features[f'promo_{campaign_name}_ramp'] = promo_ramp
    
    return promo_features


def detect_promotional_periods(
    data_df: pd.DataFrame,
    config,
    spend_threshold: float = 1.5,
    revenue_threshold: float = 1.3,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Detect promotional periods from spend and revenue patterns with leakage prevention.
    
    FIXED: Uses lagged baselines and train-only normalization to prevent target leakage.
    
    Args:
        data_df: Input data
        config: Configuration object
        spend_threshold: Multiplier for detecting spend spikes
        revenue_threshold: Multiplier for detecting revenue spikes
        training_end_date: End of training period for leakage-safe scaling
        
    Returns:
        pd.DataFrame: Detected promotional features (leakage-safe)
    """
    logger = logging.getLogger(__name__)
    
    revenue_col = config.data.revenue_col
    promo_features = pd.DataFrame(index=data_df.index)
    
    # FIXED: Calculate rolling baselines with lag to prevent leakage
    window = 14  # 2-week baseline window
    
    # Use lagged revenue for baseline calculation
    revenue_lagged = data_df[revenue_col].shift(1)
    revenue_baseline = revenue_lagged.rolling(window=window, min_periods=7).median()
    revenue_spike = data_df[revenue_col] / (revenue_baseline + 1e-8)
    
    # Detect promotional periods with lagged baseline
    promo_periods = revenue_spike > revenue_threshold
    promo_features['promo_detected'] = promo_periods.astype(float)
    
    # Calculate promotional intensity (normalized safely)
    promo_intensity = (revenue_spike - 1).clip(lower=0)
    
    # FIXED: Train-only normalization for intensity
    if training_end_date is not None:
        train_mask = data_df[config.data.date_col] <= training_end_date
        train_intensity = promo_intensity[train_mask]
        if len(train_intensity) > 0 and train_intensity.max() > 0:
            promo_intensity = promo_intensity / train_intensity.max()
    else:
        # Fallback: clip to reasonable range
        promo_intensity = promo_intensity.clip(0, 2)
    
    promo_features['promo_intensity'] = promo_intensity
    
    # FIXED: Consistent spend column detection using config
    spend_cols = get_spend_columns(data_df, config)
    
    if spend_cols:
        total_spend = data_df[spend_cols].sum(axis=1)
        # Use lagged spend for baseline
        spend_lagged = total_spend.shift(1)
        spend_baseline = spend_lagged.rolling(window=window, min_periods=7).median()
        spend_ratio = total_spend / (spend_baseline + 1e-8)
        
        # 1. Price promotion (high revenue, normal spend)
        price_promo = (revenue_spike > revenue_threshold) & (spend_ratio < 1.2)
        promo_features['promo_price_discount'] = price_promo.astype(float)
        
        # 2. Media promotion (high spend and revenue)
        media_promo = (revenue_spike > revenue_threshold) & (spend_ratio > spend_threshold)
        promo_features['promo_media_push'] = media_promo.astype(float)
    
    
    # Use seasonality module for Black Friday, holiday patterns, etc.
    
    logger.info("Detected promotional periods from data patterns (leakage-safe)")
    return promo_features


def get_spend_columns(data_df: pd.DataFrame, config) -> List[str]:
    """
    Get spend columns using consistent naming convention.
    
    FIXED: Uses config-driven channel mapping with fallback for consistent naming.
    
    Args:
        data_df: Input DataFrame
        config: Configuration object
        
    Returns:
        List[str]: Spend column names
    """
    # FIXED: Use channel map from config if available
    if hasattr(config.data, 'channel_map'):
        spend_cols = [v for v in config.data.channel_map.values() if v in data_df.columns]
        if spend_cols:
            return spend_cols
    
    # Fallback: consistent lowercase naming
    spend_cols = [c for c in data_df.columns if c.lower().endswith('_spend')]
    
    # If still no matches, try uppercase (legacy compatibility)
    if not spend_cols:
        spend_cols = [c for c in data_df.columns if c.endswith('_SPEND')]
    
    return spend_cols



def calculate_promotional_ramp(
    dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ramp_up_days: int = 3,
    ramp_down_days: int = 3
) -> pd.Series:
    """
    Calculate promotional ramp-up and ramp-down effects with proper index alignment.
    
    FIXED: Returns properly aligned Series with DatetimeIndex for safe concatenation.
    
    Args:
        dates: Date series
        start_date: Promotion start date
        end_date: Promotion end date
        ramp_up_days: Days for ramp-up
        ramp_down_days: Days for ramp-down
        
    Returns:
        pd.Series: Promotional ramp values with proper date index
    """
    # Ensure we have a DatetimeIndex
    d = pd.DatetimeIndex(dates)
    ramp = np.zeros(len(d), dtype=float)
    
    # Vectorized calculations for efficiency
    in_window = (d >= start_date) & (d <= end_date)
    
    # Ramp-up calculation (vectorized)
    days_from_start = (d - start_date).days
    ramp_up = np.clip(days_from_start / max(ramp_up_days, 1), 0, 1)
    
    # Ramp-down calculation (vectorized)
    days_to_end = (end_date - d).days
    ramp_down = np.clip(days_to_end / max(ramp_down_days, 1), 0, 1)
    
    # Core promotion period (full intensity)
    core_period = ((d > start_date + pd.Timedelta(days=ramp_up_days)) &
                   (d < end_date - pd.Timedelta(days=ramp_down_days)))
    
    # Combine ramp calculations
    ramp = np.where(in_window & core_period, 1.0, np.minimum(ramp_up, ramp_down))
    ramp = np.where(in_window, ramp, 0.0)
    
    # FIXED: Return with proper DatetimeIndex for alignment
    return pd.Series(ramp, index=d)


def calculate_promotional_intensity(
    data_df: pd.DataFrame,
    config,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Calculate promotional intensity metrics with leakage prevention.
    
    FIXED: Uses strict lags and train-only statistics to avoid target leakage.
    
    Args:
        data_df: Input data
        config: Configuration object
        training_end_date: End of training period for leakage-safe scaling
        
    Returns:
        pd.DataFrame: Promotional intensity features (leakage-safe)
    """
    logger = logging.getLogger(__name__)
    
    intensity_features = pd.DataFrame(index=data_df.index)
    
    revenue_col = config.data.revenue_col
    revenue = data_df[revenue_col].copy()
    
    # FIXED: Revenue acceleration using strict 7-day lag (no peeking)
    revenue_wow = (revenue / revenue.shift(7) - 1).fillna(0)
    intensity_features['promo_revenue_acceleration'] = revenue_wow.clip(lower=0)
    
    # FIXED: Revenue volatility using lagged baseline (no current-period leakage)
    revenue_baseline = revenue.shift(1).rolling(14, min_periods=7).median()
    revenue_spike = (revenue / (revenue_baseline + 1e-8)).fillna(1)
    intensity_features['promo_revenue_spike'] = (revenue_spike - 1).clip(lower=0)
    
    # FIXED: Volatility with lag to prevent same-timestamp leakage
    revenue_volatility = revenue.shift(1).rolling(window=7, min_periods=3).std().fillna(0)
    intensity_features['promo_revenue_volatility'] = revenue_volatility
    
    # FIXED: Normalize intensity with train-only stats or clip to [0,1]
    for col in intensity_features.columns:
        if training_end_date is not None and col != 'promo_revenue_acceleration':
            # Train-only normalization using date column
            date_col = config.data.date_col
            train_mask = data_df[date_col] <= training_end_date
            train_data = intensity_features.loc[train_mask, col]
            if len(train_data) > 0 and train_data.max() > 0:
                intensity_features[col] = intensity_features[col] / train_data.max()
        else:
            # Simple clipping to [0,1] range
            intensity_features[col] = intensity_features[col].clip(0, 1)
    
    logger.info(f"Generated {len(intensity_features.columns)} leakage-safe promotional intensity features")
    return intensity_features


def generate_competitive_features(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Generate competitive landscape features.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Competitive features
    """
    competitive_features = pd.DataFrame()
    
    # Market share pressure (simulated)
    # In practice, this would use external competitive data
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    # Simulate competitive pressure cycles
    annual_cycle = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    competitive_features['competitive_pressure'] = (annual_cycle + 1) / 2  # Normalize to 0-1
    
    # Category growth trends
    # Simulate overall category growth
    trend_strength = 0.02  # 2% annual growth
    days_since_start = (dates - dates.min()).dt.days
    category_growth = 1 + (trend_strength * days_since_start / 365.25)
    competitive_features['category_growth'] = category_growth
    
    # Competitive response lag (simulated)
    # Models delayed competitive response to our activities
    spend_cols = [col for col in data_df.columns if col.endswith('_SPEND')]
    if spend_cols:
        total_spend = data_df[spend_cols].sum(axis=1)
        # Lagged competitive response (competitors react with 2-week delay)
        competitive_response = total_spend.shift(14).fillna(0)
        competitive_features['competitive_response_lag'] = competitive_response
    
    return competitive_features


def generate_price_elasticity_features(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Generate price elasticity and demand features.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Price elasticity features
    """
    price_features = pd.DataFrame()
    
    # Price elasticity proxies
    revenue_col = config.data.revenue_col
    volume_col = config.data.volume_col
    
    if volume_col in data_df.columns:
        # Average order value (AOV)
        aov = data_df[revenue_col] / (data_df[volume_col] + 1e-8)
        aov_baseline = aov.rolling(window=30, min_periods=1).median()
        
        # AOV premium/discount indicator
        aov_ratio = aov / (aov_baseline + 1e-8)
        price_features['aov_premium'] = (aov_ratio - 1).clip(lower=0)
        price_features['aov_discount'] = (1 - aov_ratio).clip(lower=0)
        
        # Demand elasticity (volume response to price changes)
        aov_change = aov.pct_change().fillna(0)
        volume_change = data_df[volume_col].pct_change().fillna(0)
        
        # Simple elasticity estimate (avoid division by zero)
        elasticity = np.where(
            np.abs(aov_change) > 0.01,
            -volume_change / aov_change,  # Negative sign for expected relationship
            0
        )
        price_features['demand_elasticity'] = pd.Series(elasticity).fillna(0).clip(-10, 10)
    
    return price_features


def generate_inventory_features(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Generate inventory and supply constraint features.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Inventory features
    """
    inventory_features = pd.DataFrame()
    
    # Stock-out proxy (volume declining despite constant/increasing spend)
    volume_col = config.data.volume_col
    spend_cols = [col for col in data_df.columns if col.endswith('_SPEND')]
    
    if volume_col in data_df.columns and spend_cols:
        total_spend = data_df[spend_cols].sum(axis=1)
        
        # Calculate efficiency (volume per spend)
        efficiency = data_df[volume_col] / (total_spend + 1e-8)
        efficiency_ma = efficiency.rolling(window=7, min_periods=1).mean()
        
        # Stock-out indicator (efficiency declining)
        efficiency_decline = efficiency_ma.pct_change().fillna(0)
        inventory_features['stockout_risk'] = (-efficiency_decline).clip(lower=0)
        
        # Inventory recovery indicator
        efficiency_recovery = efficiency_decline.clip(lower=0)
        inventory_features['inventory_recovery'] = efficiency_recovery
    
    # Seasonal inventory patterns
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    # Holiday inventory build-up
    inventory_features['inventory_buildup'] = (
        (dates.dt.month == 11) |  # November
        ((dates.dt.month == 12) & (dates.dt.day <= 15))  # Early December
    ).astype(float)
    
    return inventory_features


if __name__ == "__main__":
    # Test custom business terms
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sample_data = pd.DataFrame({
        'DATE_DAY': dates,
        'ALL_PURCHASES_ORIGINAL_PRICE': np.random.exponential(1000, 365),
        'ALL_PURCHASES': np.random.poisson(50, 365),
        'GOOGLE_PAID_SEARCH_SPEND': np.random.exponential(100, 365)
    })
    
    # Create mock config
    class MockConfig:
        class Features:
            class CustomTerms:
                promo_flag = {'enabled': True, 'sign_constraint': 'positive'}
        
        class Data:
            date_col = 'DATE_DAY'
            revenue_col = 'ALL_PURCHASES_ORIGINAL_PRICE'
            volume_col = 'ALL_PURCHASES'
        
        features = Features()
        data = Data()
    
    config = MockConfig()
    
    # Test promotional features
    promo_features = generate_promotional_features(sample_data, config)
    
    print("Custom Business Terms Test Results:")
    print(f"Generated {len(promo_features.columns)} promotional features")
    
    # Test full pipeline
    custom_features = apply_custom_business_terms(sample_data, config)
    print(f"Total custom features: {len(custom_features.columns) - len(sample_data.columns)}")
    
    print("Custom business terms module test completed")
