"""
Competitor and external factors module.

Implements competitive pressure modeling and external factor
integration for MMM.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_competitor_factors(
    data_df: pd.DataFrame,
    config,
    external_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Apply competitive and external factor modeling.
    
    Args:
        data_df: Input data
        config: Configuration object
        external_data: External datasets (competitor spend, market data, etc.)
        
    Returns:
        pd.DataFrame: Data with competitive factors
    """
    logger = logging.getLogger(__name__)
    
    result_df = data_df.copy()
    
    # Apply competitive pressure modeling
    competitive_features = generate_competitive_pressure_features(
        data_df, config, external_data
    )
    result_df = pd.concat([result_df, competitive_features], axis=1)
    
    # Apply market dynamics
    market_features = generate_market_dynamics_features(
        data_df, config, external_data
    )
    result_df = pd.concat([result_df, market_features], axis=1)
    
    # Apply external economic factors
    economic_features = generate_economic_factors(
        data_df, config, external_data
    )
    result_df = pd.concat([result_df, economic_features], axis=1)
    
    # Apply industry/category trends
    industry_features = generate_industry_trends(
        data_df, config, external_data
    )
    result_df = pd.concat([result_df, industry_features], axis=1)
    
    logger.info("Applied competitive and external factors")
    return result_df


def generate_competitive_pressure_features(
    data_df: pd.DataFrame,
    config,
    external_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Generate competitive pressure modeling features.
    
    Args:
        data_df: Input data
        config: Configuration object
        external_data: External competitive data
        
    Returns:
        pd.DataFrame: Competitive pressure features
    """
    logger = logging.getLogger(__name__)
    
    competitive_features = pd.DataFrame()
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    if external_data and 'competitor_spend' in external_data:
        # Use external competitor spend data
        competitor_df = external_data['competitor_spend']
        competitive_features = process_competitor_spend_data(dates, competitor_df)
    else:
        # Simulate competitive pressure from market patterns
        competitive_features = simulate_competitive_pressure(data_df, config)
    
    # Calculate competitive response metrics
    response_features = calculate_competitive_response_metrics(data_df, config)
    competitive_features = pd.concat([competitive_features, response_features], axis=1)
    
    # Add competitive timing effects
    timing_features = calculate_competitive_timing_effects(data_df, config)
    competitive_features = pd.concat([competitive_features, timing_features], axis=1)
    
    logger.info(f"Generated {len(competitive_features.columns)} competitive pressure features")
    return competitive_features


def process_competitor_spend_data(
    dates: pd.DatetimeIndex,
    competitor_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process external competitor spend data.
    
    Args:
        dates: Model date series
        competitor_df: DataFrame with competitor spend by channel
        
    Returns:
        pd.DataFrame: Processed competitive features
    """
    competitive_features = pd.DataFrame()
    
    # Ensure competitor data has date column
    if 'date' in competitor_df.columns:
        competitor_df['date'] = pd.to_datetime(competitor_df['date'])
        
        # Merge competitor data with model dates
        date_df = pd.DataFrame({'date': dates})
        merged_df = pd.merge(date_df, competitor_df, on='date', how='left')
        
        # Fill missing values with interpolation
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')
        merged_df = merged_df.fillna(0)
        
        # Calculate total competitive spend
        spend_cols = [col for col in merged_df.columns if 'spend' in col.lower()]
        if spend_cols:
            competitive_features['competitor_total_spend'] = merged_df[spend_cols].sum(axis=1)
            
            # Calculate share of voice
            if 'ALL_MEDIA_SPEND' in merged_df.columns:
                total_market_spend = merged_df['ALL_MEDIA_SPEND'] + competitive_features['competitor_total_spend']
                competitive_features['share_of_voice'] = (
                    merged_df['ALL_MEDIA_SPEND'] / (total_market_spend + 1e-8)
                )
            
            # Channel-specific competitive pressure
            for col in spend_cols:
                channel_name = col.replace('_spend', '').replace('competitor_', '')
                competitive_features[f'competitive_pressure_{channel_name}'] = merged_df[col]
        
        # Calculate competitive indices
        competitive_indices = calculate_competitive_indices(merged_df)
        competitive_features = pd.concat([competitive_features, competitive_indices], axis=1)
    
    return competitive_features


def simulate_competitive_pressure(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Simulate competitive pressure from observed market patterns.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Simulated competitive features
    """
    competitive_features = pd.DataFrame()
    dates = pd.to_datetime(data_df[config.data.date_col])
    n_periods = len(dates)
    
    # Market maturity factor (markets become more competitive over time)
    days_since_start = (dates - dates.min()).dt.days
    market_maturity = 1 + (days_since_start / 365.25) * 0.1  # 10% increase per year
    competitive_features['market_maturity'] = market_maturity
    
    # Seasonal competitive cycles
    # Higher competition during peak seasons
    peak_seasons = (
        (dates.dt.month.isin([11, 12])) |  # Holiday season
        (dates.dt.month.isin([3, 4])) |   # Spring
        (dates.dt.month.isin([8, 9]))     # Back-to-school
    )
    
    base_competition = 0.3
    peak_competition = 0.7
    competitive_intensity = np.where(peak_seasons, peak_competition, base_competition)
    competitive_features['seasonal_competition'] = competitive_intensity
    
    # Response to our own activity (competitors react to our spend)
    spend_cols = [col for col in data_df.columns if col.endswith('_SPEND')]
    if spend_cols:
        our_spend = data_df[spend_cols].sum(axis=1)
        
        # Competitive response with lag (2-week delayed response)
        competitive_response = our_spend.shift(14).fillna(0)
        # Normalize by rolling average
        response_baseline = competitive_response.rolling(window=30, min_periods=1).mean()
        normalized_response = competitive_response / (response_baseline + 1e-8)
        competitive_features['competitive_response_intensity'] = normalized_response
        
        # Market pressure based on spend volatility
        spend_volatility = our_spend.rolling(window=7).std().fillna(0)
        competitive_features['market_pressure'] = spend_volatility / (our_spend.rolling(window=30).mean() + 1e-8)
    
    # Random competitive shocks (unexpected competitor campaigns)
    np.random.seed(42)  # For reproducibility
    shock_probability = 0.05  # 5% chance per period
    shock_magnitude = np.random.exponential(0.5, n_periods)
    shock_indicator = np.random.random(n_periods) < shock_probability
    competitive_shocks = shock_indicator * shock_magnitude
    competitive_features['competitive_shocks'] = competitive_shocks
    
    return competitive_features


def calculate_competitive_response_metrics(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Calculate competitive response and elasticity metrics.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Competitive response metrics
    """
    response_features = pd.DataFrame()
    
    revenue_col = config.data.revenue_col
    volume_col = config.data.volume_col
    
    # Market share erosion indicator
    # Based on revenue declining while spend increases
    spend_cols = [col for col in data_df.columns if col.endswith('_SPEND')]
    if spend_cols:
        total_spend = data_df[spend_cols].sum(axis=1)
        
        # Calculate efficiency (revenue per spend)
        efficiency = data_df[revenue_col] / (total_spend + 1e-8)
        efficiency_trend = efficiency.rolling(window=14).apply(
            lambda x: stats.linregress(range(len(x)), x)[0], raw=False
        ).fillna(0)
        
        # Market share erosion (efficiency declining)
        response_features['market_share_erosion'] = (-efficiency_trend).clip(lower=0)
        
        # Competitive response strength
        # Measured as unexpected drops in performance
        efficiency_baseline = efficiency.rolling(window=30, min_periods=1).median()
        efficiency_gap = (efficiency_baseline - efficiency) / (efficiency_baseline + 1e-8)
        response_features['competitive_response_strength'] = efficiency_gap.clip(lower=0)
    
    # Price war indicator
    if volume_col in data_df.columns:
        # AOV declining while volume increases (price competition)
        aov = data_df[revenue_col] / (data_df[volume_col] + 1e-8)
        aov_change = aov.pct_change(periods=7).fillna(0)
        volume_change = data_df[volume_col].pct_change(periods=7).fillna(0)
        
        price_war_indicator = (aov_change < -0.05) & (volume_change > 0.05)
        response_features['price_war_indicator'] = price_war_indicator.astype(float)
    
    return response_features


def calculate_competitive_timing_effects(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Calculate competitive timing and sequencing effects.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Competitive timing features
    """
    timing_features = pd.DataFrame()
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    # First-mover advantage windows
    # Higher effectiveness at beginning of months/quarters
    timing_features['first_mover_monthly'] = (dates.dt.day <= 7).astype(float)
    timing_features['first_mover_quarterly'] = (
        (dates.dt.month.isin([1, 4, 7, 10])) & (dates.dt.day <= 14)
    ).astype(float)
    
    # Competitive crowding (end of periods when everyone spends)
    timing_features['competitive_crowding_monthly'] = (dates.dt.day >= 25).astype(float)
    timing_features['competitive_crowding_quarterly'] = (
        (dates.dt.month.isin([3, 6, 9, 12])) & (dates.dt.day >= 20)
    ).astype(float)
    
    # Holiday competitive intensity
    # Model periods with expected high competitive activity
    major_holidays = get_major_holiday_periods(dates)
    timing_features['holiday_competitive_intensity'] = major_holidays
    
    return timing_features


def get_major_holiday_periods(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Identify major holiday periods with high competitive activity.
    
    Args:
        dates: Date series
        
    Returns:
        pd.Series: Holiday competitive intensity
    """
    holiday_intensity = np.zeros(len(dates))
    
    for i, date in enumerate(dates):
        year = date.year
        month = date.month
        day = date.day
        
        # Black Friday week (highest competition)
        thanksgiving = pd.to_datetime(f'{year}-11-01') + pd.DateOffset(weeks=3)
        thanksgiving += pd.DateOffset(days=(3 - thanksgiving.weekday()) % 7)
        black_friday_week = (date >= thanksgiving) & (date <= thanksgiving + pd.DateOffset(days=6))
        
        if black_friday_week:
            holiday_intensity[i] = 1.0
        
        # December holiday season
        elif month == 12 and day >= 15:
            holiday_intensity[i] = 0.8
        
        # Valentine's Day week
        elif month == 2 and 10 <= day <= 16:
            holiday_intensity[i] = 0.6
        
        # Mother's Day week (second Sunday in May)
        elif month == 5 and 8 <= day <= 14:
            holiday_intensity[i] = 0.6
        
        # Back-to-school period
        elif month == 8 or (month == 9 and day <= 15):
            holiday_intensity[i] = 0.5
        
        # Summer kickoff (Memorial Day week)
        elif month == 5 and day >= 25:
            holiday_intensity[i] = 0.4
    
    return pd.Series(holiday_intensity, index=range(len(dates)))


def calculate_competitive_indices(competitor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sophisticated competitive indices.
    
    Args:
        competitor_df: Competitor data with spend columns
        
    Returns:
        pd.DataFrame: Competitive indices
    """
    indices = pd.DataFrame()
    
    spend_cols = [col for col in competitor_df.columns if 'spend' in col.lower()]
    
    if len(spend_cols) > 1:
        spend_data = competitor_df[spend_cols].fillna(0)
        
        # Competitive concentration index (Herfindahl-Hirschman Index)
        total_spend = spend_data.sum(axis=1)
        market_shares = spend_data.div(total_spend + 1e-8, axis=0)
        hhi = (market_shares ** 2).sum(axis=1)
        indices['competitive_concentration'] = hhi
        
        # Competitive diversity index
        # Higher values indicate more diverse competitive landscape
        from scipy.stats import entropy
        competitive_diversity = market_shares.apply(
            lambda row: entropy(row[row > 0]), axis=1
        ).fillna(0)
        indices['competitive_diversity'] = competitive_diversity
        
        # Dominant player index (largest competitor share)
        dominant_share = market_shares.max(axis=1)
        indices['dominant_competitor_share'] = dominant_share
        
        # Competitive activity level (total market spend)
        indices['total_market_activity'] = total_spend
    
    return indices


def generate_market_dynamics_features(
    data_df: pd.DataFrame,
    config,
    external_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Generate market dynamics and evolution features.
    
    Args:
        data_df: Input data
        config: Configuration object
        external_data: External market data
        
    Returns:
        pd.DataFrame: Market dynamics features
    """
    market_features = pd.DataFrame()
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    # Market lifecycle stage
    market_lifecycle = calculate_market_lifecycle_stage(dates)
    market_features = pd.concat([market_features, market_lifecycle], axis=1)
    
    # Market saturation indicators
    saturation_features = calculate_market_saturation_features(data_df, config)
    market_features = pd.concat([market_features, saturation_features], axis=1)
    
    # Channel evolution patterns
    channel_evolution = calculate_channel_evolution_patterns(data_df, config)
    market_features = pd.concat([market_features, channel_evolution], axis=1)
    
    # External market shocks
    if external_data and 'market_events' in external_data:
        shock_features = process_market_shock_events(dates, external_data['market_events'])
        market_features = pd.concat([market_features, shock_features], axis=1)
    
    return market_features


def calculate_market_lifecycle_stage(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate market lifecycle stage indicators.
    
    Args:
        dates: Date series
        
    Returns:
        pd.DataFrame: Market lifecycle features
    """
    lifecycle_features = pd.DataFrame()
    
    # Time since market entry (assuming data start = market entry)
    days_since_entry = (dates - dates.min()).dt.days
    years_since_entry = days_since_entry / 365.25
    
    # Market stage based on time since entry
    # Introduction: 0-1 years, Growth: 1-3 years, Maturity: 3+ years
    lifecycle_features['market_introduction'] = (years_since_entry <= 1).astype(float)
    lifecycle_features['market_growth'] = (
        (years_since_entry > 1) & (years_since_entry <= 3)
    ).astype(float)
    lifecycle_features['market_maturity'] = (years_since_entry > 3).astype(float)
    
    # Continuous lifecycle progression
    lifecycle_features['market_age'] = years_since_entry
    
    return lifecycle_features


def calculate_market_saturation_features(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Calculate market saturation indicators.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Market saturation features
    """
    saturation_features = pd.DataFrame()
    
    revenue_col = config.data.revenue_col
    spend_cols = [col for col in data_df.columns if col.endswith('_SPEND')]
    
    if spend_cols:
        total_spend = data_df[spend_cols].sum(axis=1)
        
        # Diminishing returns indicator
        # Revenue growth rate decreasing while spend increases
        revenue_growth = data_df[revenue_col].pct_change(periods=7).fillna(0)
        spend_growth = total_spend.pct_change(periods=7).fillna(0)
        
        # Efficiency decline
        efficiency = data_df[revenue_col] / (total_spend + 1e-8)
        efficiency_trend = efficiency.rolling(window=30).apply(
            lambda x: stats.linregress(range(len(x)), x)[0], raw=False
        ).fillna(0)
        
        saturation_features['market_saturation'] = (-efficiency_trend).clip(lower=0)
        
        # Channel saturation by individual channels
        for spend_col in spend_cols:
            channel_name = spend_col.replace('_SPEND', '').lower()
            channel_efficiency = data_df[revenue_col] / (data_df[spend_col] + 1e-8)
            channel_trend = channel_efficiency.rolling(window=30).apply(
                lambda x: stats.linregress(range(len(x)), x)[0], raw=False
            ).fillna(0)
            saturation_features[f'{channel_name}_saturation'] = (-channel_trend).clip(lower=0)
    
    return saturation_features


def calculate_channel_evolution_patterns(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Calculate channel evolution and substitution patterns.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Channel evolution features
    """
    evolution_features = pd.DataFrame()
    
    spend_cols = [col for col in data_df.columns if col.endswith('_SPEND')]
    
    if len(spend_cols) > 1:
        spend_data = data_df[spend_cols].fillna(0)
        total_spend = spend_data.sum(axis=1)
        
        # Channel share evolution
        for spend_col in spend_cols:
            channel_name = spend_col.replace('_SPEND', '').lower()
            channel_share = spend_data[spend_col] / (total_spend + 1e-8)
            
            # Channel share trend
            share_trend = channel_share.rolling(window=30).apply(
                lambda x: stats.linregress(range(len(x)), x)[0], raw=False
            ).fillna(0)
            evolution_features[f'{channel_name}_share_trend'] = share_trend
            
            # Channel volatility
            share_volatility = channel_share.rolling(window=14).std().fillna(0)
            evolution_features[f'{channel_name}_volatility'] = share_volatility
        
        # Cross-channel correlation patterns
        correlation_features = calculate_channel_correlations(spend_data)
        evolution_features = pd.concat([evolution_features, correlation_features], axis=1)
    
    return evolution_features


def calculate_channel_correlations(spend_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling cross-channel correlations.
    
    Args:
        spend_data: Spend data by channel
        
    Returns:
        pd.DataFrame: Correlation features
    """
    correlation_features = pd.DataFrame()
    
    # Calculate rolling correlations with 30-day window
    window = 30
    channels = spend_data.columns
    
    # Average correlation with other channels
    for i, channel in enumerate(channels):
        correlations = []
        for j, other_channel in enumerate(channels):
            if i != j:
                rolling_corr = spend_data[channel].rolling(window=window).corr(
                    spend_data[other_channel]
                ).fillna(0)
                correlations.append(rolling_corr)
        
        if correlations:
            avg_correlation = pd.concat(correlations, axis=1).mean(axis=1)
            channel_name = channel.replace('_SPEND', '').lower()
            correlation_features[f'{channel_name}_avg_correlation'] = avg_correlation
    
    return correlation_features


def process_market_shock_events(
    dates: pd.DatetimeIndex,
    market_events_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process external market shock events.
    
    Args:
        dates: Model date series
        market_events_df: DataFrame with columns [event_date, event_type, magnitude, duration_days]
        
    Returns:
        pd.DataFrame: Market shock features
    """
    shock_features = pd.DataFrame()
    shock_features['market_shock'] = 0.0
    shock_features['market_shock_magnitude'] = 0.0
    shock_features['market_shock_type'] = 0  # Encoded shock type
    
    for _, event in market_events_df.iterrows():
        event_date = pd.to_datetime(event['event_date'])
        duration = event.get('duration_days', 1)
        magnitude = event.get('magnitude', 1.0)
        event_type = event.get('event_type', 'unknown')
        
        # Find dates affected by this event
        end_date = event_date + timedelta(days=duration)
        affected_mask = (dates >= event_date) & (dates <= end_date)
        
        if affected_mask.any():
            # Apply shock with decay over duration
            for i, date in enumerate(dates[affected_mask]):
                days_since_event = (date - event_date).days
                decay_factor = np.exp(-days_since_event / (duration / 3))  # 3x half-life
                
                shock_features.loc[affected_mask.tolist()[i], 'market_shock'] = 1.0
                shock_features.loc[affected_mask.tolist()[i], 'market_shock_magnitude'] = (
                    magnitude * decay_factor
                )
                
                # Encode shock type
                type_encoding = {
                    'economic_crisis': 1,
                    'regulatory_change': 2,
                    'new_technology': 3,
                    'pandemic': 4,
                    'supply_shortage': 5
                }.get(event_type, 0)
                shock_features.loc[affected_mask.tolist()[i], 'market_shock_type'] = type_encoding
    
    return shock_features


def generate_economic_factors(
    data_df: pd.DataFrame,
    config,
    external_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Generate economic factor features.
    
    Args:
        data_df: Input data
        config: Configuration object
        external_data: External economic data
        
    Returns:
        pd.DataFrame: Economic factor features
    """
    economic_features = pd.DataFrame()
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    if external_data and 'economic_indicators' in external_data:
        # Use external economic data
        econ_df = external_data['economic_indicators']
        economic_features = process_economic_indicators(dates, econ_df)
    else:
        # Simulate economic cycles
        economic_features = simulate_economic_cycles(dates)
    
    # Add consumer confidence proxies
    confidence_features = calculate_consumer_confidence_proxies(data_df, config)
    economic_features = pd.concat([economic_features, confidence_features], axis=1)
    
    return economic_features


def process_economic_indicators(
    dates: pd.DatetimeIndex,
    econ_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process external economic indicator data.
    
    Args:
        dates: Model date series
        econ_df: Economic indicators DataFrame
        
    Returns:
        pd.DataFrame: Processed economic features
    """
    economic_features = pd.DataFrame()
    
    if 'date' in econ_df.columns:
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        
        # Merge with model dates
        date_df = pd.DataFrame({'date': dates})
        merged_df = pd.merge(date_df, econ_df, on='date', how='left')
        
        # Interpolate missing values
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')
        merged_df = merged_df.fillna(method='bfill').fillna(method='ffill')
        
        # Key economic indicators
        indicator_cols = ['gdp_growth', 'unemployment_rate', 'inflation_rate', 
                         'consumer_confidence', 'interest_rate']
        
        for col in indicator_cols:
            if col in merged_df.columns:
                economic_features[f'econ_{col}'] = merged_df[col]
                
                # Add trend and volatility
                trend = merged_df[col].rolling(window=30).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0], raw=False
                ).fillna(0)
                economic_features[f'econ_{col}_trend'] = trend
                
                volatility = merged_df[col].rolling(window=14).std().fillna(0)
                economic_features[f'econ_{col}_volatility'] = volatility
    
    return economic_features


def simulate_economic_cycles(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Simulate economic cycles and indicators.
    
    Args:
        dates: Date series
        
    Returns:
        pd.DataFrame: Simulated economic features
    """
    economic_features = pd.DataFrame()
    
    # Business cycle (7-year cycle)
    days_since_start = (dates - dates.min()).dt.days
    business_cycle = np.sin(2 * np.pi * days_since_start / (365.25 * 7))
    economic_features['business_cycle'] = business_cycle
    
    # Seasonal economic patterns
    seasonal_cycle = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    economic_features['seasonal_economic_cycle'] = seasonal_cycle
    
    # Consumer spending seasonality
    # Higher spending in Q4, lower in Q1
    quarter_effects = np.zeros(len(dates))
    for i, date in enumerate(dates):
        if date.month in [10, 11, 12]:  # Q4
            quarter_effects[i] = 0.3
        elif date.month in [1, 2, 3]:   # Q1
            quarter_effects[i] = -0.2
        elif date.month in [4, 5, 6]:   # Q2
            quarter_effects[i] = 0.1
        else:  # Q3
            quarter_effects[i] = 0.0
    
    economic_features['consumer_spending_seasonality'] = quarter_effects
    
    # Economic uncertainty (simulated volatility spikes)
    np.random.seed(42)
    uncertainty_base = 0.1
    uncertainty_spikes = np.random.exponential(0.05, len(dates))
    spike_probability = np.random.random(len(dates)) < 0.02  # 2% chance
    uncertainty = uncertainty_base + (uncertainty_spikes * spike_probability)
    economic_features['economic_uncertainty'] = uncertainty
    
    return economic_features


def calculate_consumer_confidence_proxies(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Calculate consumer confidence proxies from data patterns.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Consumer confidence features
    """
    confidence_features = pd.DataFrame()
    
    revenue_col = config.data.revenue_col
    volume_col = config.data.volume_col
    
    # Purchase frequency trends (higher frequency = higher confidence)
    if volume_col in data_df.columns:
        volume_trend = data_df[volume_col].rolling(window=30).apply(
            lambda x: stats.linregress(range(len(x)), x)[0], raw=False
        ).fillna(0)
        confidence_features['consumer_confidence_proxy'] = volume_trend
        
        # Purchase size consistency (less volatility = higher confidence)
        if revenue_col in data_df.columns:
            aov = data_df[revenue_col] / (data_df[volume_col] + 1e-8)
            aov_volatility = aov.rolling(window=14).std().fillna(0)
            confidence_features['purchase_uncertainty'] = aov_volatility
    
    # Spending resilience (maintaining spend during downturns)
    spend_cols = [col for col in data_df.columns if col.endswith('_SPEND')]
    if spend_cols:
        total_spend = data_df[spend_cols].sum(axis=1)
        spend_stability = 1 / (total_spend.rolling(window=14).std().fillna(1) + 1e-8)
        confidence_features['spending_resilience'] = spend_stability
    
    return confidence_features


def generate_industry_trends(
    data_df: pd.DataFrame,
    config,
    external_data: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Generate industry and category trend features.
    
    Args:
        data_df: Input data
        config: Configuration object
        external_data: External industry data
        
    Returns:
        pd.DataFrame: Industry trend features
    """
    industry_features = pd.DataFrame()
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    # Technology adoption trends
    tech_trends = calculate_technology_adoption_trends(dates)
    industry_features = pd.concat([industry_features, tech_trends], axis=1)
    
    # Category evolution patterns
    category_evolution = calculate_category_evolution_patterns(data_df, config)
    industry_features = pd.concat([industry_features, category_evolution], axis=1)
    
    # Regulatory environment changes
    regulatory_features = simulate_regulatory_environment(dates)
    industry_features = pd.concat([industry_features, regulatory_features], axis=1)
    
    return industry_features


def calculate_technology_adoption_trends(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate technology adoption S-curves.
    
    Args:
        dates: Date series
        
    Returns:
        pd.DataFrame: Technology trend features
    """
    tech_features = pd.DataFrame()
    
    days_since_start = (dates - dates.min()).dt.days
    years_since_start = days_since_start / 365.25
    
    # Digital adoption S-curve
    # Rapid adoption in years 1-5, then saturation
    digital_adoption = 1 / (1 + np.exp(-2 * (years_since_start - 2.5)))
    tech_features['digital_adoption'] = digital_adoption
    
    # Mobile-first trend
    mobile_adoption = 1 / (1 + np.exp(-3 * (years_since_start - 1.5)))
    tech_features['mobile_adoption'] = mobile_adoption
    
    # AI/automation trend (later adoption)
    ai_adoption = 1 / (1 + np.exp(-2 * (years_since_start - 4)))
    tech_features['ai_adoption'] = ai_adoption
    
    return tech_features


def calculate_category_evolution_patterns(
    data_df: pd.DataFrame,
    config
) -> pd.DataFrame:
    """
    Calculate category-specific evolution patterns.
    
    Args:
        data_df: Input data
        config: Configuration object
        
    Returns:
        pd.DataFrame: Category evolution features
    """
    evolution_features = pd.DataFrame()
    
    revenue_col = config.data.revenue_col
    dates = pd.to_datetime(data_df[config.data.date_col])
    
    # Category maturity indicators
    # Based on revenue growth patterns
    revenue_growth = data_df[revenue_col].pct_change(periods=30).fillna(0)
    
    # Growth stage: high, consistent growth
    growth_stage = (revenue_growth > 0.05) & (revenue_growth < 0.3)
    evolution_features['category_growth_stage'] = growth_stage.astype(float)
    
    # Maturity stage: stable, low growth
    maturity_stage = (revenue_growth >= -0.02) & (revenue_growth <= 0.05)
    evolution_features['category_maturity_stage'] = maturity_stage.astype(float)
    
    # Decline stage: negative growth
    decline_stage = revenue_growth < -0.02
    evolution_features['category_decline_stage'] = decline_stage.astype(float)
    
    # Innovation cycles (periodic disruption)
    # Assume 2-year innovation cycles
    days_since_start = (dates - dates.min()).dt.days
    innovation_cycle = np.sin(2 * np.pi * days_since_start / (365.25 * 2))
    evolution_features['innovation_cycle'] = (innovation_cycle + 1) / 2  # Normalize to 0-1
    
    return evolution_features


def simulate_regulatory_environment(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Simulate regulatory environment changes.
    
    Args:
        dates: Date series
        
    Returns:
        pd.DataFrame: Regulatory features
    """
    regulatory_features = pd.DataFrame()
    
    # Privacy regulation impact (increasing over time)
    days_since_start = (dates - dates.min()).dt.days
    years_since_start = days_since_start / 365.25
    
    # GDPR-like regulation impact (step function)
    gdpr_impact = (years_since_start >= 2).astype(float)
    regulatory_features['privacy_regulation_impact'] = gdpr_impact
    
    # Advertising restrictions (gradual increase)
    ad_restrictions = np.tanh(years_since_start - 1)  # Gradual increase starting year 1
    regulatory_features['advertising_restrictions'] = ad_restrictions.clip(0, 1)
    
    # Platform policy changes (random shocks)
    np.random.seed(42)
    policy_changes = np.random.exponential(0.1, len(dates))
    change_probability = np.random.random(len(dates)) < 0.01  # 1% chance
    regulatory_features['platform_policy_shocks'] = policy_changes * change_probability
    
    return regulatory_features


if __name__ == "__main__":
    # Test competitor factors module
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sample_data = pd.DataFrame({
        'DATE_DAY': dates,
        'ALL_PURCHASES_ORIGINAL_PRICE': np.random.exponential(1000, 365),
        'ALL_PURCHASES': np.random.poisson(50, 365),
        'GOOGLE_PAID_SEARCH_SPEND': np.random.exponential(100, 365),
        'FACEBOOK_SPEND': np.random.exponential(80, 365)
    })
    
    # Create mock config
    class MockConfig:
        class Data:
            date_col = 'DATE_DAY'
            revenue_col = 'ALL_PURCHASES_ORIGINAL_PRICE'
            volume_col = 'ALL_PURCHASES'
        
        data = Data()
    
    config = MockConfig()
    
    # Test competitive pressure features
    competitive_features = generate_competitive_pressure_features(sample_data, config)
    print("Competitor Factors Test Results:")
    print(f"Generated {len(competitive_features.columns)} competitive pressure features")
    
    # Test market dynamics
    market_features = generate_market_dynamics_features(sample_data, config)
    print(f"Generated {len(market_features.columns)} market dynamics features")
    
    # Test economic factors
    economic_features = generate_economic_factors(sample_data, config)
    print(f"Generated {len(economic_features.columns)} economic factor features")
    
    # Test full pipeline
    all_factors = apply_competitor_factors(sample_data, config)
    print(f"Total competitive/external features: {len(all_factors.columns) - len(sample_data.columns)}")
    
    print("Competitor factors module test completed")
