"""
Seasonality feature generation module.

Creates time-based features including trends, seasonality, and holiday effects.
Addresses key issues: trend leakage, holiday date alignment, multicollinearity,
cyclic encoding, observed holidays, and comprehensive holiday coverage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Set
import logging
from datetime import datetime, timedelta
import warnings


def validate_seasonality_params(config: Any) -> Dict[str, Any]:
    """
    Validate and normalize seasonality parameters.
    
    Args:
        config: Configuration object with seasonality settings
        
    Returns:
        Dict: Validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger = logging.getLogger(__name__)
    
    # Extract seasonality config
    if hasattr(config, 'features') and hasattr(config.features, 'seasonality'):
        seasonality_config = config.features.seasonality
    elif isinstance(config, dict) and 'seasonality' in config:
        seasonality_config = config['seasonality']
    else:
        raise ValueError("No seasonality configuration found")
    
    # Default parameters
    params = {
        'fourier_terms': 4,
        'weekly_fourier': True,
        'annual_fourier': True,
        'yearly_trend': True,
        'trend_normalization': 'none',
        'pattern_mode': 'fourier',
        'weekly_patterns': True,
        'monthly_patterns': True,
        'cyclic_encoding': True,
        'holiday_calendar': 'US',
        'include_observed': True,
        'holiday_grouping': 'individual',
        'holiday_effects': ['before', 'after'],
        'major_holidays_only': False,
        'return_date': False,
        'feature_validation': True,
        'extended_holidays': {},
        'holiday_groups': {}
    }
    
    # Update with config values
    for key, default_value in params.items():
        if hasattr(seasonality_config, key):
            params[key] = getattr(seasonality_config, key)
        elif isinstance(seasonality_config, dict) and key in seasonality_config:
            params[key] = seasonality_config[key]
        else:
            params[key] = default_value
    
    # Validation
    if params['fourier_terms'] < 1 or params['fourier_terms'] > 10:
        warnings.warn(f"fourier_terms={params['fourier_terms']} may be suboptimal. Recommend 2-6.")
    
    if params['pattern_mode'] not in ['fourier', 'categorical', 'mixed']:
        raise ValueError(f"pattern_mode must be 'fourier', 'categorical', or 'mixed', got {params['pattern_mode']}")
    
    if params['trend_normalization'] not in ['none', 'training_only']:
        raise ValueError(f"trend_normalization must be 'none' or 'training_only', got {params['trend_normalization']}")
    
    if params['holiday_grouping'] not in ['individual', 'grouped']:
        raise ValueError(f"holiday_grouping must be 'individual' or 'grouped', got {params['holiday_grouping']}")
    
    # Multicollinearity warning
    if (params['pattern_mode'] in ['categorical', 'mixed'] and 
        params['weekly_fourier'] and params['weekly_patterns']):
        warnings.warn("Weekly Fourier + day-of-week dummies create multicollinearity. Consider pattern_mode='fourier'.")
    
    logger.info(f"Validated seasonality params: {params['pattern_mode']} mode, {params['fourier_terms']} Fourier terms")
    
    return params


def apply_seasonality_from_config(
    dates: pd.Series,
    config: Any,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Apply seasonality features using configuration parameters.
    
    Args:
        dates: Date series
        config: Configuration object
        training_end_date: End date of training period (for trend normalization)
        
    Returns:
        pd.DataFrame: Seasonality features
    """
    params = validate_seasonality_params(config)
    
    return generate_seasonality_robust(
        dates=dates,
        fourier_terms=params['fourier_terms'],
        weekly_fourier=params['weekly_fourier'],
        annual_fourier=params['annual_fourier'],
        include_trend=params['yearly_trend'],
        trend_normalization=params['trend_normalization'],
        pattern_mode=params['pattern_mode'],
        weekly_patterns=params['weekly_patterns'],
        monthly_patterns=params['monthly_patterns'],
        cyclic_encoding=params['cyclic_encoding'],
        include_holidays=True,
        holiday_calendar=params['holiday_calendar'],
        include_observed=params['include_observed'],
        holiday_grouping=params['holiday_grouping'],
        holiday_effects=params['holiday_effects'],
        major_holidays_only=params['major_holidays_only'],
        extended_holidays=params['extended_holidays'],
        holiday_groups=params['holiday_groups'],
        return_date=params['return_date'],
        training_end_date=training_end_date
    )


def generate_seasonality_robust(
    dates: pd.Series,
    fourier_terms: int = 4,
    weekly_fourier: bool = True,
    annual_fourier: bool = True,
    include_trend: bool = True,
    trend_normalization: str = 'none',
    pattern_mode: str = 'fourier',
    weekly_patterns: bool = True,
    monthly_patterns: bool = True,
    cyclic_encoding: bool = True,
    include_holidays: bool = True,
    holiday_calendar: str = 'US',
    include_observed: bool = True,
    holiday_grouping: str = 'individual',
    holiday_effects: List[str] = ['before', 'after'],
    major_holidays_only: bool = False,
    extended_holidays: Dict[str, bool] = None,
    holiday_groups: Dict[str, List[str]] = None,
    return_date: bool = False,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Generate robust seasonality features addressing all key issues.
    
    Args:
        dates: Date series
        fourier_terms: Number of Fourier components
        weekly_fourier: Include weekly Fourier components
        annual_fourier: Include annual Fourier components  
        include_trend: Include trend features
        trend_normalization: Trend normalization strategy
        pattern_mode: Feature selection mode to prevent multicollinearity
        weekly_patterns: Include day-of-week features
        monthly_patterns: Include month-of-year features
        cyclic_encoding: Use sin/cos encoding for cyclic features
        include_holidays: Include holiday features
        holiday_calendar: Holiday calendar to use
        include_observed: Handle observed holidays
        holiday_grouping: Holiday grouping strategy
        holiday_effects: Holiday spillover effects
        major_holidays_only: Use only major holidays
        extended_holidays: Extended holiday configuration
        holiday_groups: Holiday group definitions
        return_date: Include date column in output
        training_end_date: End of training period for normalization
        
    Returns:
        pd.DataFrame: Seasonality features
    """
    logger = logging.getLogger(__name__)
    
    # Initialize results with proper index
    n_dates = len(dates)
    if return_date:
        features_df = pd.DataFrame({'date': dates}, index=range(n_dates))
    else:
        features_df = pd.DataFrame(index=range(n_dates))
    
    # Generate Fourier seasonality features (if enabled)
    if annual_fourier or weekly_fourier:
        fourier_features = generate_fourier_features_robust(
            dates, fourier_terms, weekly_fourier, annual_fourier
        )
        fourier_features.index = range(n_dates)  # Ensure consistent index
        features_df = pd.concat([features_df, fourier_features], axis=1)
    
    # Generate trend feature (fix trend leakage)
    if include_trend:
        trend_feature = generate_trend_feature_robust(
            dates, normalization=trend_normalization, training_end_date=training_end_date
        )
        trend_feature.index = range(n_dates)  # Ensure consistent index
        features_df = pd.concat([features_df, trend_feature], axis=1)
    
    # Generate pattern features (prevent multicollinearity)
    if pattern_mode in ['categorical', 'mixed']:
        # Day-of-week features (only if not conflicting with weekly Fourier)
        if weekly_patterns and not (weekly_fourier and pattern_mode != 'mixed'):
            dow_features = generate_dow_features_robust(dates)
            dow_features.index = range(n_dates)  # Ensure consistent index
            features_df = pd.concat([features_df, dow_features], axis=1)
        
        # Monthly features
        if monthly_patterns:
            monthly_features = generate_monthly_features_robust(dates, cyclic_encoding)
            monthly_features.index = range(n_dates)  # Ensure consistent index
            features_df = pd.concat([features_df, monthly_features], axis=1)
    
    # Generate holiday features (fix date alignment and expand coverage)
    if include_holidays:
        holiday_features = generate_holiday_features_robust(
            dates, 
            calendar=holiday_calendar,
            include_observed=include_observed,
            grouping=holiday_grouping,
            effects=holiday_effects,
            major_only=major_holidays_only,
            extended_holidays=extended_holidays or {},
            holiday_groups=holiday_groups or {}
        )
        holiday_features.index = range(n_dates)  # Ensure consistent index
        features_df = pd.concat([features_df, holiday_features], axis=1)
    
    # Generate additional time features (with proper cyclic encoding)
    time_features = create_time_features_robust(dates, cyclic_encoding)
    time_features.index = range(n_dates)  # Ensure consistent index
    features_df = pd.concat([features_df, time_features], axis=1)
    
    logger.info(f"Generated {len(features_df.columns)-(1 if return_date else 0)} seasonality features")
    
    return features_df


def generate_fourier_features_robust(
    dates: Union[pd.Series, pd.DatetimeIndex], 
    num_terms: int = 4,
    weekly: bool = True,
    annual: bool = True
) -> pd.DataFrame:
    """
    Generate Fourier seasonality features with proper controls.
    
    Args:
        dates: Date series or index
        num_terms: Number of Fourier terms to generate
        weekly: Include weekly components
        annual: Include annual components
        
    Returns:
        pd.DataFrame: Fourier features
    """
    dates_dt = pd.to_datetime(dates)
    fourier_features = {}
    
    if annual:
        # Annual seasonality (365.25 days)
        day_of_year = dates_dt.dt.dayofyear
        for i in range(1, num_terms + 1):
            fourier_features[f'sin_annual_{i}'] = np.sin(2 * np.pi * i * day_of_year / 365.25)
            fourier_features[f'cos_annual_{i}'] = np.cos(2 * np.pi * i * day_of_year / 365.25)
    
    if weekly:
        # Weekly seasonality (7 days) - limit terms to avoid over-parameterization
        day_of_week = dates_dt.dt.dayofweek
        weekly_terms = min(num_terms, 3)  # Limit weekly terms
        for i in range(1, weekly_terms + 1):
            fourier_features[f'sin_weekly_{i}'] = np.sin(2 * np.pi * i * day_of_week / 7)
            fourier_features[f'cos_weekly_{i}'] = np.cos(2 * np.pi * i * day_of_week / 7)
    
    return pd.DataFrame(fourier_features)


def generate_trend_feature_robust(
    dates: pd.Series,
    normalization: str = 'none',
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Generate trend feature without data leakage.
    
    Args:
        dates: Date series
        normalization: Normalization strategy ('none' or 'training_only')
        training_end_date: End of training period
        
    Returns:
        pd.DataFrame: Trend feature
    """
    dates_dt = pd.to_datetime(dates)
    min_date = dates_dt.min()
    
    # Create linear trend (days since first date) - no normalization by default
    trend = (dates_dt - min_date).dt.days.astype(float)
    
    if normalization == 'training_only' and training_end_date is not None:
        # Normalize only using training period to prevent leakage
        training_mask = dates_dt <= training_end_date
        if training_mask.any():
            training_max = trend[training_mask].max()
            if training_max > 0:
                trend = trend / training_max
        else:
            warnings.warn("No training data found for trend normalization")
    
    return pd.DataFrame({'trend': trend})


def generate_dow_features_robust(dates: Union[pd.Series, pd.DatetimeIndex]) -> pd.DataFrame:
    """
    Generate day-of-week features with weekend grouping.
    
    Args:
        dates: Date series or index
        
    Returns:
        pd.DataFrame: Day-of-week features
    """
    dates_dt = pd.to_datetime(dates)
    dow_features = {}
    
    # Create binary indicators for each day
    for i, day_name in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
        dow_features[f'is_{day_name}'] = (dates_dt.dt.dayofweek == i).astype(int)
    
    # Weekend indicator
    dow_features['is_weekend'] = dates_dt.dt.dayofweek.isin([5, 6]).astype(int)
    
    return pd.DataFrame(dow_features)


def generate_monthly_features_robust(dates: Union[pd.Series, pd.DatetimeIndex], cyclic_encoding: bool = True) -> pd.DataFrame:
    """
    Generate month-of-year features with proper cyclic encoding.
    
    Args:
        dates: Date series or index
        cyclic_encoding: Use sin/cos encoding for cyclic features
        
    Returns:
        pd.DataFrame: Monthly features
    """
    dates_dt = pd.to_datetime(dates)
    monthly_features = {}
    
    if cyclic_encoding:
        # Cyclic encoding for month (1-12)
        month = dates_dt.dt.month
        monthly_features['sin_month'] = np.sin(2 * np.pi * month / 12)
        monthly_features['cos_month'] = np.cos(2 * np.pi * month / 12)
        
        # Quarter encoding
        quarter = dates_dt.dt.quarter
        monthly_features['sin_quarter'] = np.sin(2 * np.pi * quarter / 4)
        monthly_features['cos_quarter'] = np.cos(2 * np.pi * quarter / 4)
    else:
        # One-hot encoding for months
        for month in range(1, 13):
            month_name = pd.Timestamp(f'2000-{month:02d}-01').strftime('%B').lower()
            monthly_features[f'is_{month_name}'] = (dates_dt.dt.month == month).astype(int)
        
        # Quarter indicators  
        for quarter in range(1, 5):
            monthly_features[f'is_q{quarter}'] = (dates_dt.dt.quarter == quarter).astype(int)
    
    return pd.DataFrame(monthly_features)


def generate_holiday_features_robust(
    dates: Union[pd.Series, pd.DatetimeIndex],
    calendar: str = 'US',
    include_observed: bool = True,
    grouping: str = 'individual',
    effects: List[str] = ['before', 'after'],
    major_only: bool = False,
    extended_holidays: Dict[str, bool] = None,
    holiday_groups: Dict[str, List[str]] = None
) -> pd.DataFrame:
    """
    Generate comprehensive holiday features with proper date alignment.
    
    Args:
        dates: Date series or index
        calendar: Holiday calendar to use
        include_observed: Handle observed holidays
        grouping: Holiday grouping strategy
        effects: Holiday spillover effects
        major_only: Use only major holidays
        extended_holidays: Extended holiday configuration
        holiday_groups: Holiday group definitions
        
    Returns:
        pd.DataFrame: Holiday features
    """
    # Convert to series if needed
    if isinstance(dates, pd.DatetimeIndex):
        dates_series = pd.Series(dates)
    else:
        dates_series = dates
        
    dates_dt = pd.to_datetime(dates_series)
    extended_holidays = extended_holidays or {}
    holiday_groups = holiday_groups or {}
    
    # Get base holidays
    if calendar == 'US':
        holiday_features = _get_us_holidays_robust(
            dates_series, include_observed, major_only, extended_holidays
        )
    else:
        holiday_features = {}
    
    # Add spillover effects using proper date arithmetic
    holiday_features = _add_holiday_effects_robust(
        holiday_features, dates_series, effects
    )
    
    # Apply grouping if requested
    if grouping == 'grouped' and holiday_groups:
        holiday_features = _group_holidays(holiday_features, holiday_groups)
    
    return pd.DataFrame(holiday_features)


def _get_us_holidays_robust(
    dates: pd.Series,
    include_observed: bool = True,
    major_only: bool = False,
    extended_holidays: Dict[str, bool] = None
) -> Dict[str, pd.Series]:
    """Get comprehensive US holiday indicators with observed date handling."""
    holidays = {}
    extended_holidays = extended_holidays or {}
    
    # Convert to datetime if needed
    if isinstance(dates, pd.DatetimeIndex):
        dates = pd.Series(dates)
    
    dates_dt = pd.to_datetime(dates)
    
    # Create date index for efficient lookups
    date_set = set(dates_dt.dt.date)
    
    for year in dates_dt.dt.year.unique():
        year_holidays = {}
        
        # Core holidays
        year_holidays['new_years'] = pd.Timestamp(f'{year}-01-01')
        year_holidays['july_4th'] = pd.Timestamp(f'{year}-07-04')
        year_holidays['christmas'] = pd.Timestamp(f'{year}-12-25')
        year_holidays['veterans_day'] = pd.Timestamp(f'{year}-11-11')
        
        # Floating holidays
        year_holidays['thanksgiving'] = _get_nth_weekday(year, 11, 3, 4)  # 4th Thursday
        year_holidays['memorial_day'] = _get_last_weekday(year, 5, 0)     # Last Monday
        year_holidays['labor_day'] = _get_nth_weekday(year, 9, 0, 1)      # 1st Monday
        year_holidays['mlk_day'] = _get_nth_weekday(year, 1, 0, 3)        # 3rd Monday  
        year_holidays['presidents_day'] = _get_nth_weekday(year, 2, 0, 3) # 3rd Monday
        year_holidays['columbus_day'] = _get_nth_weekday(year, 10, 0, 2)  # 2nd Monday
        
        # Extended holidays (if enabled)
        if extended_holidays.get('good_friday', False):
            easter_date = _get_easter(year)
            if easter_date:
                year_holidays['good_friday'] = easter_date - timedelta(days=2)
                
        if extended_holidays.get('easter', False):
            easter_date = _get_easter(year)
            if easter_date:
                year_holidays['easter'] = easter_date
                
        if extended_holidays.get('black_friday', False) and 'thanksgiving' in year_holidays:
            if year_holidays['thanksgiving']:
                year_holidays['black_friday'] = year_holidays['thanksgiving'] + timedelta(days=1)
                
        if extended_holidays.get('cyber_monday', False) and 'thanksgiving' in year_holidays:
            if year_holidays['thanksgiving']:
                year_holidays['cyber_monday'] = year_holidays['thanksgiving'] + timedelta(days=4)
        
        # Apply observed rules if enabled
        if include_observed:
            year_holidays = _apply_observed_rules(year_holidays)
        
        # Filter to major holidays if requested
        if major_only:
            major_set = {'new_years', 'july_4th', 'thanksgiving', 'christmas'}
            year_holidays = {k: v for k, v in year_holidays.items() if k in major_set}
    
        # Convert to indicators
        for holiday_name, holiday_date in year_holidays.items():
            if holiday_date and holiday_date.date() in date_set:
                if holiday_name not in holidays:
                    holidays[holiday_name] = pd.Series(0, index=dates.index)
                
                mask = dates_dt.dt.date == holiday_date.date()
                holidays[holiday_name].loc[mask] = 1
    
    # Initialize missing holidays as zeros
    for holiday_name in holidays:
        if holiday_name not in holidays:
            holidays[holiday_name] = pd.Series(0, index=dates.index)
    
    return holidays


def _apply_observed_rules(holidays: Dict[str, pd.Timestamp]) -> Dict[str, pd.Timestamp]:
    """Apply observed holiday rules for weekends."""
    observed_holidays = {}
    
    for name, date in holidays.items():
        if date is None:
            observed_holidays[name] = None
            continue
            
        # If holiday falls on Saturday, observe on Friday
        if date.weekday() == 5:  # Saturday
            observed_holidays[name] = date - timedelta(days=1)
        # If holiday falls on Sunday, observe on Monday  
        elif date.weekday() == 6:  # Sunday
            observed_holidays[name] = date + timedelta(days=1)
        else:
            observed_holidays[name] = date
    
    return observed_holidays


def _add_holiday_effects_robust(
    holiday_features: Dict[str, pd.Series],
    dates: Union[pd.Series, pd.DatetimeIndex],
    effects: List[str]
) -> Dict[str, pd.Series]:
    """Add holiday spillover effects using proper date arithmetic."""
    enhanced_features = holiday_features.copy()
    
    # Convert to datetime series if needed
    if isinstance(dates, pd.DatetimeIndex):
        dates_series = pd.Series(dates)
    else:
        dates_series = dates
    
    dates_dt = pd.to_datetime(dates_series)
    
    # Create date to index mapping for efficient lookup
    date_to_idx = {date.date(): idx for idx, date in enumerate(dates_dt)}
    
    for holiday_name, holiday_series in holiday_features.items():
        holiday_dates = dates_dt[holiday_series == 1]
        
        for effect in effects:
            effect_series = pd.Series(0, index=dates_series.index)
            
            for holiday_date in holiday_dates:
                if effect == 'before':
                    target_date = (holiday_date - timedelta(days=1)).date()
                elif effect == 'after':
                    target_date = (holiday_date + timedelta(days=1)).date()
                else:
                    continue
                
                # Find target date in our date series
                if target_date in date_to_idx:
                    effect_series.iloc[date_to_idx[target_date]] = 1
            
            enhanced_features[f'{holiday_name}_{effect}'] = effect_series
    
    return enhanced_features


def _group_holidays(
    holiday_features: Dict[str, pd.Series],
    holiday_groups: Dict[str, List[str]]
) -> Dict[str, pd.Series]:
    """Group holidays to reduce feature count."""
    grouped_features = {}
    used_holidays = set()
    
    # Create grouped features
    for group_name, holiday_list in holiday_groups.items():
        group_series = None
        
        for holiday_name in holiday_list:
            # Check for exact match or partial match (including effects)
            matching_holidays = [h for h in holiday_features.keys() 
                               if holiday_name in h and h not in used_holidays]
            
            for match in matching_holidays:
                if group_series is None:
                    group_series = holiday_features[match].copy()
                else:
                    group_series = group_series | holiday_features[match]
                used_holidays.add(match)
        
        if group_series is not None:
            grouped_features[f'holiday_group_{group_name}'] = group_series
    
    # Add ungrouped holidays
    for holiday_name, holiday_series in holiday_features.items():
        if holiday_name not in used_holidays:
            grouped_features[holiday_name] = holiday_series
    
    return grouped_features


def _get_nth_weekday(year: int, month: int, weekday: int, n: int) -> Optional[pd.Timestamp]:
    """Get the nth occurrence of a weekday in a month."""
    try:
        first_day = pd.Timestamp(f'{year}-{month:02d}-01')
        days_ahead = weekday - first_day.weekday()
        if days_ahead < 0:
            days_ahead += 7
        
        target_date = first_day + pd.Timedelta(days=days_ahead + (n-1)*7)
        
        if target_date.month == month:
            return target_date
        else:
            return None
    except Exception:
        return None


def _get_last_weekday(year: int, month: int, weekday: int) -> Optional[pd.Timestamp]:
    """Get the last occurrence of a weekday in a month."""
    try:
        # Start from last day of month and work backwards
        if month == 12:
            last_day = pd.Timestamp(f'{year+1}-01-01') - timedelta(days=1)
        else:
            last_day = pd.Timestamp(f'{year}-{month+1:02d}-01') - timedelta(days=1)
        
        days_back = (last_day.weekday() - weekday) % 7
        target_date = last_day - timedelta(days=days_back)
        
        if target_date.month == month:
            return target_date
        else:
            return None
    except Exception:
        return None


def _get_easter(year: int) -> Optional[pd.Timestamp]:
    """Calculate Easter date using the algorithm."""
    try:
        # Anonymous Gregorian algorithm
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        
        return pd.Timestamp(f'{year}-{month:02d}-{day:02d}')
    except Exception:
        return None


def create_time_features_robust(dates: Union[pd.Series, pd.DatetimeIndex], cyclic_encoding: bool = True) -> pd.DataFrame:
    """
    Create comprehensive time-based features with proper cyclic encoding.
    
    Args:
        dates: Date series or index
        cyclic_encoding: Use sin/cos encoding for cyclic features
        
    Returns:
        pd.DataFrame: Time features
    """
    dates_dt = pd.to_datetime(dates)
    time_features = {}
    
    if cyclic_encoding:
        # Cyclic encoding for day of year (1-365/366)
        day_of_year = dates_dt.dt.dayofyear
        # Handle leap years by normalizing to 365.25
        time_features['sin_day_of_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
        time_features['cos_day_of_year'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Cyclic encoding for week of year (1-52/53)
        week_of_year = dates_dt.dt.isocalendar().week.astype(int)  # Fix type issue
        time_features['sin_week_of_year'] = np.sin(2 * np.pi * week_of_year / 52.0)
        time_features['cos_week_of_year'] = np.cos(2 * np.pi * week_of_year / 52.0)
        
        # Day of month (cyclic within month context)
        day_of_month = dates_dt.dt.day
        days_in_month = dates_dt.dt.days_in_month
        normalized_day = day_of_month / days_in_month  # Normalize by month length
        time_features['sin_day_of_month'] = np.sin(2 * np.pi * normalized_day)
        time_features['cos_day_of_month'] = np.cos(2 * np.pi * normalized_day)
    else:
        # Linear features (with potential issues noted)
        time_features['year'] = dates_dt.dt.year
        time_features['month'] = dates_dt.dt.month
        time_features['quarter'] = dates_dt.dt.quarter
        time_features['day_of_year'] = dates_dt.dt.dayofyear
        time_features['week_of_year'] = dates_dt.dt.isocalendar().week.astype(int)
        time_features['day_of_month'] = dates_dt.dt.day
        time_features['days_in_month'] = dates_dt.days_in_month
    
    return pd.DataFrame(time_features)


    return pd.DataFrame(time_features)


if __name__ == "__main__":
    # Test enhanced seasonality feature generation
    
    # Create sample date range
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    
    # Test robust Fourier features
    fourier_features = generate_fourier_features_robust(dates, num_terms=3, weekly=True, annual=True)
    print(f"Fourier features shape: {fourier_features.shape}")
    print(f"Fourier columns: {fourier_features.columns.tolist()}")
    
    # Test robust trend feature
    trend_feature = generate_trend_feature_robust(dates, normalization='none')
    print(f"Trend feature shape: {trend_feature.shape}")
    
    # Test robust holiday features
    holiday_features = generate_holiday_features_robust(
        dates, calendar='US', include_observed=True, effects=['before', 'after']
    )
    print(f"Holiday features shape: {holiday_features.shape}")
    print(f"Holiday columns: {holiday_features.columns.tolist()}")
    
    # Test robust day-of-week features
    dow_features = generate_dow_features_robust(dates)
    print(f"DOW features shape: {dow_features.shape}")
    
    # Test robust time features
    time_features = create_time_features_robust(dates, cyclic_encoding=True)
    print(f"Time features shape: {time_features.shape}")
    print(f"Time columns: {time_features.columns.tolist()}")
    
    print("Enhanced seasonality module test completed successfully")
