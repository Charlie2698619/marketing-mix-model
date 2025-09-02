"""
Baseline controls module - REFINED VERSION (Non-overlapping with seasonality).

Implements baseline and control variables for non-media factors
affecting business performance in MMM.

Addresses overlap issues:
- Removes calendar/holiday features (delegated to seasonality)  
- Removes trend generation (delegated to seasonality)
- Focuses on macro variables, external events, interactions, quality control
- Provides trend normalization policy for seasonality module
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import os

# Optional psutil import for memory diagnostics
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def generate_baseline_features_robust(
    data_df: pd.DataFrame,
    config,
    macro_data: Optional[pd.DataFrame] = None,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Generate refined baseline and control features for MMM modeling.
    
    IMPORTANT: This function no longer generates calendar/holiday/trend features
    to avoid overlap with seasonality module. Use apply_seasonality_from_config() 
    for temporal features.
    
    Args:
        data_df: Main dataset with dates
        config: Configuration object with enhanced baseline settings
        macro_data: Optional external macro-economic data
        training_end_date: End of training period (for leakage-free scaling)
        
    Returns:
        pd.DataFrame: Refined baseline features (non-temporal)
    """
    logger = logging.getLogger(__name__)
    
    # Safe config access with defaults
    baseline_config = getattr(config.features, 'baseline', {})
    if not baseline_config:
        baseline_config = {}
        logger.warning("No baseline configuration found - using defaults")
    
    date_col = getattr(config.data, 'date_col', 'date')
    
    # Handle column name mapping (uppercase config vs lowercase data)
    if date_col not in data_df.columns:
        # Try to find the date column by common alternatives
        date_alternatives = ['date_day', 'date', 'DATE_DAY', 'DATE']
        for alt in date_alternatives:
            if alt in data_df.columns:
                date_col = alt
                logger.info(f"Using date column: {date_col}")
                break
        else:
            raise KeyError(f"Date column not found. Config expects '{getattr(config.data, 'date_col', 'date')}', available columns: {list(data_df.columns[:10])}...")
    
    # Initialize baseline features DataFrame with proper date index
    dates = pd.to_datetime(data_df[date_col])
    baseline_features = pd.DataFrame(index=dates)
    
    logger.info("Starting baseline feature generation (non-temporal features only)")
    logger.info("NOTE: Use seasonality module for calendar/holiday/trend features")
    
    # 1. Add macro-economic variables (core baseline responsibility)
    macro_vars = baseline_config.get('macro_variables', [])
    if macro_vars and macro_data is not None:
        macro_features = process_macro_variables_robust(
            dates=dates,
            macro_data=macro_data,
            variables=macro_vars,
            macro_config=baseline_config.get('macro_config', {}),
            training_end_date=training_end_date
        )
        baseline_features = pd.concat([baseline_features, macro_features], axis=1)
        logger.info(f"Added {len(macro_features.columns)} macro variables (leakage-safe)")
    
    # 2. Generate external event features (non-seasonal events only)
    external_events = baseline_config.get('external_events', [])
    if external_events:
        event_features = generate_external_event_features_robust(
            dates=dates,
            events_config=external_events
        )
        baseline_features = pd.concat([baseline_features, event_features], axis=1)
        logger.info(f"Generated {len(event_features.columns)} external event features")
    
    # 3. Apply quality control (non-temporal features only)
    if baseline_config.get('quality_control', {}).get('drop_zero_variance', True):
        baseline_features = apply_quality_control_robust(
            baseline_features,
            quality_config=baseline_config.get('quality_control', {})
        )
    
    # 4. Logging and diagnostics
    log_baseline_diagnostics(baseline_features, baseline_config)
    
    # Reset index to return DataFrame with date column
    baseline_features = baseline_features.reset_index()
    baseline_features.rename(columns={'index': date_col}, inplace=True)
    
    logger.info(f"Baseline feature generation complete: {baseline_features.shape}")
    logger.info("Remember to add seasonality features separately!")
    
    return baseline_features


def get_trend_normalization_policy(
    config,
    training_end_date: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Get trend normalization policy for seasonality module.
    
    This prevents trend leakage by providing training-window constraints
    to the seasonality module's trend generation.
    
    Args:
        config: Configuration object
        training_end_date: End of training period
        
    Returns:
        Dict: Trend policy for seasonality module
    """
    baseline_config = getattr(config.features, 'baseline', {}) if hasattr(config, 'features') else {}
    
    policy = {
        'normalize': baseline_config.get('trend_normalization', False),
        'training_end_date': training_end_date,
        'trend_types': baseline_config.get('trend_types', ['linear']),
        'center_before_interactions': baseline_config.get('center_before_interactions', True)
    }
    
    return policy


def process_macro_variables_robust(
    dates: pd.Series,
    macro_data: pd.DataFrame,
    variables: List[str],
    macro_config: Dict[str, Any],
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Process macro variables with leakage prevention and publication lags.
    
    Issues: StandardScaler on full series = leakage. No publication lag = look-ahead.
    Fix: Train-only scaling, conservative publication lags.
    
    Args:
        dates: Model date series
        macro_data: Macro-economic data
        variables: Variables to process
        macro_config: Macro processing configuration
        training_end_date: End of training period
        
    Returns:
        pd.DataFrame: Processed macro variables (leakage-safe)
    """
    logger = logging.getLogger(__name__)
    
    if 'date' not in macro_data.columns:
        raise ValueError("Macro data must have 'date' column")
    
    dates_dt = pd.to_datetime(dates)
    macro_data = macro_data.copy()
    macro_data['date'] = pd.to_datetime(macro_data['date'])
    
    # Apply publication lag to avoid look-ahead bias
    lag_days = macro_config.get('publication_lag_days', 30)
    macro_data['date'] = macro_data['date'] + pd.Timedelta(days=lag_days)
    
    processed_vars = pd.DataFrame(index=dates_dt)
    scaling_strategy = macro_config.get('scaling_strategy', 'train_only')
    interpolation = macro_config.get('interpolation_method', 'forward_fill')
    
    for var in variables:
        if var not in macro_data.columns:
            logger.warning(f"Macro variable '{var}' not found in data")
            continue
        
        var_data = macro_data[['date', var]].dropna()
        
        # Align with model dates using proper interpolation
        if interpolation == 'forward_fill':
            # Forward-fill (step function) - more realistic for monthly data
            var_series = var_data.set_index('date')[var]
            
            # Guard against infinite forward-fill beyond last known data
            last_known = var_series.dropna().index.max()
            aligned_values = var_series.reindex(dates_dt, method='ffill')
            
            # Optional: warn or clip future values beyond last known data
            if dates_dt.max() > last_known:
                future_mask = dates_dt > last_known
                future_count = future_mask.sum()
                if macro_config.get('warn_future_extrapolation', True):
                    logger.warning(f"Macro variable '{var}' extrapolated {future_count} days beyond last known date ({last_known.date()})")
                
                # Option to set future values as NaN instead of forward-filling
                if macro_config.get('clip_future_extrapolation', False):
                    aligned_values.loc[aligned_values.index > last_known] = np.nan
                    logger.info(f"Set {future_count} future values to NaN for '{var}' (no extrapolation)")
                    
        elif interpolation == 'linear':
            # Linear interpolation
            full_dates = pd.date_range(
                start=min(dates_dt.min(), var_data['date'].min()),
                end=max(dates_dt.max(), var_data['date'].max()),
                freq='D'
            )
            var_series = var_data.set_index('date')[var].reindex(full_dates)
            var_series = var_series.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            aligned_values = var_series.reindex(dates_dt)
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")
        
        # Apply scaling strategy to prevent leakage
        if scaling_strategy == 'train_only' and training_end_date is not None:
            # Fit scaler only on training data
            train_mask = aligned_values.index <= training_end_date
            train_values = aligned_values[train_mask].dropna()
            
            if len(train_values) > 1:
                scaler = StandardScaler()
                scaler.fit(train_values.values.reshape(-1, 1))
                standardized_values = scaler.transform(aligned_values.values.reshape(-1, 1)).flatten()
            else:
                logger.warning(f"Insufficient training data for {var}, using raw values")
                standardized_values = aligned_values.values
        elif scaling_strategy == 'full_series':
            # Traditional approach (has leakage)
            warnings.warn(f"Using full-series scaling for {var} - may cause data leakage")
            scaler = StandardScaler()
            standardized_values = scaler.fit_transform(aligned_values.values.reshape(-1, 1)).flatten()
        else:
            # No scaling
            standardized_values = aligned_values.values
        
        processed_vars[f'macro_{var}'] = standardized_values
        logger.debug(f"Processed macro variable: {var} (lag={lag_days}d, scaling={scaling_strategy})")
    
    return processed_vars


def generate_external_event_features_robust(
    dates: pd.Series,
    events_config: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Generate EXTERNAL event features (non-seasonal/non-holiday events only).
    
    NOTE: This function handles external business events like COVID, strikes, 
    supply disruptions. Regular holidays and shopping events should be handled 
    by the seasonality module to avoid feature duplication.
    
    Args:
        dates: Date series
        events_config: List of external event configurations
        
    Returns:
        pd.DataFrame: External event features only
    """
    dates_dt = pd.to_datetime(dates)
    features = {}
    
    # Process configured external events (vectorized)
    for event in events_config:
        event_name = event['name']
        start_date = pd.to_datetime(event['start'])
        end_date = pd.to_datetime(event['end'])
        impact_type = event.get('impact_type', 'step')
        
        # Binary indicator
        mask = (dates_dt >= start_date) & (dates_dt <= end_date)
        features[f'external_{event_name}'] = mask.astype(int)
        
        # Vectorized ramp calculation for external events
        if impact_type == 'ramp_up':
            ramp = np.clip(
                ((dates_dt - start_date).dt.days) / ((end_date - start_date).days + 1e-9), 
                0, 1
            )
            features[f'external_{event_name}_ramp'] = np.where(mask, ramp, 0.0)
        elif impact_type == 'ramp_down':
            ramp = np.clip(
                1 - ((dates_dt - start_date).dt.days) / ((end_date - start_date).days + 1e-9), 
                0, 1
            )
            features[f'external_{event_name}_ramp'] = np.where(mask, ramp, 0.0)
    
    return pd.DataFrame(features, index=dates_dt)


def apply_quality_control_robust(
    baseline_features: pd.DataFrame,
    quality_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply quality control to baseline features.
    
    Issue: No variance checking or correlation analysis.
    Fix: Drop zero-variance and highly correlated features.
    
    Args:
        baseline_features: Input features
        quality_config: Quality control configuration
        
    Returns:
        pd.DataFrame: Quality-controlled features
    """
    logger = logging.getLogger(__name__)
    
    original_features = baseline_features.copy()
    
    # Drop zero variance features
    if quality_config.get('drop_zero_variance', True):
        min_var = quality_config.get('min_variance_threshold', 1e-6)
        
        # Calculate variance for numeric columns only (exclude datetime and date columns)
        numeric_cols = baseline_features.select_dtypes(include=[np.number]).columns
        # Exclude any date/datetime columns
        date_like_cols = baseline_features.select_dtypes(include=['datetime64', 'object']).columns
        date_like_cols = date_like_cols.union([
            col for col in baseline_features.columns 
            if any(pattern in col.lower() for pattern in ['date', 'day', 'time'])
        ])
        
        analysis_cols = numeric_cols.difference(date_like_cols)
        
        if len(analysis_cols) > 0:
            variances = baseline_features[analysis_cols].var()
            # Ensure min_var is numeric and variances are numeric
            min_var = float(min_var) if not isinstance(min_var, (int, float)) else min_var
            # Filter out any non-numeric variances
            numeric_variances = variances.dropna()
            zero_var_cols = numeric_variances[numeric_variances < min_var].index.tolist()
            if zero_var_cols:
                baseline_features = baseline_features.drop(columns=zero_var_cols)
                logger.info(f"Dropped {len(zero_var_cols)} zero-variance features: {zero_var_cols}")
    
    # Drop highly correlated features
    max_corr = quality_config.get('max_correlation_threshold', 0.95)
    if max_corr < 1.0:
        numeric_cols = baseline_features.select_dtypes(include=[np.number]).columns
        # Exclude date/datetime columns
        date_like_cols = baseline_features.select_dtypes(include=['datetime64', 'object']).columns
        date_like_cols = date_like_cols.union([
            col for col in baseline_features.columns 
            if any(pattern in col.lower() for pattern in ['date', 'day', 'time'])
        ])
        
        analysis_cols = numeric_cols.difference(date_like_cols)
        
        if len(analysis_cols) > 1:
            corr_matrix = baseline_features[analysis_cols].corr().abs()
            
            # Find pairs with high correlation
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_pairs = [
                (col, row) for col in upper_tri.columns 
                for row in upper_tri.index 
                if upper_tri.loc[row, col] > max_corr
            ]
            
            # Drop second feature in each pair
            to_drop = list(set([pair[1] for pair in high_corr_pairs]))
            if to_drop:
                baseline_features = baseline_features.drop(columns=to_drop)
                logger.info(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
    
    return baseline_features


def create_baseline_interaction_terms_robust(
    baseline_features: pd.DataFrame,
    media_features: pd.DataFrame,
    interaction_config: Dict[str, Any],
    seasonality_features: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Create interaction terms with variance-based media selection.
    
    Issue: Baseline no longer contains seasonal/trend features - they're in seasonality module.
    Fix: Accept combined features or separate seasonality_features argument.
    
    Args:
        baseline_features: Baseline feature DataFrame (macro, external events)
        media_features: Media feature DataFrame  
        interaction_config: Interaction configuration
        seasonality_features: Optional seasonality features (trend, calendar, holidays)
        
    Returns:
        pd.DataFrame: Interaction features
    """
    logger = logging.getLogger(__name__)
    
    interaction_features = pd.DataFrame(index=baseline_features.index)
    
    # Select media channels intelligently
    media_cols = [col for col in media_features.columns if col.endswith('_spend')]
    top_k = interaction_config.get('media_top_k', 3)
    selection_method = interaction_config.get('selection_method', 'variance')
    
    if selection_method == 'variance':
        # Select by variance (most variable channels)
        media_variances = media_features[media_cols].var().sort_values(ascending=False)
        selected_media = media_variances.head(top_k).index.tolist()
    elif selection_method == 'spend_share':
        # Select by total spend
        media_totals = media_features[media_cols].sum().sort_values(ascending=False)
        selected_media = media_totals.head(top_k).index.tolist()
    else:
        # Default: first k by order
        selected_media = media_cols[:top_k]
    
    logger.info(f"Selected media for interactions ({selection_method}): {selected_media}")
    
    # Combine baseline and seasonality features for interactions
    if seasonality_features is not None:
        # Use seasonality features for trend/calendar interactions
        combined_features = pd.concat([baseline_features, seasonality_features], axis=1)
        logger.info("Using combined baseline + seasonality features for interactions")
    else:
        # Fallback to baseline only (will have limited interactions)
        combined_features = baseline_features.copy()
        logger.warning("No seasonality features provided - interactions will be limited to baseline features only")
    
    # Center trend before interactions for stability (from seasonality features)
    trend_cols = [col for col in combined_features.columns if 'trend' in col.lower()]
    centered_features = combined_features.copy()
    
    if interaction_config.get('center_before_interactions', True) and trend_cols:
        for trend_col in trend_cols:
            if trend_col in centered_features.columns:
                mean_val = centered_features[trend_col].mean()
                centered_features[f'{trend_col}_centered'] = centered_features[trend_col] - mean_val
        logger.info(f"Centered {len(trend_cols)} trend columns for stable interactions")
    
    # Seasonal × Media interactions (from seasonality features)
    seasonal_cols = [
        col for col in combined_features.columns 
        if any(prefix in col for prefix in ['month_', 'quarter_', 'dow_'])
    ]
    
    max_seasonal = interaction_config.get('max_seasonal_interactions', 4)
    selected_seasonal = seasonal_cols[:max_seasonal]
    
    if selected_seasonal:
        for seasonal_col in selected_seasonal:
            for media_col in selected_media:
                if seasonal_col in combined_features.columns and media_col in media_features.columns:
                    interaction_name = f"{seasonal_col}_x_{media_col.replace('_spend', '')}"
                    interaction_features[interaction_name] = (
                        combined_features[seasonal_col] * media_features[media_col]
                    )
        logger.info(f"Created {len(selected_seasonal)} × {len(selected_media)} seasonal interactions")
    else:
        logger.info("No seasonal features found - skipping seasonal interactions")
    
    # Trend × Media interactions (from seasonality features)
    if interaction_config.get('include_trend_interactions', True):
        trend_interaction_cols = [
            col for col in centered_features.columns 
            if col.endswith('_centered') or (col in trend_cols and not col.endswith('_centered'))
        ]
        
        if trend_interaction_cols:
            for trend_col in trend_interaction_cols:
                for media_col in selected_media:
                    if trend_col in centered_features.columns and media_col in media_features.columns:
                        interaction_name = f"{trend_col.replace('_centered', '')}_x_{media_col.replace('_spend', '')}"
                        interaction_features[interaction_name] = (
                            centered_features[trend_col] * media_features[media_col]
                        )
            logger.info(f"Created {len(trend_interaction_cols)} × {len(selected_media)} trend interactions")
        else:
            logger.info("No trend features found - skipping trend interactions")
    
    logger.info(f"Total interaction terms created: {len(interaction_features.columns)}")
    return interaction_features


def log_baseline_diagnostics(
    baseline_features: pd.DataFrame,
    baseline_config: Dict[str, Any]
) -> None:
    """
    Log comprehensive baseline feature diagnostics.
    
    Issue: No feature counting, memory tracking, or variance analysis.
    Fix: Comprehensive logging and diagnostics.
    
    Args:
        baseline_features: Generated baseline features
        baseline_config: Baseline configuration
    """
    logger = logging.getLogger(__name__)
    
    if not baseline_config.get('quality_control', {}).get('log_feature_diagnostics', True):
        return
    
    # Feature counts by type (reflecting non-overlapping design)
    feature_counts = {
        'total': len(baseline_features.columns),
        'macro': len([col for col in baseline_features.columns if col.startswith('macro_')]),
        'external_events': len([col for col in baseline_features.columns if col.startswith('external_')]),
        'baseline_events': len([col for col in baseline_features.columns if col.startswith('event_')]),
        'trend (expect 0)': len([col for col in baseline_features.columns if 'trend' in col]),
        'calendar (expect 0)': len([col for col in baseline_features.columns if any(
            prefix in col for prefix in ['month_', 'quarter_', 'dow_', 'week_', 'is_']
        )]),
        'shopping (expect 0)': len([col for col in baseline_features.columns if any(
            event in col for event in ['black_friday', 'cyber_monday', 'small_business']
        )]),
    }
    
    # Memory footprint
    memory_mb = baseline_features.memory_usage(deep=True).sum() / 1024**2
    
    # Variance analysis (exclude date columns)
    numeric_cols = baseline_features.select_dtypes(include=[np.number]).columns
    date_like_cols = baseline_features.select_dtypes(include=['datetime64', 'object']).columns
    date_like_cols = date_like_cols.union([
        col for col in baseline_features.columns 
        if any(pattern in col.lower() for pattern in ['date', 'day', 'time'])
    ])
    
    analysis_cols = numeric_cols.difference(date_like_cols)
    
    if len(analysis_cols) > 0:
        variances = baseline_features[analysis_cols].var()
        zero_var_count = (variances == 0).sum()
        low_var_count = (variances < 1e-6).sum()
    else:
        zero_var_count = low_var_count = 0
    
    # Log diagnostics
    logger.info("=== BASELINE FEATURE DIAGNOSTICS ===")
    for feature_type, count in feature_counts.items():
        logger.info(f"{feature_type.capitalize()} features: {count}")
    
    logger.info(f"Memory footprint: {memory_mb:.2f} MB")
    logger.info(f"Zero variance features: {zero_var_count}")
    logger.info(f"Low variance features (< 1e-6): {low_var_count}")
    
    if HAS_PSUTIL:
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"System memory usage: {memory_percent:.1f}%")
    else:
        logger.debug("psutil not available - skipping system memory diagnostics")


def apply_baseline_transformation_robust(
    data_df: pd.DataFrame,
    config,
    macro_data: Optional[pd.DataFrame] = None,
    media_features: Optional[pd.DataFrame] = None,
    seasonality_features: Optional[pd.DataFrame] = None,
    training_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Apply complete refined baseline transformation pipeline.
    
    Issue: Unsafe merging and no duplicate handling.
    Fix: Safe DataFrame alignment with index-based joins.
    
    Args:
        data_df: Input data
        config: Configuration object
        macro_data: External macro-economic data
        media_features: Media features for interactions
        seasonality_features: Seasonality features for interactions (trend, calendar, holidays)
        training_end_date: End of training period (for leakage prevention)
        
    Returns:
        pd.DataFrame: Data with refined baseline features
    """
    logger = logging.getLogger(__name__)
    
    date_col = config.data.date_col
    
    # Ensure no duplicate dates in input data
    if data_df[date_col].duplicated().any():
        logger.warning("Duplicate dates found in input data - taking first occurrence")
        data_df = data_df.drop_duplicates(subset=[date_col], keep='first')
    
    # Generate baseline features
    baseline_features = generate_baseline_features_robust(
        data_df=data_df,
        config=config,
        macro_data=macro_data,
        training_end_date=training_end_date
    )
    
    # Safe merging using index-based join with collision guard
    data_indexed = data_df.set_index(date_col)
    baseline_indexed = baseline_features.set_index(date_col)
    
    # Check for overlapping columns and handle collisions
    overlap = set(data_indexed.columns) & set(baseline_indexed.columns)
    if overlap:
        logger.warning(f"Overlapping columns detected - dropping from baseline: {sorted(overlap)}")
        baseline_indexed = baseline_indexed.drop(columns=list(overlap))
    
    # Left join to preserve all data rows
    result_df = data_indexed.join(baseline_indexed, how='left').reset_index()
    
    # Create interaction terms if media features provided
    if media_features is not None:
        interaction_config = getattr(config.features, 'baseline', {}).get('interactions', {})
        interaction_features = create_baseline_interaction_terms_robust(
            baseline_features=baseline_features.set_index(date_col),
            media_features=media_features,
            interaction_config=interaction_config,
            seasonality_features=seasonality_features.set_index(date_col) if seasonality_features is not None else None
        )
        
        # Join interaction features
        if len(interaction_features.columns) > 0:
            result_indexed = result_df.set_index(date_col)
            result_df = result_indexed.join(interaction_features, how='left').reset_index()
    
    logger.info(f"Applied refined baseline transformation: {data_df.shape} → {result_df.shape}")
    return result_df


if __name__ == "__main__":
    """
    Test refined baseline features with updated non-overlapping design.
    Run with: python -m src.mmm.features.baseline
    """
    import numpy as np
    
    print("=== TESTING REFINED BASELINE FEATURES ===")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=730, freq='D')  # 2 years
    sample_data = pd.DataFrame({
        'date': dates,
        'revenue': np.random.exponential(1000, 730)
    })
    
    # Create mock config matching current YAML structure (non-overlapping)
    class MockConfig:
        class Features:
            baseline = {
                # Non-temporal baseline configuration only
                'macro_variables': ['GDP', 'UNEMPLOYMENT'],
                'macro_config': {
                    'publication_lag_days': 30,
                    'interpolation_method': 'forward_fill',
                    'scaling_strategy': 'train_only',
                    'warn_future_extrapolation': True
                },
                'external_events': [
                    {'name': 'covid_lockdown', 'start': '2020-03-15', 'end': '2020-06-01', 'impact_type': 'step'},
                    {'name': 'supply_crisis', 'start': '2021-09-01', 'end': '2022-03-31', 'impact_type': 'ramp_up'}
                ],
                'interactions': {
                    'media_top_k': 3,
                    'selection_method': 'variance',
                    'max_seasonal_interactions': 4,
                    'include_trend_interactions': True
                },
                'quality_control': {
                    'log_feature_diagnostics': True,
                    'drop_zero_variance': True,
                    'max_correlation_threshold': 0.99
                }
            }
        
        class Data:
            date_col = 'date'
        
        features = Features()
        data = Data()
    
    config = MockConfig()
    
    # Create mock macro data
    macro_dates = pd.date_range('2019-12-01', '2022-06-30', freq='D')
    macro_data = pd.DataFrame({
        'date': macro_dates,
        'GDP': np.random.normal(100, 5, len(macro_dates)),
        'UNEMPLOYMENT': np.random.normal(5, 1, len(macro_dates))
    })
    
    # Test baseline feature generation (non-temporal only)
    print("\n1. Testing baseline feature generation...")
    baseline_features = generate_baseline_features_robust(
        sample_data, 
        config,
        macro_data=macro_data,
        training_end_date=pd.Timestamp('2021-06-30')
    )
    
    print(f"✅ Generated {len(baseline_features.columns)} baseline features")
    print(f"✅ Shape: {baseline_features.shape}")
    print(f"✅ Features: {list(baseline_features.columns)}")
    
    # Test interaction creation with mock seasonality
    print("\n2. Testing interactions with seasonality separation...")
    
    # Mock seasonality features (would come from seasonality module)
    seasonality_features = pd.DataFrame({
        'date': dates,
        'trend_linear': np.arange(len(dates)) / len(dates),
        'month_1': np.random.randint(0, 2, len(dates)),
        'dow_monday': np.random.randint(0, 2, len(dates))
    })
    
    # Mock media features
    media_features = pd.DataFrame({
        'date': dates,
        'google_spend': np.random.exponential(100, len(dates)),
        'facebook_spend': np.random.exponential(80, len(dates))
    })
    
    interaction_features = create_baseline_interaction_terms_robust(
        baseline_features=baseline_features.set_index('date'),
        media_features=media_features.set_index('date'),
        interaction_config=config.features.baseline['interactions'],
        seasonality_features=seasonality_features.set_index('date')
    )
    
    print(f"✅ Generated {len(interaction_features.columns)} interaction terms")
    print(f"✅ Interactions: {list(interaction_features.columns)}")
    
    # Test complete pipeline
    print("\n3. Testing complete transformation pipeline...")
    result_df = apply_baseline_transformation_robust(
        data_df=sample_data,
        config=config,
        macro_data=macro_data,
        media_features=media_features,
        seasonality_features=seasonality_features,
        training_end_date=pd.Timestamp('2021-06-30')
    )
    
    print(f"✅ Complete pipeline result: {result_df.shape}")
    print(f"✅ Final columns: {len(result_df.columns)} total")
    
    # Verify no temporal overlap
    temporal_keywords = ['month_', 'dow_', 'quarter_', 'holiday', 'weekend', 'fourier', 'seasonal']
    baseline_temporal = [col for col in baseline_features.columns 
                        if any(keyword in col.lower() for keyword in temporal_keywords)]
    
    if baseline_temporal:
        print(f"❌ WARNING: Found temporal features in baseline: {baseline_temporal}")
    else:
        print("✅ VERIFIED: No temporal overlap - baseline is clean!")
    
    print("\n🎯 BASELINE MODULE TEST COMPLETED SUCCESSFULLY!")
    print("💡 Remember: Use seasonality module for calendar/holiday/trend features")
