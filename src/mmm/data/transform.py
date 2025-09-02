"""
Data transformation module for MMM project.

Performs data cleaning, harmonization, and preparation for feature engineering.
This module focuses on transforming validated data into analysis-ready format.
"""

import pandas as pd
import numpy as np
import json
import hashlib
import logging
from pandas.util import hash_pandas_object
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import warnings
import re

from ..utils.logging import StepLogger, log_data_quality


def normalize_and_resolve_columns(df: pd.DataFrame, config):
    """Normalize column names and resolve key/channel columns from YAML."""
    df = df.copy()
    # normalize names once
    df.columns = [c.strip().lower() for c in df.columns]

    # resolve keys from config (fail-hard if missing)
    date_col = getattr(config.data, "date_col", "date").lower()
    brand_key = getattr(config.data, "brand_key", "brand")
    if brand_key:
        brand_key = brand_key.lower()
    region_key = getattr(config.data, "region_key", "region")
    if region_key:
        region_key = region_key.lower()
    revenue_col = getattr(config.data, "revenue_col", "revenue").lower()

    if date_col not in df.columns:
        raise KeyError(f"Configured date_col='{date_col}' not in dataframe.")
    for k, nm in [("brand", brand_key), ("region", region_key)]:
        if nm and nm not in df.columns:
            raise KeyError(f"Configured {k}_key='{nm}' not in dataframe.")

    # UNIVERSAL FIX: Update the config object with normalized column names
    # This ensures all functions that use config.data.* get the correct lowercase names
    if hasattr(config.data, '__dict__'):
        config.data.date_col = date_col
        config.data.revenue_col = revenue_col
        if hasattr(config.data, 'brand_key'):
            config.data.brand_key = brand_key
        if hasattr(config.data, 'region_key'):
            config.data.region_key = region_key

    # resolve spend cols via channel_map
    ch_map = getattr(config.data, "channel_map", {}) or {}
    # channel_map may be dict of alias->col; allow both alias or canonical
    spend_cols = []
    for alias, raw_col in ch_map.items():
        col = raw_col.lower()
        if col in df.columns:
            spend_cols.append(col)

    # telemetry convenience lists
    impression_cols = [c for c in df.columns if "impression" in c]
    click_cols = [c for c in df.columns if "click" in c]

    resolved = {
        "date_col": date_col,
        "brand_col": brand_key if brand_key else None,
        "region_col": region_key if region_key else None,
        "revenue_col": revenue_col,
        "spend_cols": spend_cols,
        "impression_cols": impression_cols,
        "click_cols": click_cols,
    }
    return df, resolved


def transform_data(config) -> Dict[str, Any]:
    """
    Perform comprehensive data transformation and cleaning.
    
    This function takes validated data and applies cleaning transformations
    to prepare it for feature engineering.
    
    Args:
        config: Configuration object
        
    Returns:
        Dict: Transformation results and metadata
    """
    logger = logging.getLogger(__name__)
    
    with StepLogger("transform_data"):
        # Load validated data
        interim_path = Path(config.paths.interim)
        input_file = interim_path / "validated_data.parquet"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}. Run validation first.")
        
        df = pd.read_parquet(input_file)
        
        # Load validation results to inform cleaning decisions
        validation_report_path = interim_path / "validation_report.json"
        validation_results = {}
        if validation_report_path.exists():
            with open(validation_report_path, 'r') as f:
                validation_results = json.load(f)
        
        df, col = normalize_and_resolve_columns(df, config)
        date_col   = col["date_col"]
        brand_col  = col["brand_col"]
        region_col = col["region_col"]
        revenue_col= col["revenue_col"]
        spend_cols = col["spend_cols"]
        
        logger.info(f"Loaded {len(df)} rows for transformation")
        
        # Apply data cleaning and transformation
        transformed_df = apply_data_cleaning(df, config, validation_results)
        
        # Calculate transformation metadata
        transform_results = {
            'original_shape': df.shape,
            'transformed_shape': transformed_df.shape,
            'rows_changed': df.shape[0] - transformed_df.shape[0],
            'columns_added': transformed_df.shape[1] - df.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        # Log transformation results
        logger.info(f"Transformation completed. Shape: {df.shape} → {transformed_df.shape}")
        
        # Save transformed data
        output_file = interim_path / "transformed_data.parquet"
        transformed_df.to_parquet(output_file, index=False)
        logger.info(f"Transformed data saved to {output_file}")
        
        # Save transformation metadata
        metadata_file = interim_path / "transformation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(transform_results, f, indent=2, default=str)
        
        return transform_results


def apply_data_cleaning(df: pd.DataFrame, config, validation_results: Dict[str, Any]) -> pd.DataFrame:
    """Apply enhanced MMM-aware data cleaning that preserves business signal."""
    
    import logging
    import re
    
    cleaned_df = df.copy()
    logger = logging.getLogger(__name__)
    
    # Get enhanced cleaning configuration
    enhanced_cleaning = getattr(config.validation, 'enhanced_cleaning', {})
    negative_values_config = enhanced_cleaning.get('negative_values', {})
    outliers_config = enhanced_cleaning.get('outliers', {})
    duplicates_config = enhanced_cleaning.get('duplicates', {})
    missing_data_config = enhanced_cleaning.get('missing_data', {})
    documentation_config = enhanced_cleaning.get('documentation', {})
    
    # Get document_all_changes configuration
    document_all_changes = documentation_config.get('document_all_changes', True)
    
    # Initialize cleaning actions log
    cleaning_actions = []
    detailed_actions = []  # For granular tracking when document_all_changes=True
    cleaning_metadata = {
        "timestamp": datetime.now().isoformat(),
        "original_rows": len(df),
        "original_columns": list(df.columns),
        "cleaning_config": enhanced_cleaning,
        "document_all_changes": document_all_changes
    }
    
    # Helper function to conditionally log cleaning actions
    def log_cleaning_action(action: str, level: str = "standard", details: dict = None):
        """
        Log cleaning action based on document_all_changes setting.
        
        Args:
            action: Description of the cleaning action
            level: "standard" for major actions, "detailed" for granular actions
            details: Additional metadata about the action
        """
        if level == "standard":
            # Always log major actions
            cleaning_actions.append(action)
        elif level == "detailed" and document_all_changes:
            # Only log detailed actions if document_all_changes is enabled
            cleaning_actions.append(action)
            if details:
                detailed_actions.append({
                    "action": action,
                    "timestamp": datetime.now().isoformat(),
                    "details": details
                })
        
        # Always log detailed actions to detailed_actions when document_all_changes is True
        if document_all_changes and details:
            detailed_actions.append({
                "action": action,
                "level": level,
                "timestamp": datetime.now().isoformat(),
                "details": details
            })
    
    logger.info(f"Enhanced cleaning started. Document all changes: {document_all_changes}")
    
    # ========================================================================
    # 1. HANDLE NEGATIVE VALUES WITH BUSINESS INTELLIGENCE
    # ========================================================================
    logger.info("Starting intelligent negative value handling...")
    
    # Get resolved revenue column name
    revenue_col = config.data.revenue_col
    spend_cols = [col for col in df.columns if col.endswith('_spend')]
    
    # Identify telemetry columns that must be non-negative based on YAML config
    telemetry_patterns = negative_values_config.get('telemetry_columns', ['CLICKS', 'IMPRESSIONS', 'VIEWS'])
    telemetry_cols = []
    for pattern in telemetry_patterns:
        telemetry_cols.extend([col for col in df.columns if pattern in col.upper()])
    
    # Handle revenue negatives (returns/refunds)
    revenue_policy = negative_values_config.get('revenue_policy', 'route_to_adjustments')
    if revenue_policy == 'route_to_adjustments':
        # Create adjustment columns if enabled
        create_adjustment_columns = negative_values_config.get('create_adjustment_columns', True)
        if create_adjustment_columns:
            returns_col = f"{revenue_col}_RETURNS"
            if returns_col not in cleaned_df.columns:
                cleaned_df[returns_col] = 0.0
            
            # Route negative revenue to returns column
            negative_revenue_mask = cleaned_df[revenue_col] < 0
            negative_count = negative_revenue_mask.sum()
            
            if negative_count > 0:
                cleaned_df.loc[negative_revenue_mask, returns_col] = cleaned_df.loc[negative_revenue_mask, revenue_col]
                cleaned_df.loc[negative_revenue_mask, revenue_col] = 0.0
                
                # Create flag for periods with returns
                if negative_values_config.get('create_flags', True):
                    cleaned_df['has_returns'] = negative_revenue_mask.astype(int)
                
                log_cleaning_action(
                    f"Routed {negative_count} negative revenue values to {returns_col}",
                    level="standard",
                    details={
                        "policy": "route_to_adjustments",
                        "column": revenue_col,
                        "adjustment_column": returns_col,
                        "negative_count": negative_count,
                        "flag_created": negative_values_config.get('create_flags', True)
                    }
                )
                logger.info(f"Routed {negative_count} negative revenue values to returns column")
    
    # Handle spend negatives (ad credits/adjustments)
    spend_policy = negative_values_config.get('spend_policy', 'route_to_adjustments')
    if spend_policy == 'route_to_adjustments':
        create_adjustment_columns = negative_values_config.get('create_adjustment_columns', True)
        if create_adjustment_columns:
            adjustment_suffix = negative_values_config.get('adjustment_column_suffix', '_ADJUSTMENTS')
            
            for spend_col in spend_cols:
                credit_col = f"{spend_col}{adjustment_suffix}"
                if credit_col not in cleaned_df.columns:
                    cleaned_df[credit_col] = 0.0
                
                negative_spend_mask = cleaned_df[spend_col] < 0
                negative_count = negative_spend_mask.sum()
                
                if negative_count > 0:
                    cleaned_df.loc[negative_spend_mask, credit_col] = cleaned_df.loc[negative_spend_mask, spend_col]
                    cleaned_df.loc[negative_spend_mask, spend_col] = 0.0
                    
                    # Create flag for periods with ad credits
                    if negative_values_config.get('create_flags', True):
                        flag_col = f"has_ad_credits_{spend_col.replace('_SPEND', '').lower()}"
                        cleaned_df[flag_col] = negative_spend_mask.astype(int)
                    
                    log_cleaning_action(
                        f"Routed {negative_count} negative spend values in {spend_col} to {credit_col}",
                        level="standard",
                        details={
                            "policy": "route_to_adjustments",
                            "column": spend_col,
                            "adjustment_column": credit_col,
                            "negative_count": negative_count,
                            "flag_created": negative_values_config.get('create_flags', True),
                            "flag_column": flag_col if negative_values_config.get('create_flags', True) else None
                        }
                    )
                    logger.info(f"Routed {negative_count} negative spend values to credits column: {credit_col}")
    
    # Handle telemetry negatives (clip ONLY telemetry_columns from YAML config)
    metrics_policy = negative_values_config.get('metrics_policy', 'clip_to_zero')
    if metrics_policy == 'clip_to_zero':
        for col in telemetry_cols:
            if col in cleaned_df.columns:
                negative_mask = cleaned_df[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    cleaned_df.loc[negative_mask, col] = 0.0
                    
                    # Flag negative telemetry for investigation
                    flag_col = None
                    if negative_values_config.get('flag_negative_telemetry', True):
                        flag_col = f"negative_telemetry_{col.lower()}"
                        cleaned_df[flag_col] = negative_mask.astype(int)
                    
                    log_cleaning_action(
                        f"Clipped {negative_count} negative {col} values to zero",
                        level="detailed" if document_all_changes else "standard",
                        details={
                            "policy": "clip_to_zero",
                            "column": col,
                            "negative_count": negative_count,
                            "telemetry_type": "metrics",
                            "flag_created": negative_values_config.get('flag_negative_telemetry', True),
                            "flag_column": flag_col
                        }
                    )
                    logger.warning(f"Clipped {negative_count} negative telemetry values in {col}")
    
    # ========================================================================
    # 2. CONTEXTUAL OUTLIER HANDLING
    # ========================================================================
    logger.info("Starting contextual outlier detection and handling...")
    
    outlier_policy = outliers_config.get('policy', 'contextual_winsorization')
    if outlier_policy == 'contextual_winsorization':
        # Identify promotional and launch periods to exempt
        promo_flag_col = outliers_config.get('promo_flag_column', 'PROMO_FLAG')
        exempt_promo = outliers_config.get('exempt_promo_periods', True)
        exempt_launch = outliers_config.get('exempt_launch_periods', True)
        launch_detection_method = outliers_config.get('launch_detection_method', 'spend_jumps')
        
        # Create exemption mask
        exemption_mask = pd.Series(False, index=cleaned_df.index)
        
        if exempt_promo and promo_flag_col in cleaned_df.columns:
            exemption_mask |= (cleaned_df[promo_flag_col] == 1)
            logger.info(f"Exempting {exemption_mask.sum()} promotional periods from outlier detection")
        
        # Detect campaign launches if enabled
        if exempt_launch and launch_detection_method == 'spend_jumps':
            launch_threshold = outliers_config.get('launch_threshold_multiplier', 3.0)
            for spend_col in spend_cols:
                if spend_col in cleaned_df.columns:
                    # Calculate rolling mean for launch detection
                    rolling_mean = cleaned_df[spend_col].rolling(window=4, min_periods=1).mean()
                    launch_mask = cleaned_df[spend_col] > (rolling_mean * launch_threshold)
                    exemption_mask |= launch_mask
                    
                    launch_count = launch_mask.sum()
                    if launch_count > 0:
                        logger.info(f"Detected {launch_count} campaign launch periods in {spend_col}")
        
        # Implement advanced outlier detection features
        use_robust_zscore = outliers_config.get('use_robust_zscore', True)
        mad_threshold = outliers_config.get('mad_threshold', 3.5)
        seasonal_detrend = outliers_config.get('seasonal_detrend', True)
        
        # Configure grouping strategy
        grouping_columns = outliers_config.get('grouping_columns', ['channel', 'month'])
        percentile_method = outliers_config.get('percentile_method', 'per_group')
        percentile_threshold = outliers_config.get('winsorize_percentile', 99) / 100.0
        
        # Create grouping columns based on configuration
        date_col = config.data.date_col
        if 'month' in grouping_columns:
            cleaned_df['month'] = pd.to_datetime(cleaned_df[date_col]).dt.month
        if 'quarter' in grouping_columns:
            cleaned_df['quarter'] = pd.to_datetime(cleaned_df[date_col]).dt.quarter
        if 'weekday' in grouping_columns:
            cleaned_df['weekday'] = pd.to_datetime(cleaned_df[date_col]).dt.dayofweek
        
        # Implement logging glitch detection
        detect_logging_glitches = outliers_config.get('detect_logging_glitches', True)
        if detect_logging_glitches:
            glitch_rules = outliers_config.get('glitch_rules', {})
            
            # Detect spend with zero impressions
            if glitch_rules.get('spend_with_zero_impressions', True):
                for spend_col in spend_cols:
                    if spend_col in cleaned_df.columns:
                        channel = spend_col.replace('_SPEND', '')
                        impression_cols = [col for col in cleaned_df.columns if channel in col and 'IMPRESSION' in col]
                        
                        if impression_cols:
                            impression_col = impression_cols[0]
                            glitch_mask = (cleaned_df[spend_col] > 0) & (cleaned_df[impression_col] == 0)
                            glitch_count = glitch_mask.sum()
                            
                            if glitch_count > 0:
                                # Flag as data quality issue
                                cleaned_df[f'glitch_spend_no_impressions_{channel.lower()}'] = glitch_mask.astype(int)
                                cleaning_actions.append(f"Flagged {glitch_count} periods with spend but zero impressions in {channel}")
                                logger.warning(f"Data quality issue: {glitch_count} periods with {spend_col} > 0 but {impression_col} = 0")
            
            # Detect spend jump anomalies
            if glitch_rules.get('spend_jump_multiplier'):
                jump_multiplier = glitch_rules['spend_jump_multiplier']
                for spend_col in spend_cols:
                    if spend_col in cleaned_df.columns:
                        # Calculate period-over-period changes
                        spend_change = cleaned_df[spend_col].pct_change()
                        jump_mask = spend_change > jump_multiplier
                        jump_count = jump_mask.sum()
                        
                        if jump_count > 0:
                            channel = spend_col.replace('_SPEND', '')
                            cleaned_df[f'glitch_spend_jump_{channel.lower()}'] = jump_mask.astype(int)
                            cleaning_actions.append(f"Flagged {jump_count} spend jump anomalies in {channel}")
        
        # Apply contextual winsorization based on method
        if percentile_method == 'per_group':
            # Winsorize each spend column contextually by group
            for spend_col in spend_cols:
                if spend_col in cleaned_df.columns:
                    channel_name = spend_col.replace('_SPEND', '').lower()
                    cleaned_df['channel'] = channel_name
                    
                    # Determine grouping columns to use
                    available_group_cols = ['channel']
                    for group_col in grouping_columns:
                        if group_col in cleaned_df.columns and group_col != 'channel':
                            available_group_cols.append(group_col)
                    
                    # Calculate group-wise percentiles (excluding exempted periods)
                    non_exempt_data = cleaned_df[~exemption_mask]
                    if len(non_exempt_data) > 0 and len(available_group_cols) > 0:
                        try:
                            group_caps = non_exempt_data.groupby(available_group_cols)[spend_col].quantile(percentile_threshold)
                            
                            # Apply caps (but exempt promotional/launch periods)
                            for group_key, cap_value in group_caps.items():
                                if isinstance(group_key, tuple):
                                    # Multiple grouping columns
                                    group_mask = True
                                    for i, col in enumerate(available_group_cols):
                                        group_mask &= (cleaned_df[col] == group_key[i])
                                else:
                                    # Single grouping column
                                    group_mask = (cleaned_df[available_group_cols[0]] == group_key)
                                
                                outlier_mask = group_mask & (~exemption_mask) & (cleaned_df[spend_col] > cap_value)
                                outlier_count = outlier_mask.sum()
                                
                                if outlier_count > 0:
                                    cleaned_df.loc[outlier_mask, spend_col] = cap_value
                                    
                                    # Document caps by channel if enabled
                                    if outliers_config.get('document_caps_by_channel', True):
                                        group_str = f"{group_key}" if isinstance(group_key, tuple) else f"{group_key}"
                                        cleaning_actions.append(f"Winsorized {outlier_count} outliers in {spend_col} for group {group_str} at {cap_value:.2f}")
                                    
                                    logger.info(f"Contextual winsorization: {spend_col} group {group_key}, {outlier_count} values capped at {cap_value:.2f}")
                        except Exception as e:
                            logger.warning(f"Group-wise winsorization failed for {spend_col}: {e}")
                            # Fall back to global percentile
                            global_cap = non_exempt_data[spend_col].quantile(percentile_threshold)
                            outlier_mask = (~exemption_mask) & (cleaned_df[spend_col] > global_cap)
                            outlier_count = outlier_mask.sum()
                            if outlier_count > 0:
                                cleaned_df.loc[outlier_mask, spend_col] = global_cap
                                cleaning_actions.append(f"Global winsorization fallback: {outlier_count} outliers in {spend_col} at {global_cap:.2f}")
        
        # Document exemptions if enabled
        if outliers_config.get('document_exemptions', True):
            exemption_count = exemption_mask.sum()
            if exemption_count > 0:
                cleaned_df['outlier_exempted'] = exemption_mask.astype(int)
                cleaning_actions.append(f"Exempted {exemption_count} periods from outlier detection (promos/launches)")
        
        # Clean up temporary columns
        temp_cols = ['channel', 'month', 'quarter', 'weekday']
        for col in temp_cols:
            if col in cleaned_df.columns and col not in df.columns:  # Only remove if we created it
                cleaned_df.drop([col], axis=1, inplace=True)
    
    # ========================================================================
    # 3. SMART DUPLICATE HANDLING
    # ========================================================================
    logger.info("Starting smart duplicate handling...")
    
    duplicate_policy = duplicates_config.get('policy', 'smart_aggregation')
    if duplicate_policy == 'smart_aggregation':
        primary_key = duplicates_config.get('primary_key', ['date', 'brand', 'region'])
        
        # Map to actual column names
        key_cols = []
        for key in primary_key:
            if key == 'date':
                key_cols.append(config.data.date_col)
            elif key == 'brand' and 'ORGANISATION_ID' in cleaned_df.columns:
                key_cols.append('ORGANISATION_ID')
            elif key == 'region' and 'TERRITORY_NAME' in cleaned_df.columns:
                key_cols.append('TERRITORY_NAME')
        
        # Check for duplicates
        duplicate_mask = cleaned_df.duplicated(subset=key_cols, keep=False)
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count > 0:
            logger.info(f"Found {duplicate_count} duplicate records, applying smart aggregation...")
            
            # Advanced aggregation configuration
            numeric_aggregation = duplicates_config.get('numeric_aggregation', 'sum')
            string_aggregation = duplicates_config.get('string_aggregation', 'latest_timestamp')
            timestamp_column = duplicates_config.get('timestamp_column', 'ingestion_ts')
            
            # Identify additive vs non-additive columns with enhanced patterns
            additive_patterns = duplicates_config.get('additive_columns', [
                '.*_SPEND$', '.*_REVENUE$', '.*_CLICKS$', '.*_IMPRESSIONS$', '.*_CONVERSIONS$'
            ])
            non_additive_patterns = duplicates_config.get('non_additive_columns', [
                '.*_CPM$', '.*_CPC$', '.*_CTR$', '.*_CVR$'
            ])
            
            # Categorize columns
            additive_cols = []
            non_additive_cols = []
            string_cols = []
            timestamp_cols = []
            
            for col in cleaned_df.columns:
                if col not in key_cols:
                    is_additive = any(re.match(pattern, col) for pattern in additive_patterns)
                    is_non_additive = any(re.match(pattern, col) for pattern in non_additive_patterns)
                    
                    if is_additive:
                        additive_cols.append(col)
                    elif is_non_additive:
                        non_additive_cols.append(col)
                    elif cleaned_df[col].dtype in ['object', 'string']:
                        string_cols.append(col)
                    elif 'timestamp' in col.lower() or 'time' in col.lower() or col == timestamp_column:
                        timestamp_cols.append(col)
            
            # Perform smart aggregation with advanced rules
            agg_dict = {}
            
            # Handle additive columns based on configuration
            for col in additive_cols:
                if cleaned_df[col].dtype in ['float64', 'int64']:
                    if numeric_aggregation == 'sum':
                        agg_dict[col] = 'sum'
                    elif numeric_aggregation == 'mean':
                        agg_dict[col] = 'mean'
                    elif numeric_aggregation == 'max':
                        agg_dict[col] = 'max'
                    elif numeric_aggregation == 'latest_timestamp' and timestamp_column in cleaned_df.columns:
                        # Complex aggregation - take value from row with latest timestamp
                        agg_dict[col] = lambda x: x.iloc[-1]  # Will be handled specially
                    else:
                        agg_dict[col] = 'sum'  # Default fallback
            
            # Handle non-additive columns (typically ratios/rates)
            for col in non_additive_cols:
                if cleaned_df[col].dtype in ['float64', 'int64']:
                    agg_dict[col] = 'mean'  # Average for ratios makes sense
            
            # Handle string columns based on configuration
            for col in string_cols:
                if string_aggregation == 'latest_timestamp' and timestamp_column in cleaned_df.columns:
                    agg_dict[col] = 'last'  # Take the latest
                elif string_aggregation == 'first':
                    agg_dict[col] = 'first'
                else:
                    agg_dict[col] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common value
            
            # Handle timestamp columns
            for col in timestamp_cols:
                agg_dict[col] = 'max'  # Latest timestamp
            
            # Take first for remaining columns
            for col in cleaned_df.columns:
                if col not in key_cols and col not in agg_dict:
                    agg_dict[col] = 'first'
            
            # Apply aggregation with special handling for timestamp-based logic
            if string_aggregation == 'latest_timestamp' and timestamp_column in cleaned_df.columns:
                # Sort by timestamp before aggregation to ensure 'last' gets the latest
                cleaned_df_sorted = cleaned_df.sort_values(timestamp_column)
                before_dedup = len(cleaned_df_sorted)
                cleaned_df = cleaned_df_sorted.groupby(key_cols).agg(agg_dict).reset_index()
                after_dedup = len(cleaned_df)
            else:
                # Standard aggregation
                before_dedup = len(cleaned_df)
                cleaned_df = cleaned_df.groupby(key_cols).agg(agg_dict).reset_index()
                after_dedup = len(cleaned_df)
            
            # Generate dedup report if enabled
            if duplicates_config.get('generate_dedup_report', True):
                dedup_report = {
                    "before_dedup_rows": before_dedup,
                    "after_dedup_rows": after_dedup,
                    "duplicate_groups_found": (before_dedup - after_dedup),
                    "aggregation_method": numeric_aggregation,
                    "string_resolution": string_aggregation,
                    "additive_columns": additive_cols,
                    "non_additive_columns": non_additive_cols,
                    "timestamp_column_used": timestamp_column if timestamp_column in cleaned_df.columns else None
                }
                
                # Save dedup report if requested
                if duplicates_config.get('include_in_cleaning_report', True):
                    cleaning_metadata["deduplication_report"] = dedup_report
            
            log_cleaning_action(
                f"Smart aggregation: {before_dedup} → {after_dedup} rows ({before_dedup - after_dedup} groups aggregated)",
                level="standard",
                details={
                    "policy": "smart_aggregation",
                    "before_rows": before_dedup,
                    "after_rows": after_dedup,
                    "groups_aggregated": before_dedup - after_dedup,
                    "numeric_aggregation": numeric_aggregation,
                    "string_aggregation": string_aggregation,
                    "additive_columns": len(additive_cols),
                    "non_additive_columns": len(non_additive_cols)
                }
            )
            logger.info(f"Smart duplicate aggregation: {before_dedup} → {after_dedup} rows")
    
    # ========================================================================
    # 4. CAMPAIGN-AWARE MISSING DATA HANDLING
    # ========================================================================
    logger.info("Starting campaign-aware missing data handling...")
    
    spend_missing_policy = missing_data_config.get('spend_missing_policy', 'campaign_aware')
    outcome_missing_policy = missing_data_config.get('outcome_missing_policy', 'drop_periods')
    
    # Campaign-aware spend imputation
    if spend_missing_policy == 'campaign_aware':
        detect_pauses = missing_data_config.get('detect_campaign_pauses', True)
        pause_detection_method = missing_data_config.get('pause_detection_method', 'zero_impressions')
        
        if detect_pauses:
            pause_value = missing_data_config.get('paused_spend_value', 0.0)
            
            for spend_col in spend_cols:
                if spend_col in cleaned_df.columns:
                    # Find corresponding impression column
                    channel = spend_col.replace('_SPEND', '')
                    impression_cols = [col for col in cleaned_df.columns if channel in col and 'IMPRESSION' in col]
                    
                    missing_spend_mask = cleaned_df[spend_col].isnull()
                    
                    if len(impression_cols) > 0 and missing_spend_mask.any():
                        impression_col = impression_cols[0]
                        
                        if pause_detection_method == 'zero_impressions':
                            # If impressions are also 0/null, likely a campaign pause
                            zero_impressions_mask = (cleaned_df[impression_col] == 0) | cleaned_df[impression_col].isnull()
                            pause_mask = missing_spend_mask & zero_impressions_mask
                        elif pause_detection_method == 'spend_pattern':
                            # Detect pauses based on spend patterns
                            rolling_spend = cleaned_df[spend_col].rolling(window=7, min_periods=1).mean()
                            low_activity_mask = rolling_spend < (rolling_spend.quantile(0.1))
                            pause_mask = missing_spend_mask & low_activity_mask
                        else:
                            # Default to simple missing imputation
                            pause_mask = missing_spend_mask
                        
                        pause_count = pause_mask.sum()
                        if pause_count > 0:
                            cleaned_df.loc[pause_mask, spend_col] = pause_value
                            
                            log_cleaning_action(
                                f"Identified {pause_count} campaign pause periods in {spend_col}, imputed as {pause_value}",
                                level="standard",
                                details={
                                    "policy": "campaign_aware",
                                    "detection_method": pause_detection_method,
                                    "column": spend_col,
                                    "pause_count": pause_count,
                                    "imputed_value": pause_value,
                                    "corresponding_impression_col": impression_col if len(impression_cols) > 0 else None
                                }
                            )
                            logger.info(f"Campaign pause detection: {spend_col}, {pause_count} periods imputed as paused")
    
    # Preserve adstock continuity
    adstock_aware = missing_data_config.get('adstock_aware', True)
    if adstock_aware:
        # For MMM, missing spend should be treated as 0 to preserve adstock decay continuity
        for spend_col in spend_cols:
            if spend_col in cleaned_df.columns:
                missing_mask = cleaned_df[spend_col].isnull()
                missing_count = missing_mask.sum()
                
                if missing_count > 0:
                    cleaned_df.loc[missing_mask, spend_col] = 0.0
                    
                    # Create flag for adstock continuity preservation
                    if missing_data_config.get('mask_training_periods', True):
                        cleaned_df[f'adstock_continuity_{spend_col.replace("_SPEND", "").lower()}'] = missing_mask.astype(int)
                    
                    cleaning_actions.append(f"Preserved adstock continuity: imputed {missing_count} missing {spend_col} as 0")
                    logger.info(f"Adstock continuity: {spend_col}, {missing_count} missing values imputed as 0")
    
    # Handle outcome missing data
    if outcome_missing_policy == 'drop_periods':
        revenue_col_check = config.data.revenue_col
        if revenue_col_check in cleaned_df.columns:
            missing_revenue_mask = cleaned_df[revenue_col_check].isnull()
            missing_count = missing_revenue_mask.sum()
            
            if missing_count > 0:
                before_drop = len(cleaned_df)
                cleaned_df = cleaned_df[~missing_revenue_mask]
                after_drop = len(cleaned_df)
                
                cleaning_actions.append(f"Dropped {before_drop - after_drop} periods with missing revenue")
                logger.info(f"Outcome missing policy: dropped {before_drop - after_drop} periods with missing {revenue_col_check}")
    
    # ========================================================================
    # 5. GENERATE COMPREHENSIVE CLEANING REPORT
    # ========================================================================
    if documentation_config.get('generate_cleaning_report', True):
        # Calculate final metadata
        cleaning_metadata.update({
            "final_rows": len(cleaned_df),
            "final_columns": list(cleaned_df.columns),
            "rows_changed": len(df) - len(cleaned_df),
            "columns_added": len(cleaned_df.columns) - len(df.columns),
            "cleaning_actions": cleaning_actions,
            "detailed_actions": detailed_actions if document_all_changes else [],
            "total_cleaning_actions": len(cleaning_actions),
            "total_detailed_actions": len(detailed_actions) if document_all_changes else 0
        })
        
        # Advanced data hashing for reproducibility
        hash_algorithm = documentation_config.get('hash_algorithm', 'sha256')
        hash_raw_data = documentation_config.get('hash_raw_data', True)
        
        source_hash = None
        if hash_raw_data:
            try:
                import hashlib
                if hash_algorithm == 'sha256':
                    hasher = hashlib.sha256()
                elif hash_algorithm == 'md5':
                    hasher = hashlib.md5()
                else:
                    hasher = hashlib.sha256()  # fallback
                
                # Use stable pandas hashing for reproducibility
                digest_bytes = hash_pandas_object(df, index=True).values.tobytes()
                hasher.update(digest_bytes)
                source_hash = hasher.hexdigest()
            except Exception as e:
                logger.warning(f"Failed to hash raw data: {e}")
                source_hash = "hash_failed"
        
        # Save cleaning report in JSON format
        report_formats = documentation_config.get('report_format', ['json'])
        if 'json' in report_formats:
            # Create comprehensive report
            detailed_report = {
                "cleaning_metadata": cleaning_metadata,
                "validation_results_used": validation_results.get('overall_score', 0) if validation_results else 0,
                "data_quality_summary": {
                    "original_shape": df.shape,
                    "final_shape": cleaned_df.shape,
                    "missing_data_pct": cleaned_df.isnull().mean().mean() * 100,
                    "duplicate_records": 0,  # After cleaning
                    "negative_values_found": sum(1 for action in cleaning_actions if 'negative' in action.lower()),
                    "outliers_handled": sum(1 for action in cleaning_actions if 'outlier' in action.lower() or 'winsorize' in action.lower()),
                    "data_quality_flags_created": len([col for col in cleaned_df.columns if 'flag' in col.lower()]),
                    "adjustment_columns_created": len([col for col in cleaned_df.columns if '_ADJUSTMENTS' in col or '_RETURNS' in col])
                },
                "data_lineage": {
                    "source_hash": source_hash,
                    "transformation_steps": cleaning_actions,
                    "input_columns": list(df.columns),
                    "output_columns": list(cleaned_df.columns),
                    "columns_added": [col for col in cleaned_df.columns if col not in df.columns],
                    "columns_removed": [col for col in df.columns if col not in cleaned_df.columns],
                    "transformation_timestamp": datetime.now().isoformat()
                }
            }
            
            # Save cleaning report
            try:
                interim_path = Path(config.paths.interim)
                report_path = interim_path / "transformation_report.json"
                with open(report_path, 'w') as f:
                    json.dump(detailed_report, f, indent=2, default=str)
                logger.info(f"Generated transformation report: {report_path}")
            except Exception as e:
                logger.error(f"Failed to save transformation report: {e}")
    
    logger.info(f"Data transformation completed: {len(cleaning_actions)} actions taken")
    return cleaned_df


if __name__ == "__main__":
    # Test the transformation module
    try:
        from src.mmm.config import load_config
        config = load_config("config/main.yaml", profile="local")
        results = transform_data(config)
        
        print(f"Transformation completed")
        print(f"Shape change: {results['original_shape']} → {results['transformed_shape']}")
        print(f"Rows changed: {results['rows_changed']}")
        print(f"Columns added: {results['columns_added']}")
    except ImportError:
        print("Module not properly installed - skipping test run")
