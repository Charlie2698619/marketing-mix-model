"""
Data validation module for MMM project.

Performs data quality checks, anomaly detection, and schema validation.
This module focuses on validating data quality without applying transformations.
Data cleaning and transformation are handled in the separate transform.py module.
"""

import pandas as pd
import numpy as np
import json
import hashlib
import logging
from pandas.util import hash_pandas_object
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import warnings
import re

# Optional imports with fallbacks
try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    logging.warning("Pandera not available - enhanced validation features disabled")

from ..utils.logging import StepLogger, log_data_quality
from .transform import normalize_and_resolve_columns


def _detect_cadence_and_window(df: pd.DataFrame, date_col: str) -> Tuple[str, int]:
    """Detect data cadence and return appropriate rolling window for anomaly detection."""
    dt = pd.to_datetime(df[date_col]).sort_values()
    if len(dt) < 3:
        return "weekly", 4
    
    step = dt.diff().dropna().mode()
    if len(step) == 0:
        return "weekly", 4
        
    step_days = step.iloc[0].days
    if 5 <= step_days <= 9:
        return "weekly", 4   # 4-week rolling for spikes/launches
    elif 1 <= step_days <= 2:
        return "daily", 7    # 7-day rolling
    return "weekly", 4


def validate_data(config) -> Dict[str, Any]:
    """
    Perform comprehensive data validation.
    
    Args:
        config: Configuration object
        
    Returns:
        Dict: Validation results
    """
    logger = logging.getLogger(__name__)
    
    with StepLogger("validate_data"):
        # Load ingested data
        interim_path = Path(config.paths.interim)
        input_file = interim_path / "raw_ingested.parquet"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        df = pd.read_parquet(input_file)
        
        df, col = normalize_and_resolve_columns(df, config)
        date_col   = col["date_col"]
        brand_col  = col["brand_col"]
        region_col = col["region_col"]
        revenue_col= col["revenue_col"]
        spend_cols = col["spend_cols"]
        
        logger.info(f"Loaded {len(df)} rows for validation")
        
        # Perform validation checks
        validation_results = {
            'schema_validation': validate_schema(df, config),
            'pandera_validation': validate_with_pandera(df, config),
            'data_quality': check_data_quality(df, config, col),
            'anomaly_detection': detect_anomalies(df, config),
            'business_rules': check_business_rules(df, config),
            'completeness': check_completeness(df, config),
            'digital_metrics': validate_digital_metrics(df, config),
            'coverage_frequency': check_coverage_frequency(df, config),
            'identifiability': check_identifiability(df, config),
            'value_sanity': check_value_sanity(df, config),
            'keys_duplicates': check_keys_duplicates(df, config)
        }
        
        # Calculate overall validation score
        validation_results['overall_score'] = calculate_validation_score(validation_results, config)
        validation_results['timestamp'] = datetime.now().isoformat()
        
        # Log results
        log_data_quality(validation_results)
        
        # Save validation report
        report_path = interim_path / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation completed. Overall score: {validation_results['overall_score']:.2f}")
        logger.info(f"Validation report saved to {report_path}")
        
        # Save data as validated (without cleaning - cleaning is now handled in transform step)
        output_file = interim_path / "validated_data.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"Validated data saved to {output_file} (cleaning will be performed in transform step)")
        
        return validation_results


def validate_schema(df: pd.DataFrame, config) -> Dict[str, Any]:
    """Validate data against expected schema."""
    logger = logging.getLogger(__name__)
    
    # Load schema
    schema_path = Path(config.validation.schema_ref)
    if not schema_path.exists():
        return {"status": "skipped", "reason": "Schema file not found"}
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    results = {
        "status": "passed",
        "errors": [],
        "warnings": []
    }
    
    # Check required columns
    required_cols = schema.get('columns', {}).get('required', [])
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        results["status"] = "failed"
        results["errors"].append(f"Missing required columns: {missing_cols}")
    
    # Check data types
    expected_types = schema.get('columns', {})
    for col, type_info in expected_types.items():
        if col in df.columns and isinstance(type_info, dict):
            expected_type = type_info.get('type')
            if expected_type == 'number' and not pd.api.types.is_numeric_dtype(df[col]):
                results["warnings"].append(f"Column {col} should be numeric")
            elif expected_type == 'string' and not pd.api.types.is_string_dtype(df[col]):
                results["warnings"].append(f"Column {col} should be string")
    
    return results


def check_data_quality(df: pd.DataFrame, config, col_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Check basic data quality metrics using resolved column names."""
    
    results = {
        "missing_data": {},
        "duplicates": {},
        "outliers": {},
        "data_types": {}
    }
    
    # Missing data analysis
    missing_pct = df.isnull().mean() * 100 
    results["missing_data"] = {
        "overall_missing_pct": missing_pct.mean(),
        "columns_with_missing": missing_pct[missing_pct > 0].to_dict(),
        "high_missing_cols": missing_pct[missing_pct > 50].index.tolist()
    }
    
    # Duplicate analysis using resolved columns
    date_col = col_mapping["date_col"]
    key_cols = [date_col]
    
    # Add brand/region keys if they exist and are resolved
    if col_mapping["brand_col"] and col_mapping["brand_col"] in df.columns:
        key_cols.append(col_mapping["brand_col"])
    if col_mapping["region_col"] and col_mapping["region_col"] in df.columns:
        key_cols.append(col_mapping["region_col"])
    
    duplicate_count = df.duplicated(subset=key_cols).sum()
    results["duplicates"] = {
        "duplicate_rows": int(duplicate_count),
        "duplicate_pct": duplicate_count / len(df) * 100
    }
    
    # Outlier detection for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Use enhanced value_sanity outlier_percentile instead of legacy outlier_threshold
    value_sanity_config = getattr(config.validation, 'value_sanity', {})
    outlier_percentile = value_sanity_config.get('outlier_percentile', 99)
    outlier_threshold = outlier_percentile / 100.0
    
    for col in numeric_cols:
        if col in df.columns:
            q_low = df[col].quantile(1 - outlier_threshold)
            q_high = df[col].quantile(outlier_threshold)
            outliers = ((df[col] < q_low) | (df[col] > q_high)).sum()
            
            results["outliers"][col] = {
                "outlier_count": int(outliers),
                "outlier_pct": outliers / len(df) * 100,
                "q_low": float(q_low),
                "q_high": float(q_high)
            }
    
    # Data type consistency
    for col in df.columns:
        results["data_types"][col] = str(df[col].dtype)
    
    return results


def detect_anomalies(df: pd.DataFrame, config) -> Dict[str, Any]:
    """Detect anomalies in the data."""
    
    # Use enhanced anomalies configuration 
    enhanced_cleaning = getattr(config.validation, 'enhanced_cleaning', {})
    anomalies_config = getattr(config.validation, 'anomalies', {})
    
    # Check if anomaly detection is enabled (prefer enhanced config, fallback to legacy)
    outage_detection_enabled = anomalies_config.get('outage_detection', False)
    
    if not outage_detection_enabled:
        return {"status": "skipped"}
    
    results = {
        "outages": [],
        "spikes": [],
        "campaign_launches": []
    }
    
    date_col = config.data.date_col
    revenue_col = config.data.revenue_col
    
    # Aggregate daily data
    daily_data = df.groupby(date_col).agg({
        revenue_col: 'sum',
        **{col: 'sum' for col in df.columns if col.endswith('_SPEND')}
    }).reset_index()
    
    # Detect revenue outages (days with zero or very low revenue)
    revenue_threshold = daily_data[revenue_col].quantile(0.1)
    outages = daily_data[daily_data[revenue_col] < revenue_threshold]
    
    for _, row in outages.iterrows():
        results["outages"].append({
            "date": row[date_col].isoformat(),
            "revenue": float(row[revenue_col]),
            "severity": "low" if row[revenue_col] > 0 else "high"
        })
    
    # Detect spending spikes
    spend_cols = [col for col in daily_data.columns if col.endswith('_SPEND')]
    for col in spend_cols:
        if daily_data[col].sum() > 0:  # Only check channels with spend
            q95 = daily_data[col].quantile(0.95)
            spikes = daily_data[daily_data[col] > q95 * 2]  # 2x the 95th percentile
            
            for _, row in spikes.iterrows():
                results["spikes"].append({
                    "date": row[date_col].isoformat(),
                    "channel": col,
                    "spend": float(row[col]),
                    "threshold": float(q95)
                })
    
    # Detect campaign launches (sudden spend increases)
    for col in spend_cols:
        if daily_data[col].sum() > 0:
            daily_data[f"{col}_rolling_mean"] = daily_data[col].rolling(7).mean() 
            daily_data[f"{col}_pct_change"] = daily_data[col].pct_change()
            
            launches = daily_data[
                (daily_data[f"{col}_pct_change"] > 2.0) &  # 200% increase
                (daily_data[col] > daily_data[f"{col}_rolling_mean"] * 1.5)  # Above recent average
            ]
            
            for _, row in launches.iterrows():
                results["campaign_launches"].append({
                    "date": row[date_col].isoformat(),
                    "channel": col,
                    "spend": float(row[col]),
                    "pct_change": float(row[f"{col}_pct_change"])
                })
    
    return results


def validate_digital_metrics(df: pd.DataFrame, config) -> Dict[str, Any]:
    """Validate digital engagement metrics availability and quality."""
    logger = logging.getLogger(__name__)
    
    
    
    digital_specific = getattr(config.data, "digital_specific", {}) or {}
    if hasattr(digital_specific, "dict"):
        digital_specific = digital_specific.dict()
    elif hasattr(digital_specific, "model_dump"):
        digital_specific = digital_specific.model_dump()

    platform_metrics = digital_specific.get("platform_metrics", {})
    cols = set(df.columns)  # already lowercased

    available_clicks      = [c for c in platform_metrics.get('clicks', []) if c.lower() in cols]
    missing_clicks        = [c for c in platform_metrics.get('clicks', []) if c.lower() not in cols]
    available_impressions = [c for c in platform_metrics.get('impressions', []) if c.lower() in cols]
    missing_impressions   = [c for c in platform_metrics.get('impressions', []) if c.lower() not in cols]
    available_organic     = [c for c in platform_metrics.get('organic_traffic', []) if c.lower() in cols]
    missing_organic       = [c for c in platform_metrics.get('organic_traffic', []) if c.lower() not in cols]

    
    
    # Get quality thresholds configuration
    quality_thresholds = getattr(config.validation, 'quality_thresholds', {})
    min_digital_coverage_pct = quality_thresholds.get('min_digital_coverage_pct', 80.0)
    min_digital_quality_score = quality_thresholds.get('min_digital_quality_score', 75.0)
    
    results = {
        "status": "passed",
        "available_metrics": {},
        "missing_metrics": {},
        "quality_scores": {},
        "warnings": []
    }
    

    
    # Check clicks metrics
    expected_clicks = platform_metrics.get('clicks', [])
    available_clicks = [col for col in expected_clicks if col in df.columns]
    missing_clicks = [col for col in expected_clicks if col not in df.columns]
    
    results["available_metrics"]["clicks"] = {
        "expected": len(expected_clicks),
        "available": len(available_clicks),
        "missing": len(missing_clicks),
        "coverage_pct": (len(available_clicks) / len(expected_clicks) * 100) if expected_clicks else 100
    }
    results["missing_metrics"]["clicks"] = missing_clicks
    
    # Check impressions metrics
    expected_impressions = platform_metrics.get('impressions', [])
    available_impressions = [col for col in expected_impressions if col in df.columns]
    missing_impressions = [col for col in expected_impressions if col not in df.columns]
    
    results["available_metrics"]["impressions"] = {
        "expected": len(expected_impressions),
        "available": len(available_impressions),
        "missing": len(missing_impressions),
        "coverage_pct": (len(available_impressions) / len(expected_impressions) * 100) if expected_impressions else 100
    }
    results["missing_metrics"]["impressions"] = missing_impressions
    
    # Check organic traffic metrics
    expected_organic = platform_metrics.get('organic_traffic', [])
    available_organic = [col for col in expected_organic if col in df.columns]
    missing_organic = [col for col in expected_organic if col not in df.columns]
    
    results["available_metrics"]["organic_traffic"] = {
        "expected": len(expected_organic),
        "available": len(available_organic),
        "missing": len(missing_organic),
        "coverage_pct": (len(available_organic) / len(expected_organic) * 100) if expected_organic else 100
    }
    results["missing_metrics"]["organic_traffic"] = missing_organic
    
    # Calculate quality scores for available metrics
    all_digital_cols = available_clicks + available_impressions + available_organic
    
    for col in all_digital_cols:
        quality_score = 100.0
        
        # Check for missing values
        missing_pct = df[col].isnull().mean() * 100
        quality_score -= missing_pct * 0.5  # Penalize missing values
        
        # Check for negative values (shouldn't exist in digital metrics)
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            quality_score -= 20  # Heavy penalty for negative digital metrics
            results["warnings"].append(f"{col} has {negative_count} negative values")
        
        # Check for zero variance (might indicate data issues)
        if df[col].var() == 0:
            quality_score -= 30  # Penalty for zero variance
            results["warnings"].append(f"{col} has zero variance")
        
        results["quality_scores"][col] = max(0, quality_score)
    
    # Overall digital metrics coverage
    total_expected = len(expected_clicks) + len(expected_impressions) + len(expected_organic)
    total_available = len(available_clicks) + len(available_impressions) + len(available_organic)
    overall_coverage = (total_available / total_expected * 100) if total_expected > 0 else 100
    
    results["overall_coverage_pct"] = overall_coverage
    
    # Calculate overall quality score 
    quality_scores = list(results["quality_scores"].values())
    overall_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 100.0
    results["overall_quality_score"] = overall_quality_score
    
    # Set status based on configured thresholds
    if overall_coverage < min_digital_coverage_pct * 0.6:  # 60% of minimum threshold
        results["status"] = "failed"
        results["warnings"].append(f"Low digital metrics coverage: {overall_coverage:.1f}% (minimum: {min_digital_coverage_pct}%)")
    elif overall_coverage < min_digital_coverage_pct:
        results["status"] = "warning"
        results["warnings"].append(f"Below preferred digital metrics coverage: {overall_coverage:.1f}% (minimum: {min_digital_coverage_pct}%)")
    
    # Also check quality score
    if overall_quality_score < min_digital_quality_score * 0.6:  # 60% of minimum threshold
        results["status"] = "failed"
        results["warnings"].append(f"Low digital metrics quality: {overall_quality_score:.1f} (minimum: {min_digital_quality_score})")
    elif overall_quality_score < min_digital_quality_score:
        if results["status"] != "failed":  # Don't downgrade from failed to warning
            results["status"] = "warning"
        results["warnings"].append(f"Below preferred digital metrics quality: {overall_quality_score:.1f} (minimum: {min_digital_quality_score})")
    
    logger.info(f"Digital metrics validation: {overall_coverage:.1f}% coverage, {len(results['warnings'])} warnings")
    
    return results


def check_business_rules(df: pd.DataFrame, config) -> Dict[str, Any]:
    """Check business logic rules."""
    
    # Get quality thresholds configuration
    quality_thresholds = getattr(config.validation, 'quality_thresholds', {})
    roi_range = quality_thresholds.get('roi_range', [0.1, 50])
    
    results = {
        "revenue_consistency": {},
        "spend_consistency": {},
        "roi_plausibility": {}
    }
    
    date_col = config.data.date_col
    revenue_col = config.data.revenue_col
    
    # Revenue consistency checks
    negative_revenue = (df[revenue_col] < 0).sum()
    zero_revenue_days = df.groupby(date_col)[revenue_col].sum()
    zero_revenue_count = (zero_revenue_days == 0).sum()
    
    results["revenue_consistency"] = {
        "negative_revenue_rows": int(negative_revenue),
        "zero_revenue_days": int(zero_revenue_count),
        "revenue_range": {
            "min": float(df[revenue_col].min()),
            "max": float(df[revenue_col].max())
        }
    }
    
    # Spend consistency checks  
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    for col in spend_cols:
        negative_spend = (df[col] < 0).sum()
        results["spend_consistency"][col] = {
            "negative_spend_rows": int(negative_spend),
            "spend_range": {
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        }
    
    # ROI plausibility (basic check)
    daily_data = df.groupby(date_col).agg({
        revenue_col: 'sum',
        **{col: 'sum' for col in spend_cols}
    })
    
    for col in spend_cols:
        if daily_data[col].sum() > 0:
            implied_roi = daily_data[revenue_col].sum() / daily_data[col].sum()
            is_plausible = roi_range[0] <= implied_roi <= roi_range[1]
            results["roi_plausibility"][col] = {
                "implied_roi": float(implied_roi),
                "plausible": is_plausible,
                "roi_range_used": roi_range  # Document which range was used
            }
    
    return results


def check_completeness(df: pd.DataFrame, config) -> Dict[str, Any]:
    """Check data completeness across time periods and segments."""
    
    date_col = config.data.date_col
    
    # Create date range
    date_range = pd.date_range(
        start=df[date_col].min(),
        end=df[date_col].max(),
        freq='D'
    )
    
    # Check date completeness
    existing_dates = df[date_col].dt.date.unique()
    expected_dates = date_range.date
    missing_dates = set(expected_dates) - set(existing_dates)
    
    results = {
        "date_completeness": {
            "expected_days": len(expected_dates),
            "actual_days": len(existing_dates),
            "missing_days": len(missing_dates),
            "completeness_pct": len(existing_dates) / len(expected_dates) * 100
        }
    }
    
    # Check completeness by segment
    if 'ORGANISATION_ID' in df.columns:
        orgs = df['ORGANISATION_ID'].unique()
        org_completeness = {}
        
        for org in orgs:
            org_data = df[df['ORGANISATION_ID'] == org]
            org_dates = org_data[date_col].dt.date.unique()
            org_completeness[org] = len(org_dates) / len(expected_dates) * 100
        
        results["organization_completeness"] = {
            "mean_completeness": np.mean(list(org_completeness.values())),
            "min_completeness": min(org_completeness.values()),
            "organizations_below_90pct": [org for org, pct in org_completeness.items() if pct < 90]
        }
    
    return results


def calculate_validation_score(validation_results: Dict[str, Any], config) -> float:
    """Calculate overall validation score (0-1) including enhanced quality checks."""
    
    # Get quality thresholds configuration
    quality_thresholds = getattr(config.validation, 'quality_thresholds', {})
    max_missing_data_pct = quality_thresholds.get('max_missing_data_pct', 20.0)
    max_high_missing_cols = quality_thresholds.get('max_high_missing_cols', 5)
    min_digital_coverage_pct = quality_thresholds.get('min_digital_coverage_pct', 80.0)
    min_digital_quality_score = quality_thresholds.get('min_digital_quality_score', 75.0)
    max_roi_implausible_channels = quality_thresholds.get('max_roi_implausible_channels', 2)
    roi_range = quality_thresholds.get('roi_range', [0.1, 50])
    min_validation_score = quality_thresholds.get('min_validation_score', 0.8)
    warning_validation_score = quality_thresholds.get('warning_validation_score', 0.9)
    
    score = 1.0
    
    # Schema validation
    if validation_results['schema_validation']['status'] == 'failed':
        score -= 0.15
    elif validation_results['schema_validation']['status'] == 'warning':
        score -= 0.05
    
    # Data quality
    dq = validation_results['data_quality']
    if dq['missing_data']['overall_missing_pct'] > max_missing_data_pct:
        score -= 0.15
    elif dq['missing_data']['overall_missing_pct'] > max_missing_data_pct / 2:  # Half threshold as warning
        score -= 0.05
    
    if dq['duplicates']['duplicate_pct'] > 5:
        score -= 0.05
    
    # Check for high missing columns using max_high_missing_cols
    if 'missing_data' in dq and 'column_stats' in dq['missing_data']:
        high_missing_cols = sum(1 for col_stat in dq['missing_data']['column_stats'] 
                               if col_stat.get('missing_pct', 0) > 50)
        if high_missing_cols > max_high_missing_cols:
            score -= 0.1
    
    # Digital metrics quality validation using min_digital_quality_score
    if 'digital_metrics' in dq:
        digital_quality = dq['digital_metrics'].get('quality_score', 100)
        if digital_quality < min_digital_quality_score:
            score -= 0.1
            
    # ROI plausibility validation using roi_range
    if 'roi_validation' in dq:
        implausible_channels = 0
        for channel_roi in dq['roi_validation'].get('channel_rois', []):
            if not (roi_range[0] <= channel_roi <= roi_range[1]):
                implausible_channels += 1
        if implausible_channels > max_roi_implausible_channels:
            score -= 0.15
    
    # Completeness
    if 'date_completeness' in validation_results['completeness']:
        completeness_pct = validation_results['completeness']['date_completeness']['completeness_pct']
        if completeness_pct < 80:
            score -= 0.1
        elif completeness_pct < 90:
            score -= 0.05
    
    # Coverage & Frequency (new)
    if 'coverage_frequency' in validation_results:
        cf = validation_results['coverage_frequency']
        if cf['status'] == 'failed':
            score -= 0.2  # Critical for MMM
        elif len(cf.get('warnings', [])) > 0:
            score -= 0.05
    
    # Identifiability (new)
    if 'identifiability' in validation_results:
        ident = validation_results['identifiability']
        if ident['status'] == 'failed':
            score -= 0.2  # Critical for MMM
        elif len(ident.get('always_on_channels', [])) > 0:
            score -= 0.1  # Penalize always-on channels
    
    # Value Sanity (new)
    if 'value_sanity' in validation_results:
        vs = validation_results['value_sanity']
        if vs['status'] == 'failed':
            score -= 0.15
        elif len(vs.get('errors', [])) > 0:
            score -= 0.05
    
    # Keys & Duplicates (new)
    if 'keys_duplicates' in validation_results:
        kd = validation_results['keys_duplicates']
        if kd['status'] == 'failed':
            score -= 0.1
    
    # Final score assessment using configured thresholds
    final_score = max(0.0, score)
    
    # Add validation status based on configured thresholds
    if final_score < min_validation_score:
        validation_results['overall_status'] = 'failed'
    elif final_score < warning_validation_score:
        validation_results['overall_status'] = 'warning'  
    else:
        validation_results['overall_status'] = 'passed'
    
    return final_score


def check_coverage_frequency(df: pd.DataFrame, config) -> Dict[str, Any]:
    """
    Check coverage and frequency requirements for MMM modeling.
    
    Weekly cadence, consistent across columns.
    History: ≥ 104 weeks preferred (≥ 52 weeks minimum) per brand×region.
    Completeness: ≤ 2% missing weeks per brand×region.
    """
    logger = logging.getLogger(__name__)
    
    results = {
        "status": "passed",
        "errors": [],
        "warnings": [],
        "weekly_cadence": {},
        "history_length": {},
        "completeness": {}
    }
    
    date_col = config.data.date_col
    
    # Convert to weekly data
    df_weekly = df.copy()
    df_weekly['week'] = pd.to_datetime(df_weekly[date_col]).dt.to_period('W')
    
    # Check weekly cadence consistency
    date_diffs = pd.to_datetime(df[date_col]).diff().dt.days
    daily_cadence = (date_diffs == 1).mean() > 0.95  # Allow some gaps
    weekly_cadence = (date_diffs == 7).mean() > 0.95
    
    results["weekly_cadence"] = {
        "is_daily": bool(daily_cadence),
        "is_weekly": bool(weekly_cadence),
        "cadence_consistency": float((date_diffs.mode()[0] if len(date_diffs.mode()) > 0 else 7))
    }
    
    # Get brand and region identifiers
    brand_col = 'ORGANISATION_ID' if 'ORGANISATION_ID' in df.columns else None
    region_col = 'TERRITORY_NAME' if 'TERRITORY_NAME' in df.columns else None
    
    if brand_col or region_col:
        # Group by brand x region
        group_cols = [col for col in [brand_col, region_col] if col is not None]
        groups = df_weekly.groupby(group_cols)['week'].nunique()
        
        # Calculate total possible weeks
        total_weeks = df_weekly['week'].nunique()
        
        # History length check (from config)
        coverage_config = getattr(config.validation, 'coverage_frequency', {})
        min_weeks_required = coverage_config.get('min_weeks_required', 52)
        preferred_weeks = coverage_config.get('preferred_weeks', 104)
        
        insufficient_history = (groups < min_weeks_required).sum()
        suboptimal_history = ((groups >= min_weeks_required) & (groups < preferred_weeks)).sum()
        optimal_history = (groups >= preferred_weeks).sum()
        
        results["history_length"] = {
            "total_groups": len(groups),
            "insufficient_history": int(insufficient_history),
            "suboptimal_history": int(suboptimal_history),
            "optimal_history": int(optimal_history),
            "min_weeks": int(groups.min()),
            "max_weeks": int(groups.max()),
            "mean_weeks": float(groups.mean())
        }
        
        # Completeness check (≤ max_missing_weeks_pct per brand×region)
        max_missing_weeks_pct = coverage_config.get('max_missing_weeks_pct', 2.0)
        min_completeness_threshold = 100.0 - max_missing_weeks_pct
        
        completeness_pct = (groups / total_weeks) * 100
        poor_completeness = (completeness_pct < min_completeness_threshold).sum()
        
        results["completeness"] = {
            "mean_completeness": float(completeness_pct.mean()),
            "min_completeness": float(completeness_pct.min()),
            "max_missing_weeks_pct_threshold": max_missing_weeks_pct,
            "min_completeness_threshold": min_completeness_threshold,
            f"groups_below_{min_completeness_threshold:.0f}pct": int(poor_completeness),
            "completeness_distribution": {
                "q25": float(completeness_pct.quantile(0.25)),
                "q50": float(completeness_pct.quantile(0.50)),
                "q75": float(completeness_pct.quantile(0.75))
            }
        }
        
        # Set status based on checks
        if insufficient_history > 0:
            results["status"] = "failed"
            results["errors"].append(f"{insufficient_history} brand×region groups have <{min_weeks_required} weeks of history")
        
        if poor_completeness > len(groups) * 0.1:  # More than 10% of groups have poor completeness
            results["status"] = "failed" if results["status"] != "failed" else results["status"]
            results["errors"].append(f"{poor_completeness} groups have >{max_missing_weeks_pct}% missing weeks")
        
        if suboptimal_history > 0:
            results["warnings"].append(f"{suboptimal_history} groups have {min_weeks_required}-{preferred_weeks-1} weeks (prefer ≥{preferred_weeks} weeks)")
    
    else:
        # Global checks when no brand/region breakdown
        total_weeks = df_weekly['week'].nunique()
        
        # Get config values
        coverage_config = getattr(config.validation, 'coverage_frequency', {})
        min_weeks_required = coverage_config.get('min_weeks_required', 52)
        preferred_weeks = coverage_config.get('preferred_weeks', 104)
        
        results["history_length"] = {
            "total_weeks": int(total_weeks),
            "sufficient_history": bool(total_weeks >= min_weeks_required),
            "optimal_history": bool(total_weeks >= preferred_weeks)
        }
        
        if total_weeks < min_weeks_required:
            results["status"] = "failed"
            results["errors"].append(f"Insufficient history: {total_weeks} weeks (minimum {min_weeks_required} required)")
        elif total_weeks < preferred_weeks:
            results["warnings"].append(f"Suboptimal history: {total_weeks} weeks (prefer ≥{preferred_weeks} weeks)")
    
    logger.info(f"Coverage/Frequency check: {results['status']}, {len(results['errors'])} errors, {len(results['warnings'])} warnings")
    return results


def check_identifiability(df: pd.DataFrame, config) -> Dict[str, Any]:
    """
    Check identifiability requirements for MMM modeling.
    
    - Each major channel should have on/off or level variation
    - ≥ 10–20% of weeks with near-zero spend for at least some markets
    - Pairwise correlations |ρ| < 0.9 between channels
    - Promotions not perfectly collinear with single channel
    """
    logger = logging.getLogger(__name__)
    
    results = {
        "status": "passed",
        "errors": [],
        "warnings": [],
        "channel_variation": {},
        "spend_correlations": {},
        "always_on_channels": [],
        "collinearity_issues": {}
    }
    
    # Get spend columns
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    
    if not spend_cols:
        results["status"] = "skipped"
        results["warnings"].append("No spend columns found for identifiability check")
        return results
    
    # Get identifiability configuration
    identifiability_config = getattr(config.validation, 'identifiability', {})
    min_zero_spend_weeks_pct = identifiability_config.get('min_zero_spend_weeks_pct', 10)
    preferred_zero_spend_weeks_pct = identifiability_config.get('preferred_zero_spend_weeks_pct', 20)
    max_channel_correlation = identifiability_config.get('max_channel_correlation', 0.9)
    max_promo_channel_correlation = identifiability_config.get('max_promo_channel_correlation', 0.95)
    coefficient_variation_threshold = identifiability_config.get('coefficient_variation_threshold', 0.1)
    
    # Check channel variation and "always on" issues
    eps = 1e-9  # True zero threshold for MMM identifiability
    
    for col in spend_cols:
        channel_data = df[col].fillna(0.0)
        
        # Calculate TRUE zero spend weeks (critical for MMM adstock identifiability)
        true_zero_weeks_pct = (channel_data <= eps).mean() * 100
        
        # Also calculate near-zero (5th percentile) for comparison
        near_zero_threshold = channel_data.quantile(0.05)
        near_zero_weeks_pct = (channel_data <= near_zero_threshold).mean() * 100
        
        # Check variation
        cv = channel_data.std() / channel_data.mean() if channel_data.mean() > 0 else 0
        
        results["channel_variation"][col] = {
            "true_zero_weeks_pct": float(true_zero_weeks_pct),
            "near_zero_weeks_pct": float(near_zero_weeks_pct),
            "coefficient_variation": float(cv),
            "min_spend": float(channel_data.min()),
            "max_spend": float(channel_data.max()),
            "mean_spend": float(channel_data.mean())
        }
        
        # Flag "always on" channels (using true zero weeks for MMM)
        if true_zero_weeks_pct < min_zero_spend_weeks_pct:
            results["always_on_channels"].append(col)
            if true_zero_weeks_pct < min_zero_spend_weeks_pct * 0.5:  # Very concerning (less than half minimum)
                results["errors"].append(f"{col}: Only {true_zero_weeks_pct:.1f}% true zero weeks (need ≥{min_zero_spend_weeks_pct}-{preferred_zero_spend_weeks_pct}%)")
            else:
                results["warnings"].append(f"{col}: Only {true_zero_weeks_pct:.1f}% true zero weeks (prefer ≥{preferred_zero_spend_weeks_pct}%)")
        
        # Check coefficient of variation
        if cv < coefficient_variation_threshold:
            results["warnings"].append(f"{col}: Low variation (CV={cv:.3f}, prefer ≥{coefficient_variation_threshold})")
    
    # Check pairwise correlations between spend channels
    if len(spend_cols) > 1:
        spend_df = df[spend_cols]
        corr_matrix = spend_df.corr()
        
        high_correlations = []
        for i, col1 in enumerate(spend_cols):
            for j, col2 in enumerate(spend_cols[i+1:], i+1):
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > max_channel_correlation:
                    high_correlations.append({
                        "channel1": col1,
                        "channel2": col2,
                        "correlation": float(corr_val)
                    })
                    results["errors"].append(f"High correlation: {col1} ↔ {col2} (ρ={corr_val:.3f}, threshold={max_channel_correlation})")
        
        results["spend_correlations"] = {
            "high_correlations": high_correlations,
            "max_correlation": float(corr_matrix.abs().where(~np.eye(len(spend_cols), dtype=bool)).max().max()),
            "correlation_matrix": corr_matrix.round(3).to_dict()
        }
    
    # Check for promotion collinearity if promotion columns exist
    promo_cols = [col for col in df.columns if 'PROMO' in col.upper() or 'PROMOTION' in col.upper()]
    
    if promo_cols and spend_cols:
        for promo_col in promo_cols:
            for spend_col in spend_cols:
                try:
                    # Calculate correlation between promotion and spend
                    promo_spend_corr = df[promo_col].corr(df[spend_col])
                    
                    if abs(promo_spend_corr) > max_promo_channel_correlation:
                        results["collinearity_issues"][f"{promo_col}_{spend_col}"] = {
                            "correlation": float(promo_spend_corr),
                            "issue": "promotion_channel_collinearity"
                        }
                        results["errors"].append(f"Promotion-channel collinearity: {promo_col} ↔ {spend_col} (ρ={promo_spend_corr:.3f}, threshold={max_promo_channel_correlation})")
                        
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {promo_col} and {spend_col}: {e}")
    
    # Set overall status
    if results["errors"]:
        results["status"] = "failed"
    elif len(results["always_on_channels"]) > len(spend_cols) * 0.5:  # More than 50% always on
        results["status"] = "warning"
        results["warnings"].append(f"{len(results['always_on_channels'])}/{len(spend_cols)} channels are 'always on'")
    
    logger.info(f"Identifiability check: {results['status']}, {len(results['always_on_channels'])} always-on channels")
    return results


def check_value_sanity(df: pd.DataFrame, config) -> Dict[str, Any]:
    """
    Check value sanity requirements.
    
    - No negative revenue/spend/clicks
    - Spend units consistent (same currency)
    - CPM ≈ spend / (impressions/1000) within ±10–15%
    - CPC ≈ spend / clicks within ±10–15%
    - Remove/winsorize top 1% outliers
    """
    logger = logging.getLogger(__name__)
    
    results = {
        "status": "passed",
        "errors": [],
        "warnings": [],
        "negative_values": {},
        "unit_consistency": {},
        "cpm_validation": {},
        "cpc_validation": {},
        "outlier_analysis": {}
    }
    
    # Get value_sanity configuration
    value_sanity_config = getattr(config.validation, 'value_sanity', {})
    allow_negative_revenue = value_sanity_config.get('allow_negative_revenue', False)
    allow_negative_spend = value_sanity_config.get('allow_negative_spend', False)
    allow_negative_clicks = value_sanity_config.get('allow_negative_clicks', False)
    max_spend_order_difference = value_sanity_config.get('max_spend_order_difference', 3)
    cpm_tolerance_pct = value_sanity_config.get('cpm_tolerance_pct', 15)
    cpc_tolerance_pct = value_sanity_config.get('cpc_tolerance_pct', 15)
    max_discrepancy_rows_pct = value_sanity_config.get('max_discrepancy_rows_pct', 10)
    outlier_percentile = value_sanity_config.get('outlier_percentile', 99)
    document_outliers = value_sanity_config.get('document_outliers', True)
    
    # Check for negative values
    revenue_col = config.data.revenue_col
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    click_cols = [col for col in df.columns if 'CLICK' in col.upper()]
    impression_cols = [col for col in df.columns if 'IMPRESSION' in col.upper()]
    
    # Revenue negative check
    if not allow_negative_revenue:
        negative_revenue = (df[revenue_col] < 0).sum()
        if negative_revenue > 0:
            results["errors"].append(f"{negative_revenue} rows with negative revenue (not allowed)")
            results["negative_values"]["revenue"] = int(negative_revenue)
    
    # Spend negative check
    if not allow_negative_spend:
        for col in spend_cols:
            negative_spend = (df[col] < 0).sum()
            if negative_spend > 0:
                results["errors"].append(f"{negative_spend} rows with negative spend in {col} (not allowed)")
                results["negative_values"][col] = int(negative_spend)
    
    # Clicks negative check
    if not allow_negative_clicks:
        for col in click_cols:
            negative_clicks = (df[col] < 0).sum()
            if negative_clicks > 0:
                results["errors"].append(f"{negative_clicks} rows with negative clicks in {col} (not allowed)")
                results["negative_values"][col] = int(negative_clicks)
    
    # Spend unit consistency (check magnitude differences)
    if spend_cols:
        spend_means = {col: df[col].mean() for col in spend_cols}
        spend_orders = {col: np.log10(mean) if mean > 0 else 0 for col, mean in spend_means.items()}
        
        # Flag if spend columns differ by more than configured orders of magnitude
        min_order = min(spend_orders.values())
        max_order = max(spend_orders.values())
        
        results["unit_consistency"] = {
            "spend_means": spend_means,
            "log_orders": spend_orders,
            "order_difference": float(max_order - min_order),
            "max_allowed_difference": max_spend_order_difference
        }
        
        if max_order - min_order > max_spend_order_difference:
            results["warnings"].append(f"Spend units may be inconsistent: {max_order - min_order:.1f} order difference (max allowed: {max_spend_order_difference})")
    
    # CPM validation: CPM ≈ spend / (impressions/1000)
    cpm_discrepancies = []
    for spend_col in spend_cols:
        # Try to find matching impression column
        channel = spend_col.replace('_SPEND', '')
        matching_impressions = [col for col in impression_cols if channel in col]
        
        for imp_col in matching_impressions:
            # Calculate CPM from spend and impressions
            mask = (df[imp_col] > 0) & (df[spend_col] > 0)
            if mask.sum() > 0:
                calculated_cpm = (df.loc[mask, spend_col] / (df.loc[mask, imp_col] / 1000))
                
                # If there's a reported CPM column, compare
                cpm_col_candidates = [col for col in df.columns if 'CPM' in col and channel in col]
                
                for cpm_col in cpm_col_candidates:
                    reported_cpm = df.loc[mask, cpm_col]
                    pct_diff = abs((calculated_cpm - reported_cpm) / reported_cpm * 100)
                    
                    discrepancy_mask = pct_diff > 15  # More than 15% difference
                    discrepancy_count = discrepancy_mask.sum()
                    
                    if discrepancy_count > 0:
                        cpm_discrepancies.append({
                            "spend_col": spend_col,
                            "impression_col": imp_col,
                            "cpm_col": cpm_col,
                            "discrepancy_count": int(discrepancy_count),
                            "discrepancy_pct": float(discrepancy_count / len(reported_cpm) * 100),
                            "mean_pct_diff": float(pct_diff.mean())
                        })
                        
                        if discrepancy_count / len(reported_cpm) > max_discrepancy_rows_pct / 100:
                            results["errors"].append(f"CPM discrepancies in {spend_col}: {discrepancy_count} rows (>{max_discrepancy_rows_pct}% threshold)")
    
    results["cpm_validation"] = {
        "discrepancies": cpm_discrepancies,
        "channels_checked": len(cpm_discrepancies)
    }
    
    # CPC validation: CPC ≈ spend / clicks
    cpc_discrepancies = []
    for spend_col in spend_cols:
        channel = spend_col.replace('_SPEND', '')
        matching_clicks = [col for col in click_cols if channel in col]
        
        for click_col in matching_clicks:
            mask = (df[click_col] > 0) & (df[spend_col] > 0)
            if mask.sum() > 0:
                calculated_cpc = df.loc[mask, spend_col] / df.loc[mask, click_col]
                
                # If there's a reported CPC column, compare
                cpc_col_candidates = [col for col in df.columns if 'CPC' in col and channel in col]
                
                for cpc_col in cpc_col_candidates:
                    reported_cpc = df.loc[mask, cpc_col]
                    pct_diff = abs((calculated_cpc - reported_cpc) / reported_cpc * 100)
                    
                    discrepancy_mask = pct_diff > cpc_tolerance_pct
                    discrepancy_count = discrepancy_mask.sum()
                    
                    if discrepancy_count > 0:
                        cpc_discrepancies.append({
                            "spend_col": spend_col,
                            "click_col": click_col,
                            "cpc_col": cpc_col,
                            "discrepancy_count": int(discrepancy_count),
                            "discrepancy_pct": float(discrepancy_count / len(reported_cpc) * 100),
                            "mean_pct_diff": float(pct_diff.mean()),
                            "tolerance_pct": cpc_tolerance_pct
                        })
                        
                        if discrepancy_count / len(reported_cpc) > max_discrepancy_rows_pct / 100:
                            results["errors"].append(f"CPC discrepancies in {spend_col}: {discrepancy_count} rows (>{max_discrepancy_rows_pct}% threshold)")
    
    results["cpc_validation"] = {
        "discrepancies": cpc_discrepancies,
        "channels_checked": len(cpc_discrepancies)
    }
    
    # Outlier analysis (configurable percentile)
    outlier_threshold_decimal = outlier_percentile / 100.0
    
    # Revenue outliers
    revenue_q_threshold = df[revenue_col].quantile(outlier_threshold_decimal)
    revenue_outliers_mask = df[revenue_col] > revenue_q_threshold
    revenue_outliers = revenue_outliers_mask.sum()
    
    revenue_outlier_info = {
        "outlier_count": int(revenue_outliers),
        "outlier_threshold": float(revenue_q_threshold),
        "outlier_pct": float(revenue_outliers / len(df) * 100),
        "outlier_percentile": outlier_percentile
    }
    
    # Document outlier details if enabled
    if document_outliers and revenue_outliers > 0:
        outlier_rows = df[revenue_outliers_mask]
        revenue_outlier_info["outlier_details"] = {
            "max_outlier_value": float(outlier_rows[revenue_col].max()),
            "mean_outlier_value": float(outlier_rows[revenue_col].mean()),
            "outlier_dates": outlier_rows[config.data.date_col].dt.strftime('%Y-%m-%d').tolist()[:10],  # First 10 dates
            "documentation_enabled": document_outliers
        }
        logger.info(f"Documented {revenue_outliers} revenue outliers above {revenue_q_threshold:.2f}")
    
    results["outlier_analysis"]["revenue"] = revenue_outlier_info
    
    # Spend outliers
    for col in spend_cols:
        spend_q_threshold = df[col].quantile(outlier_threshold_decimal)
        spend_outliers_mask = df[col] > spend_q_threshold
        spend_outliers = spend_outliers_mask.sum()
        
        spend_outlier_info = {
            "outlier_count": int(spend_outliers),
            "outlier_threshold": float(spend_q_threshold),
            "outlier_pct": float(spend_outliers / len(df) * 100),
            "outlier_percentile": outlier_percentile
        }
        
        # Document outlier details if enabled
        if document_outliers and spend_outliers > 0:
            outlier_rows = df[spend_outliers_mask]
            spend_outlier_info["outlier_details"] = {
                "max_outlier_value": float(outlier_rows[col].max()),
                "mean_outlier_value": float(outlier_rows[col].mean()),
                "outlier_dates": outlier_rows[config.data.date_col].dt.strftime('%Y-%m-%d').tolist()[:10],  # First 10 dates
                "documentation_enabled": document_outliers
            }
            logger.info(f"Documented {spend_outliers} {col} outliers above {spend_q_threshold:.2f}")
        
        results["outlier_analysis"][col] = spend_outlier_info
    
    # Set status
    if results["errors"]:
        results["status"] = "failed"
    elif len(cpm_discrepancies) + len(cpc_discrepancies) > 0:
        results["status"] = "warning"
    
    logger.info(f"Value sanity check: {results['status']}, {len(results['errors'])} errors")
    return results


def check_keys_duplicates(df: pd.DataFrame, config) -> Dict[str, Any]:
    """
    Check keys and duplicates.
    
    - No duplicate (date, brand, region) records after aggregation
    - Stable brand & region names (use IDs if possible)
    """
    logger = logging.getLogger(__name__)
    
    results = {
        "status": "passed",
        "errors": [],
        "warnings": [],
        "duplicate_analysis": {},
        "key_stability": {}
    }
    
    # Get keys_duplicates configuration
    keys_duplicates_config = getattr(config.validation, 'keys_duplicates', {})
    allow_duplicates = keys_duplicates_config.get('allow_duplicates', False)
    check_name_consistency = keys_duplicates_config.get('check_name_consistency', True)
    prefer_ids_over_names = keys_duplicates_config.get('prefer_ids_over_names', True)
    similarity_threshold = keys_duplicates_config.get('similarity_threshold', 0.8)
    
    date_col = config.data.date_col
    
    # Define key columns
    key_cols = [date_col]
    brand_col = 'ORGANISATION_ID' if 'ORGANISATION_ID' in df.columns else None
    region_col = 'TERRITORY_NAME' if 'TERRITORY_NAME' in df.columns else None
    
    if brand_col:
        key_cols.append(brand_col)
    if region_col:
        key_cols.append(region_col)
    
    # Check for duplicates
    duplicate_mask = df.duplicated(subset=key_cols, keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0 and not allow_duplicates:
        results["errors"].append(f"{duplicate_count} duplicate (date, brand, region) records found (not allowed)")
        results["duplicate_analysis"] = {
            "duplicate_count": int(duplicate_count),
            "unique_duplicate_combinations": int(df[duplicate_mask].drop_duplicates(subset=key_cols).shape[0]),
            "duplicate_pct": float(duplicate_count / len(df) * 100),
            "allow_duplicates": allow_duplicates
        }
        results["status"] = "failed"
    elif duplicate_count > 0 and allow_duplicates:
        results["warnings"].append(f"{duplicate_count} duplicate records found (allowed by configuration)")
        results["duplicate_analysis"] = {
            "duplicate_count": int(duplicate_count),
            "unique_duplicate_combinations": int(df[duplicate_mask].drop_duplicates(subset=key_cols).shape[0]),
            "duplicate_pct": float(duplicate_count / len(df) * 100),
            "allow_duplicates": allow_duplicates
        }
    else:
        results["duplicate_analysis"] = {
            "duplicate_count": 0,
            "all_records_unique": True,
            "allow_duplicates": allow_duplicates
        }
    
    # Check key stability (consistent naming) - only if enabled
    if check_name_consistency:
        if brand_col:
            # Check for potential brand name inconsistencies
            brand_names = df[brand_col].unique()
            
            # Look for similar brand names that might be inconsistent
            similar_brands = []
            for i, brand1 in enumerate(brand_names):
                for brand2 in brand_names[i+1:]:
                    if isinstance(brand1, str) and isinstance(brand2, str):
                        # Enhanced similarity check using configured threshold
                        similarity_score = len(set(brand1.lower().split()).intersection(set(brand2.lower().split()))) / len(set(brand1.lower().split()).union(set(brand2.lower().split())))
                        if similarity_score >= similarity_threshold:
                            similar_brands.append((brand1, brand2, similarity_score))
            
            if similar_brands:
                results["warnings"].append(f"Found {len(similar_brands)} potentially inconsistent brand names (similarity ≥{similarity_threshold})")
                results["key_stability"]["similar_brands"] = similar_brands[:10]  # Limit output
            
            results["key_stability"]["brand_stats"] = {
                "unique_brands": len(brand_names),
                "has_null_brands": df[brand_col].isnull().any(),
                "prefer_ids_over_names": prefer_ids_over_names
            }
    
        if region_col and check_name_consistency:
            region_names = df[region_col].unique()
            
            # Similar check for regions
            similar_regions = []
            for i, region1 in enumerate(region_names):
                for region2 in region_names[i+1:]:
                    if isinstance(region1, str) and isinstance(region2, str):
                        similarity_score = len(set(region1.lower().split()).intersection(set(region2.lower().split()))) / len(set(region1.lower().split()).union(set(region2.lower().split())))
                        if similarity_score >= similarity_threshold:
                            similar_regions.append((region1, region2, similarity_score))
            
            if similar_regions:
                results["warnings"].append(f"Found {len(similar_regions)} potentially inconsistent region names (similarity ≥{similarity_threshold})")
                results["key_stability"]["similar_regions"] = similar_regions[:10]
            
            results["key_stability"]["region_stats"] = {
                "unique_regions": len(region_names),
                "has_null_regions": df[region_col].isnull().any(),
                "prefer_ids_over_names": prefer_ids_over_names
            }
    
    logger.info(f"Keys/Duplicates check: {results['status']}, {duplicate_count} duplicates")
    return results


def create_pandera_schema(config) -> Any:
    """Create Pandera schema for data validation."""
    if not PANDERA_AVAILABLE:
        raise ImportError("Pandera is required for schema validation. Install with: pip install pandera")
    
    # Get pandera configuration
    pandera_config = getattr(config.validation, 'pandera', {})
    if not pandera_config.get('enabled', False):
        return None
    
    # Build schema dynamically based on configuration
    schema_checks = {}
    
    # Date column validation
    date_col = getattr(config.data, "date_col", "date")
    
    from datetime import datetime, timezone
    _now = pd.Timestamp(datetime.now(tz=timezone.utc).date())
    
    schema_checks[date_col] = Column(
        dtype="datetime64[ns]",
        checks=[
            Check.not_null(),
            Check(lambda s: (s <= _now).all(), error="Date contains future timestamps")
        ]
    )
    
    # Revenue column validation
    revenue_col = getattr(config.data, "revenue_col", "revenue")
    schema_checks[revenue_col] = Column(
        dtype="float64",
        checks=[
            Check.ge(0, error="Revenue should be non-negative after adjustments"),
            Check.lt(1e9, error="Revenue seems unrealistically high")
        ]
    )
    
    # Spend columns validation
    ch_map = getattr(config.data, "channel_map", {}) or {}
    for _, raw in ch_map.items():
        schema_checks[raw] = Column("float64", checks=[Check.ge(0, error="Spend should be non-negative after credits")])

    
    # Create the schema
    schema = DataFrameSchema(
        columns=schema_checks,
        checks=[
            Check(lambda df: len(df) > 0, error="DataFrame cannot be empty")
        ]
    )
    
    return schema


def validate_with_pandera(df: pd.DataFrame, config) -> Dict[str, Any]:
    """Validate DataFrame using Pandera schema."""
    pandera_config = getattr(config.validation, 'pandera', {})
    
    if not pandera_config.get('enabled', False):
        return {"status": "skipped", "reason": "Pandera validation disabled"}
    
    try:
        schema = create_pandera_schema(config)
        if schema is None:
            return {"status": "skipped", "reason": "Schema creation failed"}
        
        # Validate the dataframe
        validated_df = schema.validate(df, lazy=True)
        
        return {
            "status": "passed",
            "schema_version": hash(str(schema)),
            "validated_rows": len(validated_df),
            "validated_columns": list(validated_df.columns)
        }
        
    except Exception as e:
        return {
            "status": "failed", 
            "errors": [{"error_message": str(e)}],
            "total_errors": 1
        }


if __name__ == "__main__":
    # Test the validation module
    try:
        from src.mmm.config import load_config
        config = load_config("config/main.yaml", profile="local")
        results = validate_data(config)
        
        print(f"Validation score: {results['overall_score']:.2f}")
        print(f"Schema validation: {results['schema_validation']['status']}")
        print(f"Missing data: {results['data_quality']['missing_data']['overall_missing_pct']:.1f}%")
        print("Validation completed. Data saved as validated_data.parquet (cleaning will be performed in transform step).")
    except ImportError:
        print("Module not properly installed - skipping test run")
