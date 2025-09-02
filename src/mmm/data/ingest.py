"""
Data ingestion module for MMM project.

Handles loading data from various sources including Figshare CSV files
and future API connectors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from ..utils.logging import StepLogger


def ingest_figshare(config, slice_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Ingest data from Figshare CSV file.
    
    Args:
        config: Configuration object
        slice_hint: Optional hint for data slicing (e.g., 'last_12_weeks')
        
    Returns:
        pd.DataFrame: Raw ingested data
    """
    logger = logging.getLogger(__name__)
    
    with StepLogger("ingest_figshare"):
        # Construct file path
        raw_path = Path(config.paths.raw)
        figshare_config = config.ingest.get('figshare', {})
        file_pattern = figshare_config.get('file_pattern', 'conjura_mmm_data.csv')
        
        file_path = raw_path / file_pattern
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        # Load CSV data
        df = pd.read_csv(file_path)
        
        # Basic data info
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Date range: {df[config.data.date_col].min()} to {df[config.data.date_col].max()}")
        
        # Apply slice hint if provided
        if slice_hint:
            df = apply_slice_hint(df, slice_hint, config.data.date_col)
            logger.info(f"After slicing ({slice_hint}): {len(df)} rows")
        
        # Basic validation
        required_cols = [
            config.data.date_col,
            config.data.revenue_col,
            config.data.volume_col
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column
        df[config.data.date_col] = pd.to_datetime(df[config.data.date_col])
        
        # Save to interim
        interim_path = Path(config.paths.interim)
        interim_path.mkdir(parents=True, exist_ok=True)
        output_file = interim_path / "raw_ingested.parquet"
        df.to_parquet(output_file, index=False)
        
        logger.info(f"Saved ingested data to {output_file}")
        
        return df


def apply_slice_hint(df: pd.DataFrame, slice_hint: str, date_col: str) -> pd.DataFrame:
    """
    Apply data slicing based on hint.
    
    Args:
        df: Input dataframe
        slice_hint: Slicing hint (e.g., 'last_12_weeks', 'last_6_months')
        date_col: Name of date column
        
    Returns:
        pd.DataFrame: Sliced dataframe
    """
    df_copy = df.copy()
    
    # Ensure date column is datetime
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Find the latest date
    max_date = df_copy[date_col].max()
    
    if slice_hint.startswith('last_') and slice_hint.endswith('_weeks'):
        # Extract number of weeks
        num_weeks = int(slice_hint.split('_')[1])
        cutoff_date = max_date - timedelta(weeks=num_weeks)
        return df_copy[df_copy[date_col] >= cutoff_date]
    
    elif slice_hint.startswith('last_') and slice_hint.endswith('_months'):
        # Extract number of months (approximate)
        num_months = int(slice_hint.split('_')[1])
        cutoff_date = max_date - timedelta(days=num_months * 30)
        return df_copy[df_copy[date_col] >= cutoff_date]
    
    elif slice_hint.startswith('last_') and slice_hint.endswith('_days'):
        # Extract number of days
        num_days = int(slice_hint.split('_')[1])
        cutoff_date = max_date - timedelta(days=num_days)
        return df_copy[df_copy[date_col] >= cutoff_date]
    
    else:
        logging.warning(f"Unknown slice hint: {slice_hint}. Returning full dataset.")
        return df_copy


def ingest_api(config, source: str, since: datetime, until: datetime) -> pd.DataFrame:
    """
    Placeholder for future API ingestion capabilities.
    
    Args:
        config: Configuration object
        source: API source name (e.g., 'google_ads', 'meta', 'ga4')
        since: Start date for data pull
        until: End date for data pull
        
    Returns:
        pd.DataFrame: API data
    """
    raise NotImplementedError("API ingestion not yet implemented")


def get_data_summary(df: pd.DataFrame, config) -> Dict[str, Any]:
    """
    Generate summary statistics for ingested data.
    
    Args:
        df: Input dataframe
        config: Configuration object
        
    Returns:
        Dict: Summary statistics
    """
    date_col = config.data.date_col
    revenue_col = config.data.revenue_col
    
    summary = {
        'total_rows': len(df),
        'date_range': {
            'start': df[date_col].min().isoformat(),
            'end': df[date_col].max().isoformat(),
            'days': (df[date_col].max() - df[date_col].min()).days
        },
        'revenue_stats': {
            'total': df[revenue_col].sum(),
            'mean_daily': df.groupby(date_col)[revenue_col].sum().mean(),
            'missing_pct': df[revenue_col].isna().mean() * 100
        },
        'channels': {},
        'territories': df['TERRITORY_NAME'].unique().tolist() if 'TERRITORY_NAME' in df.columns else [],
        'organizations': df['ORGANISATION_ID'].nunique() if 'ORGANISATION_ID' in df.columns else 0
    }
    
    # Analyze spend channels
    spend_cols = [col for col in df.columns if col.endswith('_SPEND')]
    for col in spend_cols:
        summary['channels'][col] = {
            'total_spend': df[col].sum(),
            'mean_daily': df.groupby(date_col)[col].sum().mean(),
            'missing_pct': df[col].isna().mean() * 100,
            'zero_pct': (df[col] == 0).mean() * 100
        }
    
    return summary


if __name__ == "__main__":
    # Test the ingestion module
    from ..config import load_config
    
    config = load_config("config/main.yaml", profile="local")
    df = ingest_figshare(config, slice_hint="last_4_weeks")
    summary = get_data_summary(df, config)
    
    print(f"Ingested {summary['total_rows']} rows")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Total revenue: ${summary['revenue_stats']['total']:,.2f}")
