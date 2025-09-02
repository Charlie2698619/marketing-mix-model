"""
Feature Engineering Orchestrator

This module coordinates all feature engineering transformations in the correct order,
applying adstock, saturation, seasonality, and other feature transformations
according to the configuration.
"""

import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import load_config
from ..utils.logging import setup_logging, get_logger

# Import feature engineering modules
from .adstock import apply_adstock, apply_platform_adstock
from .saturation import apply_saturation_from_config, apply_platform_saturation
from .seasonality import apply_seasonality_from_config
from .baseline import generate_baseline_features_robust
from .attribution import apply_attribution_modeling
from .creative_fatigue import apply_creative_fatigue_from_config
from .competitors import apply_competitor_factors
from .custom_terms import apply_custom_business_terms


def engineer_features(
    data_df: pd.DataFrame,
    config: Optional[Any] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Orchestrate all feature engineering transformations.
    
    This function applies feature engineering transformations in the correct order:
    1. Baseline features (trend, seasonality, external events)
    2. Marketing transformations (adstock, saturation)
    3. Attribution modeling (multi-touch attribution)
    4. Creative fatigue effects
    5. Competitor and market factors
    6. Custom terms and interactions
    
    Args:
        data_df: Input dataframe with transformed/cleaned data
        config: Configuration object (if None, loads from main.yaml)
        output_path: Optional path to save engineered features
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe ready for modeling
    """
    logger = logging.getLogger(__name__)
    
    if config is None:
        config = load_config()
    
    logger.info("Starting comprehensive feature engineering...")
    
    # Start with a copy of the input data
    result_df = data_df.copy()
    
    # Store feature engineering metadata
    feature_metadata = {
        'input_shape': data_df.shape,
        'transformations_applied': [],
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        # Step 1: Generate baseline features (seasonality, trend, external events)
        logger.info("Step 1: Generating baseline features...")
        if hasattr(config, 'features') and hasattr(config.features, 'seasonality'):
            # Extract date column for seasonality features
            date_col = 'date_day'  # Standard date column name
            if date_col in result_df.columns:
                seasonality_features = apply_seasonality_from_config(result_df[date_col], config)
                # Merge seasonality features with main dataframe
                seasonality_features.index = result_df.index  # Ensure matching indices
                result_df = pd.concat([result_df, seasonality_features], axis=1)
                feature_metadata['transformations_applied'].append('seasonality')
                logger.info("✅ Seasonality features applied")
            else:
                logger.warning(f"Date column '{date_col}' not found, skipping seasonality features")
        
        if hasattr(config, 'features') and hasattr(config.features, 'baseline'):
            baseline_features = generate_baseline_features_robust(result_df, config)
            if baseline_features is not None and not baseline_features.empty:
                # Merge baseline features with main dataframe
                result_df = pd.concat([result_df, baseline_features], axis=1)
                feature_metadata['transformations_applied'].append('baseline')
                logger.info("✅ Baseline features applied")
        
        # Step 2: Apply marketing transformations (adstock, then saturation)
        logger.info("Step 2: Applying marketing transformations...")
        if hasattr(config, 'features') and hasattr(config.features, 'adstock'):
            result_df = apply_platform_adstock(result_df, config)
            feature_metadata['transformations_applied'].append('adstock')
            logger.info("✅ Adstock transformations applied")
        
        if hasattr(config, 'features') and hasattr(config.features, 'saturation'):
            try:
                result_df = apply_platform_saturation(result_df, config)
                feature_metadata['transformations_applied'].append('saturation')
                logger.info("✅ Saturation transformations applied")
            except Exception as e:
                logger.warning(f"Saturation transformation failed: {e}, skipping...")
        
        # Step 3: Apply attribution modeling
        logger.info("Step 3: Applying attribution modeling...")
        if hasattr(config, 'features') and hasattr(config.features, 'attribution'):
            try:
                # Attribution modeling may need interaction and conversion data
                # For now, we'll apply it to the main dataframe
                result_df = apply_attribution_modeling(result_df, None, config.__dict__)
                feature_metadata['transformations_applied'].append('attribution')
                logger.info("✅ Attribution modeling applied")
            except Exception as e:
                logger.warning(f"Attribution modeling failed: {e}, skipping...")
        
        # Step 4: Apply creative fatigue effects
        logger.info("Step 4: Applying creative fatigue effects...")
        if hasattr(config, 'features') and hasattr(config.features, 'creative_fatigue'):
            try:
                result_df = apply_creative_fatigue_from_config(result_df, config)
                feature_metadata['transformations_applied'].append('creative_fatigue')
                logger.info("✅ Creative fatigue effects applied")
            except Exception as e:
                logger.warning(f"Creative fatigue failed: {e}, skipping...")
        
        # Step 5: Apply competitor and market factors
        logger.info("Step 5: Applying competitor and market factors...")
        if hasattr(config, 'features') and hasattr(config.features, 'competitors'):
            try:
                result_df = apply_competitor_factors(result_df, config)
                feature_metadata['transformations_applied'].append('competitors')
                logger.info("✅ Competitor factors applied")
            except Exception as e:
                logger.warning(f"Competitor factors failed: {e}, skipping...")
        
        # Step 6: Apply custom terms and interactions
        logger.info("Step 6: Applying custom terms...")
        if hasattr(config, 'features') and hasattr(config.features, 'custom_terms'):
            try:
                result_df = apply_custom_business_terms(result_df, config)
                feature_metadata['transformations_applied'].append('custom_terms')
                logger.info("✅ Custom terms applied")
            except Exception as e:
                logger.warning(f"Custom terms failed: {e}, skipping...")
        
        # Add final metadata
        feature_metadata['output_shape'] = result_df.shape
        feature_metadata['features_added'] = result_df.shape[1] - data_df.shape[1]
        result_df.attrs['feature_engineering'] = feature_metadata
        
        logger.info(f"Feature engineering completed successfully!")
        logger.info(f"Input shape: {feature_metadata['input_shape']}")
        logger.info(f"Output shape: {feature_metadata['output_shape']}")
        logger.info(f"Features added: {feature_metadata['features_added']}")
        logger.info(f"Transformations applied: {', '.join(feature_metadata['transformations_applied'])}")
        
        # Check for and handle duplicate columns
        if result_df.columns.duplicated().any():
            logger.warning(f"Found duplicate columns")
            # Get the actual duplicate column names for logging
            duplicate_mask = result_df.columns.duplicated(keep=False)
            duplicate_cols = result_df.columns[duplicate_mask].tolist()
            logger.warning(f"Duplicate column names: {duplicate_cols}")
            # Remove duplicate columns, keeping the first occurrence
            result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            logger.info(f"Removed duplicate columns, final shape: {result_df.shape}")
        
        # Save the results using the provided output path or config default
        if output_path:
            final_output_file = output_path
        else:
            # Get path from config
            features_path = getattr(config.paths, 'features', 'data/features')
            final_output_file = os.path.join(features_path, 'feature_set.parquet')
        os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
        result_df.to_parquet(final_output_file, index=False)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.error(f"Failed at transformations: {feature_metadata['transformations_applied']}")
        raise


def validate_feature_engineering_config(config: Any) -> Dict[str, Any]:
    """
    Validate that the configuration has all required sections for feature engineering.
    
    Args:
        config: Configuration object
        
    Returns:
        Dict[str, Any]: Validation results with recommendations
    """
    logger = logging.getLogger(__name__)
    
    validation_results = {
        'valid': True,
        'missing_sections': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check for main features section
    if not hasattr(config, 'features'):
        validation_results['valid'] = False
        validation_results['missing_sections'].append('features')
        return validation_results
    
    # Check for individual feature sections
    expected_sections = ['adstock', 'saturation', 'seasonality', 'baseline']
    optional_sections = ['attribution', 'creative_fatigue', 'competitors', 'custom_terms']
    
    for section in expected_sections:
        if not hasattr(config.features, section):
            validation_results['missing_sections'].append(f'features.{section}')
            validation_results['warnings'].append(f"Missing required section: features.{section}")
    
    for section in optional_sections:
        if not hasattr(config.features, section):
            validation_results['recommendations'].append(f"Consider adding optional section: features.{section}")
    
    if validation_results['missing_sections']:
        validation_results['valid'] = False
    
    logger.info(f"Feature engineering config validation: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
    
    return validation_results


def run_feature_engineering():
    """
    CLI entry point for feature engineering.
    
    Loads transformed data, applies feature engineering, and saves results.
    """
    # Load configuration first
    config = load_config()
    
    # Setup logging with full config
    setup_logging(
        level=config.logging.level,
        format=config.logging.format,
        log_file=Path(config.paths.artifacts) / "logs" / "features.log" if config.logging.file_rotation else None,
        mask_keys=config.logging.mask_keys
    )
    
    logger = get_logger(__name__)
    
    try:
        
        # Validate configuration
        validation = validate_feature_engineering_config(config)
        if not validation['valid']:
            logger.error("Configuration validation failed!")
            for missing in validation['missing_sections']:
                logger.error(f"Missing section: {missing}")
            return False
        
        # Load transformed data
        input_path = Path(config.paths.interim) / "transformed_data.parquet"
        if not input_path.exists():
            logger.error(f"Transformed data not found: {input_path}")
            logger.error("Please run data transformation first: mmm transform")
            return False
        
        logger.info(f"Loading transformed data from: {input_path}")
        data_df = pd.read_parquet(input_path)
        
        # Apply feature engineering
        output_path = Path(config.paths.features) / "engineered_features.parquet"
        result_df = engineer_features(data_df, config, str(output_path))
        
        # Log success
        logger.info("🎉 Feature engineering completed successfully!")
        logger.info(f"Engineered features saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        return False


if __name__ == "__main__":
    run_feature_engineering()
