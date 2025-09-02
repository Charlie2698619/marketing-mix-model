"""
Features package for MMM project.

This package contains all feature engineering modules for the
Hierarchical Bayesian MMM system.
"""

from .adstock import apply_adstock
from .seasonality import apply_seasonality_from_config  
from .saturation import apply_saturation, apply_saturation_from_config
from .baseline import generate_baseline_features_robust, apply_baseline_transformation_robust
from .creative_fatigue import apply_creative_fatigue, apply_creative_fatigue_from_config
from .attribution import apply_attribution_modeling
from .custom_terms import apply_custom_business_terms
from .competitors import apply_competitor_factors
from .engineer import engineer_features, run_feature_engineering

__all__ = [
    'apply_adstock',
    'apply_seasonality_from_config',
    'apply_saturation',
    'apply_saturation_from_config',
    'generate_baseline_features_robust',
    'apply_baseline_transformation_robust',
    'apply_creative_fatigue',
    'apply_creative_fatigue_from_config',
    'apply_attribution_modeling',
    'apply_custom_business_terms',
    'apply_competitor_factors',
    'engineer_features',
    'run_feature_engineering'
]
