"""
Universal metadata builder for both Meridian and PyMC models.

This module provides a unified interface for preparing model metadata that can be
consumed by either backend while allowing fine-tuning through YAML configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import re
from pathlib import Path


class UniversalMetadataBuilder:
    """Build metadata compatible with both Meridian and PyMC models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the metadata builder.
        
        Args:
            config: Full pipeline configuration with model settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Handle both Pydantic Config objects and regular dictionaries
        if hasattr(config, 'model'):
            # Pydantic Config object
            self.model_config = config.model.__dict__ if hasattr(config.model, '__dict__') else {}
            self.data_config = config.data.__dict__ if hasattr(config.data, '__dict__') else {}
            self.backend = getattr(config.model, 'backend', 'meridian')
        else:
            # Regular dictionary
            self.model_config = config.get('model', {})
            self.data_config = config.get('data', {})
            self.backend = self.model_config.get('backend', 'meridian')
        
        # Feature classification patterns from config
        self.feature_patterns = self.model_config.get('feature_classification', {})
        
    def build_metadata(self, 
                      X: pd.DataFrame, 
                      y: pd.Series, 
                      run_id: str,
                      dates: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Build universal metadata for model training.
        
        Args:
            X: Feature matrix
            y: Target variable
            run_id: Unique run identifier
            dates: Optional date series aligned with X and y
            
        Returns:
            Metadata dictionary compatible with both backends
        """
        self.logger.info("Building universal metadata for model training")
        
        # Base metadata
        meta = {
            'run_id': run_id,
            'backend': self.backend,
            'n_observations': len(X),
            'n_samples': len(X),  # Alias for compatibility
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'target_name': y.name or 'target',
            'target_stats': self._get_target_stats(y),
            'data_shape': X.shape,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add dates if provided
        if dates is not None:
            meta['dates'] = dates
            meta['date_range'] = {
                'start': dates.min().isoformat(),
                'end': dates.max().isoformat(),
                'periods': len(dates),
                'frequency': self._infer_frequency(dates)
            }
        
        # Feature classification
        feature_classification = self._classify_features(X.columns)
        
        # Filter zero-variance features to prevent convergence issues
        feature_classification = self._filter_zero_variance_features(X, feature_classification)
        
        meta.update(feature_classification)
        
        # Backend-specific metadata
        if self.backend == 'meridian':
            meta.update(self._build_meridian_metadata(X, y, feature_classification))
        elif self.backend == 'pymc':
            meta.update(self._build_pymc_metadata(X, y, feature_classification))
        
        # Model configuration from YAML
        meta['model_config'] = self._extract_model_config()
        
        self.logger.info(f"Built metadata with {len(meta)} keys for {self.backend} backend")
        return meta
    
    def _get_target_stats(self, y: pd.Series) -> Dict[str, Any]:
        """Get target variable statistics."""
        return {
            'mean': float(y.mean()),
            'std': float(y.std()),
            'min': float(y.min()),
            'max': float(y.max()),
            'median': float(y.median()),
            'q25': float(y.quantile(0.25)),
            'q75': float(y.quantile(0.75)),
            'zeros': int((y == 0).sum()),
            'nulls': int(y.isnull().sum()),
            'positive_pct': float((y > 0).mean() * 100)
        }
    
    def _infer_frequency(self, dates: pd.Series) -> str:
        """Infer the frequency of the date series."""
        try:
            if len(dates) < 2:
                return 'unknown'
            
            diffs = pd.Series(dates).diff().dropna()
            median_diff = diffs.median()
            
            if median_diff == pd.Timedelta(days=1):
                return 'daily'
            elif median_diff == pd.Timedelta(days=7):
                return 'weekly'
            elif 28 <= median_diff.days <= 31:
                return 'monthly'
            else:
                return f'{median_diff.days}D'
        except Exception:
            return 'unknown'
    
    def _classify_features(self, feature_names: List[str]) -> Dict[str, Any]:
        """Classify features into media, control, and other categories."""
        
        # Get patterns from config or use defaults
        patterns = self.feature_patterns.get('patterns', {
            'media_spend': [r'.*_spend$', r'.*_investment$', r'.*_cost$'],
            'media_clicks': [r'.*_clicks$', r'.*_sessions$'],
            'media_impressions': [r'.*_impressions$', r'.*_views$'],
            'adstock': [r'adstock_.*', r'.*_adstocked$'],
            'saturation': [r'.*_saturated$', r'saturated_.*'],
            'seasonality': [r'sin_.*', r'cos_.*', r'trend', r'.*_seasonal$'],
            'baseline': [r'external_.*', r'baseline_.*', r'.*_baseline$'],
            'holidays': [r'.*_day$', r'.*_before$', r'.*_after$'],
            'creative': [r'creative_.*', r'.*_creative$'],
            'competitors': [r'competitor_.*', r'.*_competitor$'],
            'adjustments': [r'.*_ADJUSTMENTS$', r'.*_RETURNS$', r'.*_adjustments$', r'.*_returns$']
        })
        
        classification = {
            'media_features': [],
            'media_spend_features': [],
            'media_clicks_features': [],
            'media_impressions_features': [],
            'adstock_features': [],
            'saturation_features': [],
            'seasonality_features': [],
            'baseline_features': [],
            'holiday_features': [],
            'creative_features': [],
            'competitor_features': [],
            'adjustment_features': [],  # Track adjustment features separately
            'control_features': [],
            'unknown_features': []
        }
        
        # Classify each feature
        for feature in feature_names:
            classified = False
            
            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    if re.match(pattern, feature, re.IGNORECASE):
                        if category == 'media_spend':
                            classification['media_spend_features'].append(feature)
                            classification['media_features'].append(feature)
                        elif category == 'media_clicks':
                            classification['media_clicks_features'].append(feature)
                            classification['media_features'].append(feature)
                        elif category == 'media_impressions':
                            classification['media_impressions_features'].append(feature)
                            classification['media_features'].append(feature)
                        elif category == 'adjustments':
                            classification['adjustment_features'].append(feature)
                        elif category in classification:
                            classification[f'{category}_features'].append(feature)
                        classified = True
                        break
                if classified:
                    break
            
            # If not classified as media or specific category, it's a control
            if not classified:
                if feature not in classification['media_features'] and feature not in classification['adjustment_features']:
                    classification['control_features'].append(feature)
                    classification['unknown_features'].append(feature)
        
        # Additional aggregations
        classification['all_media_features'] = list(set(
            classification['media_spend_features'] +
            classification['media_clicks_features'] +
            classification['media_impressions_features']
        ))
        
        classification['all_control_features'] = list(set(
            classification['seasonality_features'] +
            classification['baseline_features'] +
            classification['holiday_features'] +
            classification['control_features']
        ))
        
        # Channel mapping from config
        channel_map = self.data_config.get('channel_map', {})
        classification['channels'] = list(channel_map.keys())
        classification['channel_spend_mapping'] = {
            channel: [f for f in classification['media_spend_features'] 
                     if any(chan_name in f.lower() for chan_name in [channel, channel.replace('_', '')])]
            for channel in classification['channels']
        }
        
        # Log classification summary
        self.logger.info(f"Feature classification: "
                        f"media={len(classification['all_media_features'])}, "
                        f"control={len(classification['all_control_features'])}, "
                        f"adjustments={len(classification['adjustment_features'])}, "
                        f"unknown={len(classification['unknown_features'])}")
        
        return classification
    
    def _filter_zero_variance_features(self, X: pd.DataFrame, feature_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out zero-variance features that can cause convergence issues.
        
        Args:
            X: Feature matrix
            feature_classification: Initial feature classification
            
        Returns:
            Updated feature classification with zero-variance features removed
        """
        # Calculate variance for all features
        feature_variances = X.var()
        zero_variance_threshold = 1e-10  # Very small threshold for numerical stability
        
        zero_variance_features = []
        for feature in feature_variances.index:
            if feature_variances[feature] < zero_variance_threshold:
                zero_variance_features.append(feature)
        
        if zero_variance_features:
            self.logger.warning(f"Found {len(zero_variance_features)} zero-variance features that will be excluded: {zero_variance_features}")
            
            # Remove zero-variance features from all categories
            updated_classification = {}
            for category, features in feature_classification.items():
                if isinstance(features, list):
                    updated_classification[category] = [f for f in features if f not in zero_variance_features]
                else:
                    updated_classification[category] = features
            
            # Track which features were removed
            updated_classification['zero_variance_features'] = zero_variance_features
            updated_classification['variance_filtered'] = True
            
            return updated_classification
        else:
            feature_classification['zero_variance_features'] = []
            feature_classification['variance_filtered'] = False
            return feature_classification
    
    def _build_meridian_metadata(self, X: pd.DataFrame, y: pd.Series, 
                                feature_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Build Meridian-specific metadata."""
        meridian_meta = {}
        
        # Meridian-specific channel configuration
        meridian_config = self.model_config.get('backend_params', {}).get(self.backend, {})
        
        # Channel configuration for Meridian
        if 'channels' in meridian_config:
            meridian_meta['meridian_channels'] = meridian_config['channels']
        else:
            # Auto-detect from features
            meridian_meta['meridian_channels'] = feature_classification.get('channels', [])
        
        # Media data requirements for Meridian
        meridian_meta['meridian_media_data'] = {
            'spend_columns': feature_classification.get('media_spend_features', []),
            'clicks_columns': feature_classification.get('media_clicks_features', []),
            'impressions_columns': feature_classification.get('media_impressions_features', []),
            'has_media': len(feature_classification.get('all_media_features', [])) > 0,
            'has_reach_frequency': len(feature_classification.get('media_impressions_features', [])) > 0
        }
        
        # Control data for Meridian
        meridian_meta['meridian_control_data'] = {
            'control_columns': feature_classification.get('all_control_features', []),
            'has_controls': len(feature_classification.get('all_control_features', [])) > 0
        }
        
        # Geo and time configuration
        meridian_meta['meridian_geo_time'] = {
            'geo_column': 'geo',  # Default geo column name
            'time_column': 'time',  # Default time column name
            'kpi_type': 'revenue',  # Default KPI type
            'default_geo': self.data_config.get('regions', ['US'])[0] if self.data_config.get('regions') else 'US'
        }
        
        return meridian_meta
    
    def _build_pymc_metadata(self, X: pd.DataFrame, y: pd.Series,
                            feature_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Build PyMC-specific metadata."""
        pymc_meta = {}
        
        # PyMC-specific configuration
        pymc_config = self.model_config.get('backend_params', {}).get(self.backend, {})
        
        # Feature groupings for PyMC hierarchical modeling
        pymc_meta['pymc_feature_groups'] = {
            'media_spend': feature_classification.get('media_spend_features', []),
            'media_clicks': feature_classification.get('media_clicks_features', []),
            'seasonality': feature_classification.get('seasonality_features', []),
            'baseline': feature_classification.get('baseline_features', []),
            'controls': feature_classification.get('all_control_features', [])
        }
        
        # Prior configuration for PyMC
        priors_config = self.model_config.get('priors', {})
        pymc_meta['pymc_priors'] = {
            'roi_bounds': priors_config.get('roi_bounds', {}),
            'adstock_decay': priors_config.get('adstock_decay', [0.3, 0.95]),
            'saturation_params': priors_config.get('saturation_params', [0.1, 2.0]),
            'baseline_priors': priors_config.get('baseline_priors', [-2.0, 2.0])
        }
        
        # Hierarchical structure for PyMC
        hierarchy_config = self.model_config.get('hierarchy', {})
        pymc_meta['pymc_hierarchy'] = {
            'levels': hierarchy_config.get('levels', []),
            'sharing_strategy': hierarchy_config.get('sharing_strategy', 'partial_pooling')
        }
        
        return pymc_meta
    
    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract model configuration from YAML for backend-specific use."""
        
        # Handle both Pydantic and dictionary configs
        if hasattr(self.config, 'model'):
            # Pydantic Config object
            backend_params = getattr(self.config.model, 'backend_params', {})
            priors = getattr(self.config.model, 'priors', {})
            constraints = getattr(self.config.model, 'constraints', {})
            hierarchy = getattr(self.config.model, 'hierarchy', {})
            
            # Data config
            channel_map = getattr(self.config.data, 'channel_map', {})
            outcome = getattr(self.config.data, 'outcome', '')
            date_col = getattr(self.config.data, 'date_col', '')
            revenue_col = getattr(self.config.data, 'revenue_col', '')
        else:
            # Regular dictionary
            backend_params = self.model_config.get('backend_params', {})
            priors = self.model_config.get('priors', {})
            constraints = self.model_config.get('constraints', {})
            hierarchy = self.model_config.get('hierarchy', {})
            
            # Data config
            channel_map = self.data_config.get('channel_map', {})
            outcome = self.data_config.get('outcome', '')
            date_col = self.data_config.get('date_col', '')
            revenue_col = self.data_config.get('revenue_col', '')
        
        return {
            'backend': self.backend,
            'backend_params': backend_params,
            'priors': priors,
            'constraints': constraints,
            'hierarchy': hierarchy,
            'data_config': {
                'channel_map': channel_map,
                'outcome': outcome,
                'date_col': date_col,
                'revenue_col': revenue_col
            }
        }
    
    def validate_metadata(self, meta: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate metadata for completeness and consistency.
        
        Args:
            meta: Metadata dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = [
            'n_observations', 'n_features', 'feature_names', 
            'target_name', 'backend'
        ]
        
        for field in required_fields:
            if field not in meta:
                errors.append(f"Missing required field: {field}")
        
        # Backend-specific validation
        if meta.get('backend') == 'meridian':
            errors.extend(self._validate_meridian_metadata(meta))
        elif meta.get('backend') == 'pymc':
            errors.extend(self._validate_pymc_metadata(meta))
        
        # Feature consistency
        if 'feature_names' in meta and 'n_features' in meta:
            if len(meta['feature_names']) != meta['n_features']:
                errors.append(f"Feature count mismatch: {len(meta['feature_names'])} names vs {meta['n_features']} count")
        
        return len(errors) == 0, errors
    
    def _validate_meridian_metadata(self, meta: Dict[str, Any]) -> List[str]:
        """Validate Meridian-specific metadata."""
        errors = []
        
        # Check for media or reach+frequency requirement
        meridian_media = meta.get('meridian_media_data', {})
        has_media = meridian_media.get('has_media', False)
        has_reach_freq = meridian_media.get('has_reach_frequency', False)
        
        if not (has_media or has_reach_freq):
            errors.append("Meridian requires at least one of media or reach+frequency data")
        
        return errors
    
    def _validate_pymc_metadata(self, meta: Dict[str, Any]) -> List[str]:
        """Validate PyMC-specific metadata."""
        errors = []
        
        # Check for minimum feature requirements
        pymc_groups = meta.get('pymc_feature_groups', {})
        total_features = sum(len(group) for group in pymc_groups.values())
        
        if total_features == 0:
            errors.append("PyMC requires at least some features to be classified")
        
        return errors


def create_universal_metadata(config: Dict[str, Any], 
                             X: pd.DataFrame, 
                             y: pd.Series, 
                             run_id: str,
                             dates: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Convenience function to create universal metadata.
    
    Args:
        config: Pipeline configuration
        X: Feature matrix
        y: Target variable
        run_id: Run identifier
        dates: Optional date series
        
    Returns:
        Universal metadata dictionary
    """
    builder = UniversalMetadataBuilder(config)
    return builder.build_metadata(X, y, run_id, dates)
