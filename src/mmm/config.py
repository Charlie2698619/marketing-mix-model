"""
Configuration management for MMM project.

Handles YAML loading, Pydantic validation, and profile overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
from pydantic import BaseModel, Field, validator
import logging


class PathsConfig(BaseModel):
    """Configuration for data and output paths."""
    raw: str = "data/raw"
    interim: str = "data/interim"
    models: str = "data/models"
    features: str = "data/features"
    artifacts: str = "artifacts"
    reports: str = "reports"
    synthetic: str = "data/synthetic"


class DataConfig(BaseModel):
    """Configuration for data processing."""
    frequency: str = "daily"
    timezone: str = "UTC"
    outcome: str = "revenue"
    brands: List[str] = ["all"]
    regions: List[str] = ["US"]
    keys: List[str] = ["date", "brand", "region"]
    brand_key: str = "ORGANISATION_ID"  # Column name for brand identifier
    region_key: str = "TERRITORY_NAME"  # Column name for region identifier
    date_col: str = "DATE_DAY"
    revenue_col: str = "ALL_PURCHASES_ORIGINAL_PRICE"
    volume_col: str = "ALL_PURCHASES"
    channel_map: Dict[str, str] = Field(default_factory=dict)
    digital_specific: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('digital_specific')
    def validate_digital_specific(cls, v):
        """Validate digital_specific configuration structure."""
        if not isinstance(v, dict):
            return v
            
        # Check for required platform_metrics structure
        platform_metrics = v.get('platform_metrics', {})
        if platform_metrics:
            required_metric_types = ['clicks', 'impressions', 'organic_traffic']
            for metric_type in required_metric_types:
                if metric_type in platform_metrics:
                    if not isinstance(platform_metrics[metric_type], list):
                        raise ValueError(f"platform_metrics.{metric_type} must be a list")
        
        return v


class ValidationConfig(BaseModel):
    """Configuration for data validation."""
    schema_ref: str = "config/schema.json"
    min_weeks_per_market: int = 52
    missing_policy: str = "interpolate"
    outlier_policy: str = "winsorize"
    outlier_threshold: float = 0.95
    anomalies: Dict[str, bool] = Field(default_factory=dict)
    data_quality: Dict[str, float] = Field(default_factory=lambda: {"missing_data_threshold": 0.3})


class FeaturesConfig(BaseModel):
    """Configuration for feature engineering."""
    adstock: Dict[str, Any] = Field(default_factory=dict)
    saturation: Dict[str, Any] = Field(default_factory=dict)
    attribution: Dict[str, Any] = Field(default_factory=dict)
    creative_fatigue: Dict[str, Any] = Field(default_factory=dict)
    seasonality: Dict[str, Any] = Field(default_factory=dict)
    baseline: Dict[str, Any] = Field(default_factory=dict)
    custom_terms: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('adstock')
    def validate_adstock_config(cls, v):
        """Validate adstock configuration structure."""
        if not isinstance(v, dict):
            return v
            
        # Check platform_overrides structure
        platform_overrides = v.get('platform_overrides', {})
        if platform_overrides:
            for platform, params in platform_overrides.items():
                if not isinstance(params, dict):
                    raise ValueError(f"adstock.platform_overrides.{platform} must be a dictionary")
                
                # Validate required parameters based on type
                adstock_type = params.get('type', 'geometric')
                if adstock_type == 'geometric':
                    if 'decay' not in params and 'lambda' not in params:
                        raise ValueError(f"Geometric adstock for {platform} requires 'decay' or 'lambda' parameter")
                elif adstock_type == 'weibull':
                    if 'shape' not in params or 'scale' not in params:
                        raise ValueError(f"Weibull adstock for {platform} requires 'shape' and 'scale' parameters")
        
        return v


class ModelConfig(BaseModel):
    """Configuration for model training."""
    backend: str = "meridian"
    likelihood: str = "lognormal"
    link_function: str = "log"
    hierarchy: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, List[str]] = Field(default_factory=dict)
    priors: Dict[str, Any] = Field(default_factory=dict)
    backend_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    rolling_splits: Dict[str, int] = Field(default_factory=dict)
    seeds: List[int] = [42, 123, 456]
    runtime_guardrails: Dict[str, Union[int, float]] = Field(default_factory=dict)


class OptimizationConfig(BaseModel):
    """Configuration for budget optimization."""
    objective: str = "roas"
    platform_constraints: Dict[str, Any] = Field(default_factory=dict)
    business_logic: Dict[str, bool] = Field(default_factory=dict)
    scenario_presets: Dict[str, Union[float, Dict[str, Any]]] = Field(default_factory=dict)  # Support both old (float) and new (dict) formats
    uncertainty_propagation: bool = True
    uncertainty_samples: int = 1000
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-9
    algorithm: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('scenario_presets')
    def validate_scenario_presets(cls, v):
        """Validate scenario presets can be either old format (float) or new format (dict)."""
        if not isinstance(v, dict):
            return v
            
        for scenario_name, scenario_config in v.items():
            if isinstance(scenario_config, (int, float)):
                # Old format - just a budget multiplier
                continue
            elif isinstance(scenario_config, dict):
                # New format - validate structure
                if 'budget_multiplier' not in scenario_config:
                    raise ValueError(f"Scenario '{scenario_name}' missing required 'budget_multiplier'")
                
                budget_mult = scenario_config['budget_multiplier']
                if not isinstance(budget_mult, (int, float)) or budget_mult <= 0:
                    raise ValueError(f"Scenario '{scenario_name}' budget_multiplier must be positive number")
                    
                # Validate platform_adjustments if present
                platform_adjustments = scenario_config.get('platform_adjustments', {})
                if platform_adjustments and not isinstance(platform_adjustments, dict):
                    raise ValueError(f"Scenario '{scenario_name}' platform_adjustments must be a dictionary")
                    
                for platform, adjustment in platform_adjustments.items():
                    if not isinstance(adjustment, (int, float)) or adjustment <= 0:
                        raise ValueError(f"Platform adjustment for '{platform}' must be positive number")
            else:
                raise ValueError(f"Scenario '{scenario_name}' must be either a number or dictionary")
        
        return v


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    validation_strategies: Dict[str, bool] = Field(default_factory=dict)
    digital_checks: Dict[str, Any] = Field(default_factory=dict)  # Changed from bool to Any to support both bool and float
    metrics: Dict[str, float] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "json"
    file_rotation: bool = True
    mask_keys: List[str] = ["api_key", "secret"]


class TrackingConfig(BaseModel):
    """Configuration for MLflow tracking."""
    mlflow: Dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    """Configuration for runtime settings."""
    device: str = "auto"
    n_threads: int = 4
    memory_limit_gb: int = 16


class Config(BaseModel):
    """Main configuration class for MMM project."""
    project: Dict[str, str] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    
    # Additional configurations
    complexity: Dict[str, Any] = Field(default_factory=dict)
    enhanced_cleaning: Dict[str, Any] = Field(default_factory=dict)  # Data transformation/cleaning config
    external_data: Dict[str, Any] = Field(default_factory=dict)      # External data integration config
    profiles: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ingest: Dict[str, Any] = Field(default_factory=dict)
    privacy: Dict[str, bool] = Field(default_factory=dict)
    reports: Dict[str, bool] = Field(default_factory=dict)
    synthetic_data: Dict[str, Any] = Field(default_factory=dict)
    orchestration: Dict[str, Any] = Field(default_factory=dict)

    @validator('model')
    def validate_backend(cls, v):
        """Validate model backend selection."""
        if v.backend not in ["meridian", "pymc"]:
            raise ValueError("Backend must be 'meridian' or 'pymc'")
        return v


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file and return as dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning(f"YAML file not found: {file_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {file_path}: {e}")
        raise


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_config(config_path: str = "config/main.yaml", profile: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with optional profile overrides.
    
    Args:
        config_path: Path to main configuration file
        profile: Optional profile name for overrides (e.g., 'local', 'docker', 'k8s')
        
    Returns:
        Config: Validated configuration object
    """
    # Load main configuration
    main_config = load_yaml_file(config_path)
    
    # Apply profile overrides if specified
    if profile:
        profile_path = f"config/profiles/{profile}.yaml"
        if os.path.exists(profile_path):
            profile_config = load_yaml_file(profile_path)
            main_config = merge_configs(main_config, profile_config)
            logging.info(f"Applied profile overrides from {profile_path}")
        else:
            logging.warning(f"Profile file not found: {profile_path}")
    
    # Validate and return configuration
    return Config(**main_config)


def resolve_paths(config: Config) -> None:
    """Ensure all configured directories exist."""
    paths_to_create = [
        config.paths.raw,
        config.paths.interim,
        config.paths.features,
        config.paths.artifacts,
        config.paths.reports,
        config.paths.synthetic,
    ]
    
    for path in paths_to_create:
        Path(path).mkdir(parents=True, exist_ok=True)
        logging.debug(f"Ensured directory exists: {path}")


def validate_config(config: Config) -> None:
    """Perform additional validation on configuration values."""
    # Validate required channels exist in channel_map
    required_channels = ["google_search", "meta_facebook"]
    missing_channels = [ch for ch in required_channels if ch not in config.data.channel_map]
    if missing_channels:
        raise ValueError(f"Missing required channels in channel_map: {missing_channels}")
    
    # Validate prior bounds
    if hasattr(config.model, 'priors') and 'roi_bounds' in config.model.priors:
        for channel, bounds in config.model.priors['roi_bounds'].items():
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid ROI bounds for {channel}: {bounds}")
    
    logging.info("Configuration validation passed")


if __name__ == "__main__":
    # Example usage
    config = load_config("config/main.yaml", profile="local")
    resolve_paths(config)
    validate_config(config)
    print(f"Loaded configuration for project: {config.project.get('name', 'Unknown')}")
