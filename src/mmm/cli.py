"""
Command-line interface for the MMM project.

Provides a Click-based CLI for running the complete MMM pipeline
or individual steps.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional, List
import click
import logging
# import mlflow  # TODO: Add MLflow back when available
from datetime import datetime

from .config import load_config, resolve_paths, validate_config
from .utils.logging import setup_logging, get_logger, get_step_logger


@click.group()
@click.option('--config', default="config/main.yaml", help='Path to configuration file')
@click.option('--profile', default=None, help='Configuration profile (local, docker, k8s)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, profile, verbose):
    """Hierarchical Bayesian Marketing Mix Modeling CLI."""
    # Ensure context exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        cfg = load_config(config, profile)
        resolve_paths(cfg)
        validate_config(cfg)
        ctx.obj['config'] = cfg
        ctx.obj['config_path'] = config
        ctx.obj['profile'] = profile
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)
    
    # Setup logging with full config integration
    log_level = "DEBUG" if verbose else cfg.logging.level
    log_file = None
    if getattr(cfg.logging, 'file_rotation', False):
        # Create log file path based on timestamp
        log_dir = Path(getattr(cfg.paths, 'artifacts', 'artifacts')) / 'logs'
        log_file = str(log_dir / f"mmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    setup_logging(
        level=log_level, 
        format=cfg.logging.format,
        log_file=log_file,
        mask_keys=getattr(cfg.logging, 'mask_keys', ['api_key', 'secret'])
    )
    
    logging.info(f"MMM CLI started with profile: {profile or 'default'}")


@cli.command()
@click.option('--steps', default=None, help='Comma-separated list of steps to run')
@click.option('--slice', 'data_slice', default=None, help='Data slice hint (e.g., last_12_weeks)')
@click.option('--backend', default=None, help='Override model backend (meridian, pymc)')
@click.pass_context
def run(ctx, steps, data_slice, backend):
    """Run the complete MMM pipeline or specific steps."""
    cfg = ctx.obj['config']
    
    # Override backend if specified
    if backend:
        cfg.model.backend = backend
    
    # Determine steps to run
    if steps:
        step_list = [s.strip() for s in steps.split(',')]
    else:
        step_list = cfg.orchestration.get('default_steps', [
            'ingest', 'validate', 'transform', 'features', 'train', 'optimize', 'evaluate', 'report'
        ])
    
    # Generate run ID
    run_id = f"mmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup MLflow (TODO: Re-enable when MLflow is available)
    # mlflow.set_experiment(cfg.tracking.mlflow.get('experiment_name', 'mmm_experiment'))
    
    # with mlflow.start_run(run_name=run_id):
    if True:  # Temporary replacement for MLflow context
        # Log configuration (TODO: Re-enable MLflow logging)
        # mlflow.log_params({
        #     'backend': cfg.model.backend,
        #     'data_slice': data_slice,
        #     'steps': ','.join(step_list),
        #     'profile': ctx.obj.get('profile', 'default')
        # })
        
        start_time = time.time()
        
        try:
            for step in step_list:
                click.echo(f"Running step: {step}")
                step_start = time.time()
                
                if step == 'ingest':
                    run_ingest(cfg, data_slice)
                elif step == 'validate':
                    run_validate(cfg)
                elif step == 'transform':
                    run_transform(cfg)
                elif step == 'features':
                    run_features(cfg)
                elif step == 'train':
                    run_train(cfg, run_id)
                elif step == 'optimize':
                    run_optimize(cfg, run_id)
                elif step == 'evaluate':
                    run_evaluate(cfg, run_id)
                elif step == 'report':
                    run_report(cfg, run_id)
                else:
                    click.echo(f"Unknown step: {step}", err=True)
                    sys.exit(1)
                
                step_duration = time.time() - step_start
                # mlflow.log_metric(f"{step}_duration_seconds", step_duration)
                click.echo(f"Completed {step} in {step_duration:.2f} seconds")
        
        except Exception as e:
            logging.error(f"Pipeline failed at step {step}: {e}")
            # mlflow.log_param('status', 'failed')
            # mlflow.log_param('error', str(e))
            raise
        
        total_duration = time.time() - start_time
        # mlflow.log_metric('total_duration_seconds', total_duration)
        # mlflow.log_param('status', 'completed')
        
        click.echo(f"Pipeline completed successfully in {total_duration:.2f} seconds")
        click.echo(f"Run ID: {run_id}")


@cli.command()
@click.option('--slice', 'data_slice', default=None, help='Data slice hint')
@click.pass_context
def ingest(ctx, data_slice):
    """Run data ingestion step."""
    cfg = ctx.obj['config']
    run_ingest(cfg, data_slice)


@cli.command()
@click.pass_context
def validate(ctx):
    """Run data validation step."""
    cfg = ctx.obj['config']
    run_validate(cfg)


@cli.command()
@click.pass_context
def transform(ctx):
    """Run data transformation step."""
    cfg = ctx.obj['config']
    run_transform(cfg)


@cli.command()
@click.pass_context
def features(ctx):
    """Run feature engineering step."""
    cfg = ctx.obj['config']
    run_features(cfg)


@cli.command()
@click.option('--backend', default=None, help='Model backend (meridian, pymc)')
@click.pass_context
def train(ctx, backend):
    """Run model training step."""
    cfg = ctx.obj['config']
    if backend:
        cfg.model.backend = backend
    
    run_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_train(cfg, run_id)


@cli.command()
@click.pass_context
def optimize(ctx):
    """Run budget optimization step."""
    cfg = ctx.obj['config']
    run_id = f"optimize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_optimize(cfg, run_id)


@cli.command()
@click.pass_context
def evaluate(ctx):
    """Run model evaluation step."""
    cfg = ctx.obj['config']
    run_id = f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_evaluate(cfg, run_id)


@cli.command()
@click.pass_context
def report(ctx):
    """Generate reports and dashboards."""
    cfg = ctx.obj['config']
    run_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_report(cfg, run_id)


# Step implementations
def run_ingest(cfg, data_slice: Optional[str] = None):
    """Run data ingestion."""
    from .data.ingest import ingest_figshare
    
    click.echo("Starting data ingestion...")
    df = ingest_figshare(cfg, data_slice)
    click.echo(f"Ingested {len(df)} rows")


def run_validate(cfg):
    """Run data validation."""
    from .data.validate import validate_data
    
    click.echo("Starting data validation...")
    validate_data(cfg)
    click.echo("Data validation completed")


def run_transform(cfg):
    """Run data transformation."""
    from .data.transform import transform_data
    
    click.echo("Starting data transformation...")
    transform_data(cfg)
    click.echo("Data transformation completed")


def run_features(cfg):
    """Run feature engineering."""
    from .features.engineer import run_feature_engineering
    
    click.echo("Starting feature engineering...")
    success = run_feature_engineering()
    
    if success:
        click.echo("✅ Feature engineering completed successfully")
    else:
        click.echo("❌ Feature engineering failed")
        raise click.ClickException("Feature engineering failed")


def run_train(cfg, run_id: str):
    """Run model training."""
    import time
    import pandas as pd
    from pathlib import Path
    from .models.metadata import create_universal_metadata
    
    start_time = time.time()
    step_logger = get_step_logger("train_model")
    
    try:
        step_logger.info(f"Starting model training with backend: {cfg.model.backend}")
        click.echo(f"Starting model training with backend: {cfg.model.backend}")
        
        # Load feature set (output from feature engineering)
        feature_path = Path(cfg.paths.features) / "feature_set.parquet"
        if feature_path.exists():
            data = pd.read_parquet(feature_path)
            step_logger.info(f"Loaded feature set with {len(data)} rows, {len(data.columns)} columns")
        else:
            # Fallback to transformed data
            data_path = Path(cfg.paths.interim) / "transformed_data.parquet"
            if not data_path.exists():
                click.echo("Error: No feature set or transformed data found. Run 'mmm features' first.")
                return
            data = pd.read_parquet(data_path)
            step_logger.info(f"Loaded transformed data with {len(data)} rows for training")
        
        # Initialize model based on backend
        if cfg.model.backend == "meridian":
            from .models.meridian import MeridianModel
            # Pass the full config, not just meridian config
            model = MeridianModel(cfg)
        elif cfg.model.backend == "pymc":
            from .models.pymc import PyMCModel
            # Pass the full config, not just pymc config
            model = PyMCModel(cfg)
        else:
            raise ValueError(f"Unknown model backend: {cfg.model.backend}")
        
        # Prepare target variable from config
        outcome_col = getattr(cfg.data, 'outcome', 'ALL_PURCHASES_ORIGINAL_PRICE')
        revenue_col = getattr(cfg.data, 'revenue_col', 'ALL_PURCHASES_ORIGINAL_PRICE')
        
        # Try multiple target column possibilities
        target_candidates = [
            outcome_col,
            revenue_col,
            'first_purchases_original_price',
            'all_purchases_original_price',
            'FIRST_PURCHASES_ORIGINAL_PRICE',
            'ALL_PURCHASES_ORIGINAL_PRICE'
        ]
        
        target_col = None
        for candidate in target_candidates:
            if candidate in data.columns:
                target_col = candidate
                break
        
        if target_col is None:
            # Fallback to any revenue-like column
            revenue_cols = [col for col in data.columns if 
                           any(keyword in col.upper() for keyword in ['PRICE', 'REVENUE', 'SALES'])]
            if revenue_cols:
                target_col = revenue_cols[0]
            else:
                raise ValueError("No revenue/target column found in data")
        
        step_logger.info(f"Using target column: {target_col}")
        
        # Prepare date column for metadata
        date_candidates = [
            getattr(cfg.data, 'date_col', 'DATE_DAY'),
            'date_day',
            'date',
            'DATE_DAY',
            'DATE'
        ]
        
        date_col = None
        for candidate in date_candidates:
            if candidate in data.columns:
                date_col = candidate
                break
        
        # Prepare feature matrix (exclude target, date, and metadata columns)
        exclude_patterns = [
            target_col.upper(),
            'DATE', 'ORGANISATION', 'TERRITORY', 'CURRENCY', 'MMM_TIMESERIES',
            'BRAND', 'REGION', 'ID'
        ]
        
        feature_cols = [col for col in data.columns 
                       if not any(pattern in col.upper() for pattern in exclude_patterns)]
        
        step_logger.info(f"Selected {len(feature_cols)} features for training")
        
        X = data[feature_cols].fillna(0)  # Fill NaN with 0 for media data
        y = data[target_col].fillna(data[target_col].median())  # Fill NaN with median
        
        # Prepare dates if available
        dates = None
        if date_col and date_col in data.columns:
            dates = pd.to_datetime(data[date_col])
            step_logger.info(f"Using date column: {date_col}")
        
        # Build universal metadata using the new system
        meta = create_universal_metadata(
            config=cfg,
            X=X,
            y=y,
            run_id=run_id,
            dates=dates
        )
        
        step_logger.info(f"Built metadata with {len(meta)} keys")
        step_logger.info(f"Feature classification: "
                        f"media={len(meta.get('all_media_features', []))}, "
                        f"control={len(meta.get('all_control_features', []))}")
        
        # Validate metadata
        from .models.metadata import UniversalMetadataBuilder
        builder = UniversalMetadataBuilder(cfg)
        is_valid, errors = builder.validate_metadata(meta)
        
        if not is_valid:
            step_logger.error(f"Metadata validation failed: {errors}")
            for error in errors:
                click.echo(f"Error: {error}")
            return
        
        step_logger.info("Metadata validation passed")
        
        # Train the model with universal metadata
        step_logger.info(f"Training with {len(feature_cols)} features on {len(data)} samples")
        step_logger.info(f"Target column: {target_col}")
        step_logger.info(f"Features: {feature_cols[:10]}...")  # Log first 10 features
        
        # Fit the model
        fit_result = model.fit(X, y, meta)
        
        # Create output directory
        output_dir = Path(cfg.paths.models)
        output_dir.mkdir(exist_ok=True)
        
        # Save fit results
        import pickle
        results_path = output_dir / f"{cfg.model.backend}_fit_result.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(fit_result, f)
        
        step_logger.info(f"Saved fit results to {results_path}")
        
        # Save model artifacts
        model.save_artifacts(str(output_dir))
        
        # Generate predictions and contributions
        try:
            predictions = model.predict(X)
            contributions = model.get_media_contribution()
            
            # Save predictions
            pred_df = pd.DataFrame({
                'actual': y,
                'predicted': predictions,
                'residual': y - predictions
            })
            
            # Add basic metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np
            mae = mean_absolute_error(y, predictions)
            rmse = np.sqrt(mean_squared_error(y, predictions))
            mape = np.mean(np.abs((y - predictions) / y)) * 100
            
            pred_df.to_parquet(output_dir / "predictions.parquet")
            step_logger.info(f"Model performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            # Save contributions
            contributions.to_parquet(output_dir / "media_contributions.parquet")
            step_logger.info(f"Saved {len(contributions)} media contribution records")
            
        except Exception as e:
            step_logger.warning(f"Error generating predictions/contributions: {e}")
        
        duration = time.time() - start_time
        step_logger.info(f"Model training completed in {duration:.2f} seconds")
        step_logger.info(f"Diagnostics: {fit_result.diagnostics}")
        
        click.echo(f"✅ Model training completed successfully!")
        click.echo(f"Backend: {cfg.model.backend}")
        
        # Display convergence status with details
        converged = fit_result.diagnostics.get('converged', None)
        if converged is True:
            convergence_status = "✅ Converged"
        elif converged is False:
            convergence_status = "❌ Not Converged"
        else:
            convergence_status = "❓ Unknown"
        
        click.echo(f"Convergence: {convergence_status}")
        
        # Show detailed diagnostics if available
        max_rhat = fit_result.diagnostics.get('max_rhat')
        min_ess = fit_result.diagnostics.get('min_ess_bulk')
        if max_rhat is not None or min_ess is not None:
            details = []
            if max_rhat is not None:
                details.append(f"R-hat: {max_rhat:.3f}")
            if min_ess is not None:
                details.append(f"ESS: {min_ess:.0f}")
            click.echo(f"Diagnostics: {', '.join(details)}")
        
        click.echo(f"Fit time: {fit_result.fit_time:.2f} seconds")
        click.echo(f"Run ID: {run_id}")
        
    except Exception as e:
        step_logger.error(f"Training failed: {e}")
        click.echo(f"❌ Error: {e}")
        raise


def run_optimize(cfg, run_id: str):
    """Run budget optimization."""
    from .optimization.allocator import run_budget_optimization
    
    click.echo("Starting budget optimization...")
    success = run_budget_optimization()
    
    if success:
        click.echo("✅ Budget optimization completed successfully")
    else:
        click.echo("❌ Budget optimization failed")
        raise click.ClickException("Budget optimization failed")


def run_evaluate(cfg, run_id: str):
    """Run model evaluation."""
    from .evaluation.validator import run_model_evaluation
    
    click.echo("Starting model evaluation...")
    success = run_model_evaluation()
    
    if success:
        click.echo("✅ Model evaluation completed successfully")
    else:
        click.echo("❌ Model evaluation failed")
        raise click.ClickException("Model evaluation failed")


def run_report(cfg, run_id: str):
    """Generate reports."""
    from .reporting.dashboard import run_report_generation
    
    click.echo("Starting report generation...")
    success = run_report_generation()
    
    if success:
        click.echo("✅ Report generation completed successfully")
    else:
        click.echo("❌ Report generation failed")
        raise click.ClickException("Report generation failed")


@cli.command()
@click.pass_context
def config_check(ctx):
    """Validate configuration for digital metrics and adstock settings."""
    cfg = ctx.obj['config']
    
    click.echo("🔍 Checking enhanced MMM configuration...")
    
    # Check digital metrics configuration
    digital_config = cfg.data.digital_specific or {}
    platform_metrics = digital_config.get('platform_metrics', {})
    
    # Validate digital metrics structure
    expected_metric_types = ['clicks', 'impressions', 'organic_traffic']
    for metric_type in expected_metric_types:
        if metric_type in platform_metrics:
            count = len(platform_metrics[metric_type])
            click.echo(f"✅ {metric_type}: {count} metrics configured")
        else:
            click.echo(f"⚠️  {metric_type}: Not configured")
    
    # Check adstock configuration
    adstock_config = cfg.features.adstock or {}
    platform_overrides = adstock_config.get('platform_overrides', {})
    
    click.echo(f"\n🔧 Adstock transformations:")
    click.echo(f"  Default type: {adstock_config.get('default_type', 'geometric')}")
    click.echo(f"  Platform-specific configs: {len(platform_overrides)}")
    
    for platform, params in platform_overrides.items():
        adstock_type = params.get('type', 'geometric')
        if adstock_type == 'geometric':
            decay = params.get('decay', params.get('lambda', 'N/A'))
            click.echo(f"    {platform}: {adstock_type} (decay={decay})")
        elif adstock_type == 'weibull':
            shape = params.get('shape', 'N/A')
            scale = params.get('scale', 'N/A')
            click.echo(f"    {platform}: {adstock_type} (shape={shape}, scale={scale})")
    
    # Check channel mapping
    channel_map = cfg.data.channel_map or {}
    click.echo(f"\n📊 Channel mapping: {len(channel_map)} channels configured")
    
    # Validate consistency between channel_map and adstock platforms
    mapped_channels = set(channel_map.keys())
    adstock_platforms = set(platform_overrides.keys())
    
    missing_adstock = mapped_channels - adstock_platforms
    extra_adstock = adstock_platforms - mapped_channels
    
    if missing_adstock:
        click.echo(f"⚠️  Channels missing adstock config: {missing_adstock}")
    if extra_adstock:
        click.echo(f"⚠️  Adstock configs for unmapped channels: {extra_adstock}")
    
    if not missing_adstock and not extra_adstock:
        click.echo("✅ Channel mapping and adstock configuration are consistent")
    
    click.echo("\n✨ Configuration check completed!")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
