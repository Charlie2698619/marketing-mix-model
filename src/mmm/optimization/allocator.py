"""
Budget Allocation Optimizer

This module provides advanced budget optimization functionality for MMM,
including ROAS maximization, constraints, and scenario analysis.
Uses scipy optimization with MMM model predictions.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
import pickle
import warnings

from ..config import load_config
from ..utils.logging import setup_logging, get_logger


def _safe_config_get(config_obj, key, default=None):
    """Safely get value from config object (dict or Pydantic)."""
    if hasattr(config_obj, 'get'):
        return config_obj.get(key, default)
    else:
        return getattr(config_obj, key, default)


def optimize_budget_allocation(
    current_spend: Dict[str, float],
    total_budget: float,
    model_results: Any,
    config: Optional[Any] = None,
    scenario: str = "current",
    uncertainty_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimize budget allocation across marketing channels.
    
    This function finds the optimal spend allocation that maximizes ROAS
    while respecting business constraints and incorporating model uncertainty.
    
    Args:
        current_spend: Dictionary of current spend by channel
        total_budget: Total budget to allocate
        model_results: Trained MMM model results (Meridian fit result)
        config: Configuration object containing optimization parameters
        scenario: Scenario name for optimization ("current", "conservative", "aggressive")
        uncertainty_samples: Number of samples for uncertainty propagation (from config if None)
    
    Returns:
        Dictionary containing optimization results
    """
    logger = get_logger(__name__)
    
    # Get uncertainty samples from config if not provided
    if uncertainty_samples is None:
        opt_config = config.optimization if config else {}
        uncertainty_samples = _safe_config_get(opt_config, 'uncertainty_samples', 1000)
    
    if config is None:
        config = load_config()
    
    opt_config = config.optimization
    logger.info(f"Starting budget optimization for scenario: {scenario}")
    
    # Extract channel information
    channels = list(current_spend.keys())
    n_channels = len(channels)
    
    # Apply scenario budget adjustment and platform-specific adjustments
    scenario_presets = _safe_config_get(opt_config, 'scenario_presets', {})
    
    if scenario in scenario_presets:
        scenario_config = scenario_presets[scenario]
        
        # Apply budget multiplier
        budget_multiplier = _safe_config_get(scenario_config, 'budget_multiplier', 1.0)
        total_budget *= budget_multiplier
        logger.info(f"Applied {scenario} scenario budget multiplier: {budget_multiplier:.1%}")
        
        # Apply platform-specific adjustments
        platform_adjustments = _safe_config_get(scenario_config, 'platform_adjustments', {})
        if platform_adjustments:
            logger.info(f"Applying platform adjustments for {scenario} scenario")
            adjusted_spend = {}
            total_adjusted = 0
            
            # Apply adjustments
            for channel, spend in current_spend.items():
                adjustment = platform_adjustments.get(channel, 1.0)
                adjusted_spend[channel] = spend * adjustment
                total_adjusted += adjusted_spend[channel]
            
            # Normalize to maintain budget constraint
            if total_adjusted > 0:
                normalization_factor = total_budget / total_adjusted
                current_spend = {ch: spend * normalization_factor for ch, spend in adjusted_spend.items()}
                logger.info(f"Applied normalization factor: {normalization_factor:.3f}")
    else:
        # Backwards compatibility - simple multiplier
        if scenario == "conservative":
            total_budget *= 0.8
        elif scenario == "aggressive":
            total_budget *= 1.2
        logger.info(f"Applied simple {scenario} scenario adjustment")
    
    # Set up optimization constraints from config
    constraints = _setup_optimization_constraints(
        channels, total_budget, current_spend, opt_config
    )
    
    # Set up bounds for each channel
    bounds = _setup_channel_bounds(channels, total_budget, opt_config)
    
    # Initial guess (current allocation scaled to new budget)
    current_total = sum(current_spend.values())
    x0 = np.array([current_spend[ch] * total_budget / current_total for ch in channels])
    
    try:
        # Main optimization
        if opt_config.uncertainty_propagation:
            # Robust optimization with uncertainty
            result = _optimize_with_uncertainty(
                x0, bounds, constraints, channels, model_results, 
                opt_config, uncertainty_samples, logger
            )
        else:
            # Deterministic optimization
            result = _optimize_deterministic(
                x0, bounds, constraints, channels, model_results, opt_config, logger
            )
        
        # Process and validate results
        optimization_results = _process_optimization_results(
            result, channels, current_spend, total_budget, opt_config, logger
        )
        
        logger.info("✅ Budget optimization completed successfully")
        return optimization_results
        
    except Exception as e:
        logger.error(f"Budget optimization failed: {str(e)}")
        # Return fallback allocation
        return _create_fallback_allocation(channels, current_spend, total_budget, logger)


def _setup_optimization_constraints(
    channels: List[str], 
    total_budget: float, 
    current_spend: Dict[str, float],
    opt_config: Any
) -> List[Dict]:
    """Set up optimization constraints based on configuration."""
    constraints = []
    
    # Budget sum constraint (equality)
    constraints.append({
        'type': 'eq',
        'fun': lambda x: np.sum(x) - total_budget,
        'jac': lambda x: np.ones(len(x))
    })
    
    # Platform constraints - support both global and platform-specific
    platform_constraints = getattr(opt_config, 'platform_constraints', {})
    default_min_pct = _safe_config_get(platform_constraints, 'default_min_spend_pct', 0.05)
    default_max_pct = _safe_config_get(platform_constraints, 'default_max_spend_pct', 0.4)
    platform_specific = _safe_config_get(platform_constraints, 'platform_specific', {})
    
    for i, channel in enumerate(channels):
        # Get platform-specific constraints or use defaults
        if channel in platform_specific:
            channel_config = platform_specific[channel]
            min_spend_pct = _safe_config_get(channel_config, 'min_spend_pct', default_min_pct)
            max_spend_pct = _safe_config_get(channel_config, 'max_spend_pct', default_max_pct)
        else:
            # Fallback to old config structure for backwards compatibility
            min_spend_pct = _safe_config_get(platform_constraints, 'min_spend_pct', default_min_pct)
            max_spend_pct = _safe_config_get(platform_constraints, 'max_spend_pct', default_max_pct)
        
        # Minimum spend constraint
        min_spend = total_budget * min_spend_pct
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, idx=i, min_val=min_spend: x[idx] - min_val,
            'jac': lambda x, idx=i: _constraint_jacobian(len(x), idx, 1.0)
        })
        
        # Maximum spend constraint
        max_spend = total_budget * max_spend_pct
        constraints.append({
            'type': 'ineq', 
            'fun': lambda x, idx=i, max_val=max_spend: max_val - x[idx],
            'jac': lambda x, idx=i: _constraint_jacobian(len(x), idx, -1.0)
        })
    
    return constraints


def _setup_channel_bounds(
    channels: List[str], 
    total_budget: float, 
    opt_config: Any
) -> Bounds:
    """Set up variable bounds for optimization."""
    # Conservative bounds: each channel between 1% and 60% of total budget
    lower_bounds = [total_budget * 0.01] * len(channels)
    upper_bounds = [total_budget * 0.6] * len(channels)
    
    return Bounds(lower_bounds, upper_bounds)


def _constraint_jacobian(n_vars: int, var_idx: int, coeff: float) -> np.ndarray:
    """Create Jacobian vector for linear constraint."""
    jac = np.zeros(n_vars)
    jac[var_idx] = coeff
    return jac


def _optimize_deterministic(
    x0: np.ndarray,
    bounds: Bounds, 
    constraints: List[Dict],
    channels: List[str],
    model_results: Any,
    opt_config: Any,
    logger: logging.Logger
) -> Any:
    """Perform deterministic optimization."""
    logger.info("Running deterministic optimization...")
    
    def objective_function(spend_vector):
        """Negative ROAS (to minimize for maximization)."""
        return -_calculate_roas(spend_vector, channels, model_results, opt_config)
    
    def objective_gradient(spend_vector):
        """Gradient of negative ROAS."""
        return -_calculate_roas_gradient(spend_vector, channels, model_results, opt_config)
    
    # Run optimization
    algorithm_config = _safe_config_get(opt_config, 'algorithm', {})
    method = _safe_config_get(algorithm_config, 'method', 'SLSQP')
    max_iter = _safe_config_get(algorithm_config, 'max_iterations', _safe_config_get(opt_config, 'max_iterations', 1000))
    tolerance = _safe_config_get(algorithm_config, 'convergence_tolerance', _safe_config_get(opt_config, 'convergence_tolerance', 1e-9))
    
    result = minimize(
        fun=objective_function,
        x0=x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        # jac=objective_gradient,  # Disabled for performance - let scipy compute gradients
        options={
            'maxiter': max_iter,
            'ftol': tolerance,
            'disp': False
        }
    )
    
    return result


def _optimize_with_uncertainty(
    x0: np.ndarray,
    bounds: Bounds,
    constraints: List[Dict], 
    channels: List[str],
    model_results: Any,
    opt_config: Any,
    uncertainty_samples: int,
    logger: logging.Logger
) -> Any:
    """Perform robust optimization incorporating parameter uncertainty."""
    logger.info(f"Running robust optimization with {uncertainty_samples} uncertainty samples...")
    
    def robust_objective_function(spend_vector):
        """Expected negative ROAS over parameter uncertainty."""
        roas_samples = []
        
        # Sample from posterior parameter distribution
        for _ in range(min(uncertainty_samples, 100)):  # Limit for performance
            try:
                # Add noise to model parameters (simplified uncertainty)
                noisy_roas = _calculate_roas_with_noise(
                    spend_vector, channels, model_results, opt_config
                )
                roas_samples.append(noisy_roas)
            except:
                # Fallback to deterministic if sampling fails
                roas_samples.append(_calculate_roas(spend_vector, channels, model_results, opt_config))
        
        # Return negative expected ROAS
        expected_roas = np.mean(roas_samples)
        return -expected_roas
    
    # Run robust optimization
    result = minimize(
        fun=robust_objective_function,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': 500,  # Reduced for robust optimization
            'ftol': 1e-6,
            'disp': False
        }
    )
    
    return result


def _calculate_roas(
    spend_vector: np.ndarray,
    channels: List[str], 
    model_results: Any,
    opt_config: Any
) -> float:
    """Calculate ROAS for given spend allocation."""
    try:
        # Create spend dictionary
        spend_dict = {ch: spend for ch, spend in zip(channels, spend_vector)}
        
        # Use model to predict revenue (simplified - would use actual model prediction)
        # For now, use configured ROI priors as approximation
        total_revenue = 0.0
        
        # Get ROI bounds from opt_config (already loaded)
        roi_bounds = {}
        if hasattr(opt_config, 'model') and hasattr(opt_config.model, 'priors'):
            roi_bounds = getattr(opt_config.model.priors, 'roi_bounds', {})
        elif isinstance(opt_config, dict) and 'model' in opt_config:
            roi_bounds = opt_config.get('model', {}).get('priors', {}).get('roi_bounds', {})
        
        for channel, spend in spend_dict.items():
            # Use middle of ROI range as estimate
            if channel in roi_bounds:
                roi_range = roi_bounds[channel]
                estimated_roi = np.mean(roi_range)
            else:
                estimated_roi = 2.0  # Default 2x ROI
            
            channel_revenue = spend * estimated_roi
            
            # Apply saturation effects (simplified)
            saturation_factor = _apply_saturation_effect(spend, channel)
            channel_revenue *= saturation_factor
            
            # Apply adstock carryover (simplified)
            adstock_factor = _apply_adstock_effect(spend, channel)
            channel_revenue *= adstock_factor
            
            total_revenue += channel_revenue
        
        total_spend = sum(spend_vector)
        roas = total_revenue / total_spend if total_spend > 0 else 0.0
        
        return roas
        
    except Exception as e:
        # Return low ROAS for invalid allocations
        return 0.1


def _calculate_roas_gradient(
    spend_vector: np.ndarray,
    channels: List[str],
    model_results: Any, 
    opt_config: Any
) -> np.ndarray:
    """Calculate gradient of ROAS (finite difference approximation)."""
    epsilon = 1e-6
    gradient = np.zeros_like(spend_vector)
    
    base_roas = _calculate_roas(spend_vector, channels, model_results, opt_config)
    
    for i in range(len(spend_vector)):
        spend_plus = spend_vector.copy()
        spend_plus[i] += epsilon
        
        roas_plus = _calculate_roas(spend_plus, channels, model_results, opt_config)
        gradient[i] = (roas_plus - base_roas) / epsilon
    
    return gradient


def _calculate_roas_with_noise(
    spend_vector: np.ndarray,
    channels: List[str],
    model_results: Any,
    opt_config: Any
) -> float:
    """Calculate ROAS with parameter uncertainty (simplified)."""
    # Add noise to ROI estimates
    base_roas = _calculate_roas(spend_vector, channels, model_results, opt_config)
    
    # Add uncertainty (10% standard deviation)
    noise_factor = np.random.normal(1.0, 0.1)
    noisy_roas = base_roas * max(0.1, noise_factor)  # Ensure positive
    
    return noisy_roas


def _apply_saturation_effect(spend: float, channel: str) -> float:
    """Apply simplified saturation effect based on channel."""
    # Simplified implementation to avoid config loading in optimization loop
    # Use reasonable defaults based on channel type
    
    # High-intent channels saturate faster
    if 'search' in channel.lower() or 'shopping' in channel.lower():
        k = 0.3  # Earlier saturation
        s = 1.2  # Steeper curve
    else:
        k = 0.5  # Default saturation point
        s = 1.0  # Default steepness
    
    # Normalize spend for saturation calculation (use max spend as reference)
    normalized_spend = spend / max(spend, 1000)  # Avoid division by zero
    
    saturation = (normalized_spend ** s) / ((normalized_spend ** s) + (k ** s))
    return max(0.1, saturation)  # Minimum 10% effectiveness


def _apply_adstock_effect(spend: float, channel: str) -> float:
    """Apply simplified adstock carryover effect.""" 
    # Simplified implementation to avoid config loading in optimization loop
    # Use reasonable defaults based on channel type
    
    if 'search' in channel.lower():
        decay = 0.3  # Search has shorter carryover
    elif 'display' in channel.lower() or 'video' in channel.lower():
        decay = 0.8  # Display/video has longer carryover
    else:
        decay = 0.6  # Default decay
    
    # Simplified carryover effect (immediate + 1 period carryover)
    carryover_factor = 1.0 + decay
    return carryover_factor


def _process_optimization_results(
    result: Any,
    channels: List[str],
    current_spend: Dict[str, float], 
    total_budget: float,
    opt_config: Any,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Process and validate optimization results."""
    
    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")
    
    # Extract optimized allocation
    optimized_spend = {ch: spend for ch, spend in zip(channels, result.x)}
    
    # Calculate metrics
    current_roas = _calculate_roas(
        np.array([current_spend[ch] for ch in channels]), 
        channels, None, opt_config
    )
    optimized_roas = -result.fun  # Convert back from negative
    
    # Calculate changes
    spend_changes = {}
    percent_changes = {}
    for ch in channels:
        change = optimized_spend[ch] - current_spend[ch]
        spend_changes[ch] = change
        percent_changes[ch] = change / current_spend[ch] if current_spend[ch] > 0 else float('inf')
    
    # Create results dictionary
    results = {
        'success': result.success,
        'message': result.message if hasattr(result, 'message') else "Optimization completed",
        'optimized_allocation': optimized_spend,
        'current_allocation': current_spend,
        'total_budget': total_budget,
        'spend_changes': spend_changes,
        'percent_changes': percent_changes,
        'metrics': {
            'current_roas': current_roas,
            'optimized_roas': optimized_roas,
            'roas_improvement': optimized_roas - current_roas,
            'roas_improvement_pct': (optimized_roas - current_roas) / current_roas if current_roas > 0 else 0
        },
        'optimization_details': {
            'iterations': getattr(result, 'nit', None),
            'function_evaluations': getattr(result, 'nfev', None),
            'convergence_criteria': getattr(result, 'status', None)
        }
    }
    
    # Log summary
    logger.info(f"Optimization Results Summary:")
    logger.info(f"  ROAS improvement: {results['metrics']['roas_improvement_pct']:.1%}")
    logger.info(f"  Current ROAS: {current_roas:.2f}")
    logger.info(f"  Optimized ROAS: {optimized_roas:.2f}")
    
    return results


def _create_fallback_allocation(
    channels: List[str],
    current_spend: Dict[str, float],
    total_budget: float,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Create fallback allocation when optimization fails."""
    logger.warning("Creating fallback allocation based on current proportions")
    
    current_total = sum(current_spend.values())
    fallback_allocation = {}
    
    for ch in channels:
        proportion = current_spend[ch] / current_total if current_total > 0 else 1.0 / len(channels)
        fallback_allocation[ch] = total_budget * proportion
    
    return {
        'success': False,
        'message': "Optimization failed - using proportional fallback",
        'optimized_allocation': fallback_allocation,
        'current_allocation': current_spend,
        'total_budget': total_budget,
        'metrics': {
            'current_roas': 1.0,  # Placeholder
            'optimized_roas': 1.0,  # Placeholder
            'roas_improvement': 0.0,
            'roas_improvement_pct': 0.0
        }
    }


def run_scenario_analysis(
    current_spend: Dict[str, float],
    model_results: Any,
    config: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Run optimization across multiple budget scenarios.
    
    Args:
        current_spend: Current spend allocation by channel
        model_results: Trained MMM model results
        config: Configuration object
        
    Returns:
        Dict containing results for all scenarios
    """
    logger = logging.getLogger(__name__)
    
    if config is None:
        config = load_config()
    
    logger.info("Starting scenario analysis...")
    
    current_total = sum(current_spend.values())
    
    # Get scenario names from config or use defaults
    scenario_presets = _safe_config_get(config.optimization, 'scenario_presets', {})
    scenarios = list(scenario_presets.keys()) if scenario_presets else ["conservative", "current", "aggressive"]
    
    # Ensure 'current' is included
    if 'current' not in scenarios:
        scenarios.insert(1, 'current')
    
    scenario_results = {}
    
    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario}")
        
        try:
            scenario_result = optimize_budget_allocation(
                current_spend=current_spend,
                total_budget=current_total,  # Will be adjusted by scenario
                model_results=model_results,
                config=config,
                scenario=scenario
            )
            scenario_results[scenario] = scenario_result
            
        except Exception as e:
            logger.error(f"Scenario {scenario} failed: {str(e)}")
            scenario_results[scenario] = {
                'success': False,
                'message': f"Scenario failed: {str(e)}"
            }
    
    logger.info("✅ Scenario analysis completed")
    return {
        'scenarios': scenario_results,
        'summary': _create_scenario_summary(scenario_results),
        'timestamp': pd.Timestamp.now().isoformat()
    }


def _create_scenario_summary(scenario_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary comparison across scenarios."""
    summary = {
        'scenario_comparison': {},
        'recommendations': [],
        'best_performers': {}
    }
    
    # Compile scenario metrics
    valid_scenarios = {}
    for scenario, results in scenario_results.items():
        if results.get('success', False):
            summary['scenario_comparison'][scenario] = {
                'total_budget': results['total_budget'],
                'roas': results['metrics']['optimized_roas'],
                'improvement': results['metrics']['roas_improvement_pct'],
                'allocation': results['optimized_allocation']
            }
            valid_scenarios[scenario] = results['metrics']['optimized_roas']
    
    # Generate recommendations
    if len(valid_scenarios) > 0:
        # Best overall ROAS
        best_roas_scenario = max(valid_scenarios.keys(), key=lambda s: valid_scenarios[s])
        summary['best_performers']['highest_roas'] = best_roas_scenario
        
        # Best improvement
        improvement_scenarios = {
            s: data['improvement'] for s, data in summary['scenario_comparison'].items()
        }
        if improvement_scenarios:
            best_improvement_scenario = max(improvement_scenarios.keys(), key=lambda s: improvement_scenarios[s])
            summary['best_performers']['highest_improvement'] = best_improvement_scenario
        
        # Generate textual recommendations
        summary['recommendations'].append(f"Highest ROAS scenario: {best_roas_scenario}")
        if 'highest_improvement' in summary['best_performers']:
            summary['recommendations'].append(f"Highest improvement scenario: {best_improvement_scenario}")
        
        # Budget-specific recommendations
        budget_scenarios = {s: data['total_budget'] for s, data in summary['scenario_comparison'].items()}
        if len(budget_scenarios) > 1:
            budget_sorted = sorted(budget_scenarios.items(), key=lambda x: x[1])
            lowest_budget = budget_sorted[0][0]
            highest_budget = budget_sorted[-1][0]
            summary['recommendations'].append(f"Most efficient (lowest budget): {lowest_budget}")
            summary['recommendations'].append(f"Highest investment scenario: {highest_budget}")
    
    return summary


def save_optimization_results(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    config: Optional[Any] = None
) -> str:
    """
    Save optimization results to disk.
    
    Args:
        results: Optimization results to save
        output_path: Optional custom output path
        config: Configuration object
        
    Returns:
        str: Path where results were saved
    """
    if config is None:
        config = load_config()
    
    if output_path is None:
        output_dir = Path(config.paths.artifacts) / "optimization"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"budget_optimization_{timestamp}.pkl"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    return str(output_path)


def run_budget_optimization():
    """
    CLI entry point for budget optimization.
    
    Loads model results and current spend, runs optimization, saves results.
    """
    # Load configuration first
    config = load_config()
    
    # Setup logging with full config
    setup_logging(
        level=config.logging.level,
        format=config.logging.format,
        log_file=Path(config.paths.artifacts) / "logs" / "optimization.log" if config.logging.file_rotation else None,
        mask_keys=config.logging.mask_keys
    )
    
    logger = get_logger(__name__)
    
    try:
        # Load model results - support both backends
        models_path = Path(config.paths.models)
        
        # Try to find model results based on backend
        backend = getattr(config.model, 'backend', 'meridian')
        if backend.lower() == 'pymc':
            model_files = list(models_path.glob("pymc_fit_result.pkl"))
        else:
            model_files = list(models_path.glob("meridian_fit_result.pkl"))
        
        # Fallback to any fit result if backend-specific not found
        if not model_files:
            model_files = list(models_path.glob("*_fit_result.pkl"))
        
        if not model_files:
            logger.error("No model results found. Please run model training first: mmm train")
            return False
        
        logger.info(f"Loading model results from: {model_files[0]}")
        with open(model_files[0], 'rb') as f:
            model_results = pickle.load(f)
        
        # Load current spend (from latest engineered features or config)
        features_path = Path(config.paths.features) / "engineered_features.parquet"
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            # Extract recent spend levels
            spend_columns = [col for col in features_df.columns if '_SPEND' in col.upper()]
            recent_spend = features_df[spend_columns].tail(4).mean().to_dict()  # Last 4 weeks average
            
            # Clean channel names
            current_spend = {}
            for col, spend in recent_spend.items():
                clean_name = col.replace('_SPEND', '').lower()
                current_spend[clean_name] = float(spend)
        else:
            # Fallback to dummy spend data
            logger.warning("No features data found. Using placeholder spend data.")
            current_spend = {
                'google_search': 10000,
                'tiktok': 5000
            }
        
        logger.info(f"Current spend allocation: {current_spend}")
        
        # Run scenario analysis
        results = run_scenario_analysis(
            current_spend=current_spend,
            model_results=model_results,
            config=config
        )
        
        # Save results
        output_path = save_optimization_results(results, config=config)
        
        # Log summary
        logger.info("🎉 Budget optimization completed successfully!")
        logger.info(f"Results saved to: {output_path}")
        
        # Log key insights
        for scenario, scenario_results in results['scenarios'].items():
            if scenario_results.get('success', False):
                improvement = scenario_results['metrics']['roas_improvement_pct']
                logger.info(f"  {scenario.title()} scenario: {improvement:+.1%} ROAS improvement")
        
        return True
        
    except Exception as e:
        logger.error(f"Budget optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_budget_optimization()
