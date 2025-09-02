# MMM Pipeline Configuration Usage Guide

## 🎯 Overview
Your `config/main.yaml` has been reorganized to follow the natural pipeline flow from data ingestion to reporting. Each configuration section is now clearly labeled with:
- **Stage**: Which pipeline step it belongs to
- **Hint**: When and how to adjust these settings  
- **Scripts**: Which files use these configurations

## 📋 Pipeline Stages & Configuration Sections

### 🚀 INITIALIZATION (Setup Once)
- **project**: Project metadata and identification
- **paths**: File system paths for different environments
- **Hint**: Set these once at project start, rarely needs changes

### 📥 1. INGEST (Data Loading)
- **ingest**: Data source connectors (Figshare, CSV, etc.)
- **data**: Raw data structure mapping and channel definitions
- **Scripts**: `src/mmm/data/ingest.py`, `src/mmm/cli.py (ingest)`
- **Hint**: Map your raw data columns to standardized MMM format

### ✅ 2. VALIDATE (Data Quality)
- **validation**: Comprehensive data quality checks and business rules
- **Scripts**: `src/mmm/data/validate.py`
- **Hint**: Adjust thresholds based on your data quality requirements
- **Key settings**: 
  - `min_weeks_required`: Minimum data history (52+ weeks)
  - `max_channel_correlation`: Avoid multicollinearity (0.9)
  - `roi_range`: Plausible ROI bounds

### 🔄 3. TRANSFORM (Data Cleaning)
- **enhanced_cleaning**: Smart data cleaning policies
- **Scripts**: `src/mmm/data/transform.py`
- **Hint**: Configure cleaning policies based on validation findings
- **Key settings**:
  - `outliers.policy`: "contextual_winsorization" preserves business events
  - `negative_values`: Route to adjustment columns vs clipping
  - `missing_data`: Campaign-aware imputation

### ⚙️ 4. FEATURES (Feature Engineering)
- **features**: Transform data into model-ready features
- **Scripts**: `src/mmm/features/engineer.py` and feature modules
- **Hint**: Start with defaults, tune based on model performance

#### Feature Sub-modules:
- **adstock**: Media carryover effects (decay rates per channel)
- **saturation**: Diminishing returns modeling (Hill/logistic curves)
- **seasonality**: Time-based patterns (holidays, trends, Fourier)
- **baseline**: Non-media factors (events, macro variables)
- **attribution**: Cross-channel credit distribution (FUTURE EXTENSION)
- **creative_fatigue**: Creative effectiveness decay (FUTURE EXTENSION)

### 🤖 5. TRAIN (Model Training)
- **model**: Bayesian model structure and priors
- **training**: Cross-validation and training procedures  
- **complexity**: Convergence diagnostics and simplification
- **Scripts**: `src/mmm/models/meridian.py`, `src/mmm/models/pymc.py`
- **Hint**: Start with Meridian backend, adjust priors based on business knowledge

#### Key Training Settings:
- `rolling_splits.window_weeks`: Training window (104 = 2 years)
- `complexity.identifiability.max_rhat`: Convergence threshold (1.1)
- `priors.roi_bounds`: Business-informed ROI expectations per channel

### 📈 6. OPTIMIZE (Budget Allocation)
- **optimization**: Optimal spend allocation across channels
- **Scripts**: `src/mmm/optimization/allocator.py`
- **Hint**: Adjust constraints based on business rules
- **Key settings**:
  - `platform_constraints`: Min/max spend per channel
  - `scenario_presets`: Budget change scenarios
  - `uncertainty_propagation`: Include parameter uncertainty

### 🎯 7. EVALUATE (Model Validation)
- **evaluation**: Model quality assessment and diagnostics
- **Scripts**: `src/mmm/evaluation/validator.py`
- **Hint**: Adjust thresholds based on business accuracy requirements
- **Key settings**:
  - `metrics.mape_threshold`: Accuracy requirement (15%)
  - `digital_checks`: Compare to platform attribution
  - `validation_strategies`: Holdout approaches

### 📊 8. REPORT (Insights & Recommendations)
- **reports**: Output generation and dashboard exports
- **Scripts**: `src/mmm/reporting/dashboard.py`
- **Hint**: Enable executive reports for stakeholders

## 🏗️ Infrastructure Configuration
- **logging**: System monitoring and debugging
- **tracking**: MLflow experiment tracking
- **runtime**: Resource management (CPU/GPU, memory)
- **profiles**: Environment-specific settings (local/docker/k8s)
- **orchestration**: End-to-end pipeline management

## 🔮 Future Extensions (Disabled by Default)
- **external_data**: Competitor data, economic indicators
- **privacy**: GDPR compliance, differential privacy
- **synthetic_data**: Testing and validation data
- **attribution**: Advanced attribution modeling
- **creative_fatigue**: Creative refresh detection

## 🚀 Getting Started

### 1. Initial Setup
```bash
# Configure for your environment
vim config/main.yaml  # Edit paths, project name

# Set up data mapping  
# Edit data.channel_map to match your raw data columns
```

### 2. Run Pipeline Stages
```bash
# Full pipeline
mmm run

# Individual stages
mmm ingest     # Stage 1: Load data
mmm validate   # Stage 2: Quality checks  
mmm transform  # Stage 3: Clean data
mmm features   # Stage 4: Engineer features
mmm train      # Stage 5: Train model
mmm optimize   # Stage 6: Budget allocation
mmm evaluate   # Stage 7: Validate model
mmm report     # Stage 8: Generate insights
```

### 3. Iterative Development
- **Start with defaults**: Most settings have sensible defaults
- **Validate early**: Run stages 1-2 first to check data quality
- **Tune gradually**: Adjust parameters based on validation results
- **Monitor convergence**: Watch R-hat and ESS in training logs

### 4. Production Deployment
- Use `profiles` section for environment-specific settings
- Enable `tracking.mlflow` for experiment management
- Set appropriate `runtime.memory_limit_gb` for your system

## 🎯 Common Configuration Adjustments

### Data Quality Issues
```yaml
validation:
  quality_thresholds:
    min_validation_score: 0.7  # Lower if data has known issues
  identifiability:
    max_channel_correlation: 0.95  # Relax if channels are correlated
```

### Model Convergence Problems
```yaml
complexity:
  identifiability:
    max_rhat: 1.2  # Relax convergence criteria
    auto_simplify_on_fail: true  # Enable auto-simplification
```

### Business Constraints
```yaml
optimization:
  platform_constraints:
    min_spend_pct: 0.1  # Adjust based on business rules
    max_spend_pct: 0.5
```

## 📚 Further Reading
- See `CONFIG_TUNING_GUIDE.md` for detailed parameter tuning
- Check individual script docstrings for implementation details
- Review `artifacts/` folder for model outputs and diagnostics
