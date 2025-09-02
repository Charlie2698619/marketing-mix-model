# Configuration Tuning Guide for Hierarchical Bayesian MMM

This guide explains how to tune the parameters in `config/main.yaml` to optimize your Marketing Mix Model performance for different business scenarios and data characteristics.

## Table of Contents

1. [Quick Start Templates](#quick-start-templates)
2. [Data Configuration](#data-configuration)
3. [Adstock Transformation Tuning](#adstock-transformation-tuning)
4. [Digital Metrics Configuration](#digital-metrics-configuration)
5. [Advanced Feature Engineering](#advanced-feature-engineering)
6. [Model Parameters](#model-parameters)
7. [Performance Optimization](#performance-optimization)
8. [Business Scenario Templates](#business-scenario-templates)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Quick Start Templates

### Conservative/Stable Business
```yaml
features:
  adstock:
    default_decay: 0.8  # Higher retention for stable business
model:
  priors:
    adstock_decay: [0.5, 0.95]  # Conservative decay range
training:
  runtime_guardrails:
    max_hours: 2  # Shorter training for stable patterns
```

### Aggressive/Growth Business
```yaml
features:
  adstock:
    default_decay: 0.6  # Lower retention for dynamic business
model:
  priors:
    adstock_decay: [0.2, 0.8]  # Wider decay range
training:
  runtime_guardrails:
    max_hours: 6  # Longer training for complex patterns
```

### Digital-First Business
```yaml
data:
  digital_specific:
    attribution_windows:
      view_through: 3  # Longer view-through for digital
      click_through: 14  # Extended click attribution
    audience_overlap: true  # Important for digital platforms
```

## Data Configuration

### Frequency Settings
```yaml
data:
  frequency: "daily"    # Use for granular analysis
  # frequency: "weekly"  # Use for stable, long-term patterns
```

**When to use daily:**
- High volatility in spending/performance
- Need to capture intra-week patterns
- Short campaign cycles
- Real-time optimization needs

**When to use weekly:**
- Stable spending patterns
- Long campaign cycles (>4 weeks)
- Limited data volume
- Computational constraints

### Geographic and Brand Filtering
```yaml
data:
  brands: ["all"]           # All brands combined
  # brands: ["brand_1", "brand_2"]  # Specific brands only
  regions: ["US"]           # Single market
  # regions: ["US", "UK", "CA"]     # Multi-market analysis
```

**Multi-brand considerations:**
- Use hierarchical modeling when brands have different behaviors
- Consider separate models for very different brand portfolios
- Account for brand interaction effects

### Revenue and Volume Targets
```yaml
data:
  outcome: "revenue"        # Primary KPI
  # outcome: "volume"       # For volume-focused businesses
  revenue_col: "ALL_PURCHASES_ORIGINAL_PRICE"  # Gross revenue
  volume_col: "ALL_PURCHASES"                  # Transaction count
```

## Adstock Transformation Tuning

### Platform-Specific Decay Patterns

#### Search and Shopping (Immediate Intent)
```yaml
features:
  adstock:
    platform_overrides:
      google_search:
        type: "geometric"
        decay: 0.7          # Quick decay for immediate intent
        lambda: 0.8         # Alternative parameterization
      google_shopping:
        type: "geometric"
        decay: 0.8          # Slightly longer for product research
```

**Tuning guidelines:**
- **High intent keywords**: decay 0.5-0.7 (quick conversion)
- **Research keywords**: decay 0.7-0.9 (longer consideration)
- **Branded search**: decay 0.3-0.6 (immediate response)

#### Display and Video (Awareness Building)
```yaml
features:
  adstock:
    platform_overrides:
      google_display:
        type: "weibull"
        shape: 1.2          # Increasing hazard (building awareness)
        scale: 3.0          # Peak effect at ~2-3 weeks
        decay: 0.85         # Long carryover for awareness
      google_video:
        type: "weibull"
        shape: 0.8          # Decreasing hazard (immediate then decay)
        scale: 4.0          # Peak effect at ~3-4 weeks
```

**Weibull shape parameter guide:**
- **shape < 1**: Decreasing hazard (immediate impact, then decay)
- **shape = 1**: Constant hazard (exponential decay)
- **shape > 1**: Increasing hazard (building momentum)

**Scale parameter guide:**
- **1-2 weeks**: Short-term campaigns, direct response
- **3-4 weeks**: Standard awareness campaigns
- **5-8 weeks**: Brand building, long consideration cycles

#### Social Media (Viral/Engagement Patterns)
```yaml
features:
  adstock:
    platform_overrides:
      meta_facebook:
        type: "weibull"
        shape: 0.9          # Slight viral decay
        scale: 2.5          # 2-week engagement cycle
        decay: 0.6          # Moderate carryover
      tiktok:
        type: "weibull"
        shape: 0.7          # Strong viral pattern
        scale: 1.5          # 1-week viral cycle
        decay: 0.5          # Short carryover (ephemeral content)
```

### Advanced Adstock Parameters
```yaml
features:
  adstock:
    advanced_params:
      convolve_func: "adstock_hill"    # Standard transformation
      # convolve_func: "adstock_geometric"  # Simple exponential
      # convolve_func: "adstock_weibull"    # Complex decay patterns
      normalizing: true                # Always keep true
      mode: "multiplicative"           # Standard approach
      max_lag: 8                       # 8 weeks maximum carryover
      # max_lag: 12                    # For longer B2B cycles
```

## Digital Metrics Configuration

### Attribution Windows
```yaml
data:
  digital_specific:
    attribution_windows:
      view_through: 1      # Conservative view-through
      # view_through: 7    # Aggressive view-through for brand campaigns
      click_through: 7     # Standard click attribution
      # click_through: 30  # Extended for B2B/high consideration
```

**Industry benchmarks:**
- **E-commerce**: view_through: 1, click_through: 7
- **B2B**: view_through: 7, click_through: 30
- **Travel**: view_through: 3, click_through: 14
- **Finance**: view_through: 1, click_through: 21

### Platform Metrics Selection
```yaml
data:
  digital_specific:
    platform_metrics:
      clicks: [
        "GOOGLE_PAID_SEARCH_CLICKS",
        "META_FACEBOOK_CLICKS"    # Start with core platforms
        # Add more as needed
      ]
      impressions: [
        "GOOGLE_PAID_SEARCH_IMPRESSIONS",
        "META_FACEBOOK_IMPRESSIONS"
      ]
      organic_traffic: [
        "DIRECT_CLICKS",
        "BRANDED_SEARCH_CLICKS"   # Focus on owned traffic
      ]
```

### Audience Overlap Handling
```yaml
data:
  digital_specific:
    audience_overlap: true     # Enable for multi-platform campaigns
    # audience_overlap: false  # Disable for single-platform analysis
features:
  attribution:
    overlap_penalty: 0.1       # Light penalty (most businesses)
    # overlap_penalty: 0.3     # Heavy penalty (highly overlapped audiences)
```

## Advanced Feature Engineering

### Creative Fatigue Modeling
```yaml
features:
  creative_fatigue:
    enabled: true               # Enable creative fatigue detection
    half_life: 14               # Days for effectiveness to halve
    # half_life: 7              # Aggressive fatigue (fast-changing creative)
    # half_life: 28             # Conservative fatigue (evergreen creative)
    refresh_signal: "weekly_creative_change"    # Signal for creative refresh
```

**Creative fatigue tuning by industry:**

**Fashion/Lifestyle (High Creative Turnover)**
```yaml
creative_fatigue:
  enabled: true
  half_life: 7                  # Weekly creative refreshes
  refresh_signal: "weekly_creative_change"
```

**B2B/Professional Services (Stable Creative)**
```yaml
creative_fatigue:
  enabled: true
  half_life: 42                 # 6-week creative lifecycle
  refresh_signal: "monthly_creative_change"
```

**Technology/Gaming (Dynamic Creative)**
```yaml
creative_fatigue:
  enabled: true
  half_life: 10                 # 10-day creative cycle
  refresh_signal: "campaign_launch_detection"
```

### Multi-Touch Attribution Configuration
```yaml
features:
  attribution:
    view_through_weight: 0.3    # Weight for view-through conversions
    # view_through_weight: 0.1  # Conservative view-through credit
    # view_through_weight: 0.5  # Aggressive view-through credit
    assisted_conversion_weight: 0.4    # Mid-funnel touchpoint credit
    overlap_penalty: 0.1        # Penalty for audience overlap
```

**Attribution tuning by customer journey length:**

**Short Journey (Impulse Purchase)**
```yaml
attribution:
  view_through_weight: 0.2
  assisted_conversion_weight: 0.3
  overlap_penalty: 0.05
```

**Long Journey (Considered Purchase)**
```yaml
attribution:
  view_through_weight: 0.4
  assisted_conversion_weight: 0.6
  overlap_penalty: 0.15
```

### Baseline Controls and External Factors
```yaml
features:
  baseline:
    trend: true                 # Include business growth trend
    macro_variables: []         # External economic factors
    # macro_variables: ["gdp_growth", "unemployment_rate", "consumer_confidence"]
```

**Advanced baseline configuration:**
```yaml
baseline:
  trend: true
  macro_variables: [
    "gdp_growth",               # Economic growth impact
    "unemployment_rate",        # Consumer spending capacity
    "consumer_confidence",      # Market sentiment
    "seasonal_index"            # Industry seasonality
  ]
  business_cycles: true         # Include economic cycles
  external_events: true        # Model market shocks
```

### Custom Business Terms
```yaml
features:
  custom_terms:
    promo_flag: 
      enabled: true             # Enable promotional modeling
      # enabled: false          # Disable if no promotions
      sign_constraint: "positive"  # Promotions increase sales
      # sign_constraint: "unconstrained"  # Allow negative promotional effects
```

**Promotional modeling by business type:**

**Retail/E-commerce (Frequent Promotions)**
```yaml
custom_terms:
  promo_flag:
    enabled: true
    sign_constraint: "positive"
    detection_sensitivity: "high"     # Detect smaller promotions
    seasonal_promotions: true         # Include holiday patterns
```

**B2B/Services (Rare Promotions)**
```yaml
custom_terms:
  promo_flag:
    enabled: false              # Disable if promotions are rare
    # OR if modeling them:
    # enabled: true
    # detection_sensitivity: "low"    # Only major promotions
```

**Subscription Business (Price Testing)**
```yaml
custom_terms:
  promo_flag:
    enabled: true
    sign_constraint: "unconstrained"  # Allow price increase effects
    price_elasticity: true            # Model price sensitivity
```

## Model Parameters

### ROI Prior Bounds (Critical for Performance)
```yaml
model:
  priors:
    roi_bounds:
      google_search: [0.5, 8.0]      # Conservative search ROI
      # google_search: [1.0, 15.0]   # Aggressive search ROI
      google_shopping: [1.0, 12.0]   # High-intent shopping
      google_display: [0.3, 4.0]     # Awareness/display
      meta_facebook: [0.6, 10.0]     # Social targeting
      tiktok: [0.4, 6.0]             # Viral potential
```

**ROI tuning by business type:**

**E-commerce (High Conversion)**
```yaml
roi_bounds:
  google_search: [1.0, 12.0]
  google_shopping: [2.0, 20.0]
  meta_facebook: [0.8, 15.0]
```

**B2B (Long Sales Cycle)**
```yaml
roi_bounds:
  google_search: [0.3, 5.0]
  google_display: [0.1, 2.0]
  meta_facebook: [0.2, 3.0]
```

**Brand/Awareness (Lower Direct ROI)**
```yaml
roi_bounds:
  google_search: [0.2, 3.0]
  google_display: [0.1, 1.5]
  meta_facebook: [0.1, 2.0]
```

### Saturation Parameters
```yaml
features:
  saturation:
    type: "hill"               # S-curve saturation
    default_inflection: 0.5    # 50% of max spend for inflection
    # default_inflection: 0.3  # Early saturation (competitive markets)
    # default_inflection: 0.7  # Late saturation (growth markets)
    default_slope: 1.0         # Standard curve steepness
    # default_slope: 2.0       # Steep saturation (quick diminishing returns)
    # default_slope: 0.5       # Gentle saturation (gradual diminishing returns)
```

### Backend Selection
```yaml
model:
  backend: "meridian"          # Recommended for production
  # backend: "pymc"            # For custom modeling needs
```

**Meridian vs PyMC:**
- **Meridian**: Google's production-ready framework, faster, more stable
- **PyMC**: More flexible, custom distributions, research use cases

## Performance Optimization

### Training Configuration
```yaml
training:
  rolling_splits:
    window_weeks: 104          # 2 years training data
    # window_weeks: 156        # 3 years for seasonal businesses
    # window_weeks: 52         # 1 year for fast-changing markets
    step_weeks: 13             # Quarterly validation
    # step_weeks: 4            # Monthly validation for dynamic markets
  
  runtime_guardrails:
    max_hours: 4               # Prevent runaway training
    memory_limit_gb: 16        # Adjust based on your system
    # memory_limit_gb: 32      # For large datasets
```

### Model Complexity Management
```yaml
complexity:
  identifiability:
    max_rhat: 1.1              # Convergence threshold (stricter: 1.05)
    min_ess: 400               # Minimum effective sample size
    max_param_correlation: 0.9  # Parameter correlation limit
    auto_simplify_on_fail: true # Automatic model simplification
```

**Convergence troubleshooting:**
- **max_rhat > 1.1**: Increase warmup samples, check priors
- **min_ess < 400**: Increase total samples, check for parameter correlation
- **High correlation**: Review feature engineering, consider regularization

### Backend-Specific Tuning

#### Meridian Parameters
```yaml
model:
  backend_params:
    meridian:
      draws: 2000              # Standard for production
      # draws: 4000            # For high-stakes decisions
      # draws: 1000            # For quick prototyping
      warmup: 1000             # 50% warmup ratio
      chains: 4                # Standard for most systems
      # chains: 8              # For faster convergence on multi-core systems
      target_accept: 0.8       # Conservative acceptance rate
      # target_accept: 0.95    # For difficult posterior geometries
```

## Business Scenario Templates

### Scenario 1: Performance Marketing Focus
```yaml
data:
  digital_specific:
    attribution_windows:
      view_through: 1
      click_through: 7
    audience_overlap: true

features:
  adstock:
    default_decay: 0.7
    platform_overrides:
      google_search:
        type: "geometric"
        decay: 0.6
      meta_facebook:
        type: "geometric"
        decay: 0.7

model:
  priors:
    roi_bounds:
      google_search: [1.0, 15.0]
      meta_facebook: [0.8, 12.0]
```

### Scenario 2: Brand Building Campaign
```yaml
features:
  adstock:
    default_decay: 0.85
    platform_overrides:
      google_display:
        type: "weibull"
        shape: 1.5
        scale: 4.0
        decay: 0.9
      google_video:
        type: "weibull"
        shape: 1.2
        scale: 5.0
        decay: 0.9

model:
  priors:
    roi_bounds:
      google_display: [0.2, 3.0]
      google_video: [0.3, 4.0]
```

### Scenario 3: Experimental/Testing Phase
```yaml
validation:
  outlier_policy: "winsorize"
  outlier_threshold: 0.9

training:
  rolling_splits:
    window_weeks: 52
    step_weeks: 4

complexity:
  identifiability:
    max_rhat: 1.05
    auto_simplify_on_fail: true

model:
  priors:
    adstock_decay: [0.2, 0.95]  # Wide exploration range
```

### Scenario 4: Production/Stable Model
```yaml
training:
  rolling_splits:
    window_weeks: 156
    step_weeks: 13
  
  runtime_guardrails:
    max_hours: 8
    memory_limit_gb: 32

complexity:
  identifiability:
    max_rhat: 1.05
    min_ess: 800

model:
  backend_params:
    meridian:
      draws: 4000
      chains: 8
      target_accept: 0.95
```

### Scenario 5: Competitive Market Analysis
```yaml
features:
  baseline:
    trend: true
    macro_variables: ["gdp_growth", "unemployment_rate", "consumer_confidence"]
    
  creative_fatigue:
    enabled: true
    half_life: 10               # Faster fatigue in competitive markets
    
  custom_terms:
    promo_flag:
      enabled: true
      sign_constraint: "positive"
      competitive_response: true  # Model competitive reactions

# External competitive data integration
external_data:
  competitor_spend: true        # Include competitor spend data
  market_events: true           # Track market disruptions
  economic_indicators: true     # Include macro-economic factors
```

### Scenario 6: Brand Building with Long-Term Effects
```yaml
features:
  adstock:
    default_decay: 0.9          # Long carryover for brand building
    platform_overrides:
      google_display:
        type: "weibull"
        shape: 1.5              # Building momentum
        scale: 6.0              # 6-week peak effect
        decay: 0.95             # Very long carryover
      google_video:
        type: "weibull"
        shape: 1.2
        scale: 8.0              # 8-week peak for video
        decay: 0.9
        
  attribution:
    view_through_weight: 0.5    # Higher view-through credit for brand
    assisted_conversion_weight: 0.7
    
  baseline:
    trend: true
    brand_equity_proxy: true    # Track brand strength over time
```

## Troubleshooting Common Issues

### Model Convergence Problems

**Symptom: High R-hat values (>1.1)**
```yaml
model:
  backend_params:
    meridian:
      warmup: 2000     # Increase warmup
      draws: 4000      # Increase total samples
      target_accept: 0.95  # More conservative sampling

complexity:
  identifiability:
    auto_simplify_on_fail: true  # Enable auto-simplification
```

**Symptom: Low effective sample size**
```yaml
training:
  seeds: [42, 123, 456, 789, 999]  # More random seeds
model:
  priors:
    roi_bounds:
      google_search: [0.8, 6.0]    # Tighter priors
```

### Data Quality Issues

**Symptom: Many missing digital metrics**
```yaml
validation:
  missing_policy: "interpolate"   # Handle missing data
  outlier_policy: "winsorize"     # Robust to outliers

data:
  digital_specific:
    platform_metrics:
      clicks: [
        # Reduce to available metrics only
        "GOOGLE_PAID_SEARCH_CLICKS",
        "META_FACEBOOK_CLICKS"
      ]
```

### Performance Issues

**Symptom: Training takes too long**
```yaml
training:
  runtime_guardrails:
    max_hours: 2             # Reduce time limit

model:
  backend_params:
    meridian:
      draws: 1000            # Reduce samples
      warmup: 500            # Reduce warmup

features:
  adstock:
    advanced_params:
      max_lag: 4             # Reduce carryover window
```

**Symptom: Memory issues**
```yaml
data:
  frequency: "weekly"        # Reduce data granularity

training:
  rolling_splits:
    window_weeks: 52         # Reduce training window

runtime:
  memory_limit_gb: 8         # Set appropriate limit
```

### Attribution Issues

**Symptom: Unrealistic channel attribution**
```yaml
model:
  priors:
    roi_bounds:
      # Tighten bounds based on business knowledge
      google_search: [1.0, 6.0]  # More realistic range

features:
  attribution:
    overlap_penalty: 0.2      # Increase overlap penalty
    
data:
  digital_specific:
    attribution_windows:
      click_through: 3        # Reduce attribution window
```

### Platform-Specific Tuning

**For low-volume platforms:**
```yaml
features:
  adstock:
    platform_overrides:
      tiktok:
        type: "geometric"     # Simpler transformation
        decay: 0.5            # Shorter carryover
```

**For high-volume platforms:**
```yaml
features:
  adstock:
    platform_overrides:
      google_search:
        type: "weibull"       # More complex transformation
        shape: 1.0
        scale: 2.0
        decay: 0.8
```

### Advanced Feature Issues

**Symptom: Creative fatigue not detected**
```yaml
features:
  creative_fatigue:
    enabled: true
    half_life: 7              # Reduce half-life for faster detection
    refresh_signal: "spend_pattern_changes"  # More sensitive detection
    detection_sensitivity: "high"  # Lower threshold for fatigue
```

**Symptom: Attribution model assigns unrealistic credit**
```yaml
features:
  attribution:
    view_through_weight: 0.2  # Reduce view-through credit
    overlap_penalty: 0.25     # Increase overlap penalty
    position_based_weights: [0.4, 0.2, 0.4]  # First/last click focus
```

**Symptom: Baseline controls are too volatile**
```yaml
features:
  baseline:
    trend: true
    macro_variables: ["gdp_growth"]  # Start with single stable variable
    smoothing_window: 14      # Smooth macro variables
    external_validation: true # Validate against known events
```

**Symptom: Custom business terms show inconsistent effects**
```yaml
features:
  custom_terms:
    promo_flag:
      enabled: true
      sign_constraint: "positive"    # Enforce positive promotional effects
      detection_threshold: 0.15      # Higher threshold for promotion detection
      seasonal_adjustment: true      # Adjust for seasonal baseline
```

**Symptom: Competitive effects are unstable**
```yaml
# Simplify competitive modeling
features:
  baseline:
    macro_variables: ["consumer_confidence"]  # Single macro variable
    competitive_pressure: false  # Disable if data is noisy
    market_saturation: true      # Focus on saturation effects only
```

## Monitoring and Iteration

### Key Metrics to Track
1. **Model convergence**: R-hat < 1.1, ESS > 400
2. **Prediction accuracy**: MAPE < 15%, coverage > 80%
3. **Business reasonableness**: ROI within expected ranges
4. **Attribution stability**: Consistent channel contributions
5. **Creative fatigue detection**: Realistic refresh patterns
6. **Baseline trend stability**: Smooth, interpretable trends
7. **Promotional effect consistency**: Positive promotional impacts
8. **Competitive response realism**: Logical market dynamics

### Advanced Feature Validation
```yaml
# Monitor creative fatigue patterns
creative_fatigue:
  validation:
    refresh_detection_rate: [0.1, 0.3]  # 10-30% of periods should show refreshes
    fatigue_decay_consistency: true     # Monotonic decay between refreshes
    
# Validate attribution consistency
attribution:
  validation:
    total_attribution_sum: [0.95, 1.05] # Attribution should sum to ~1.0
    channel_contribution_stability: 0.2  # Max 20% change period-over-period
    
# Check baseline controls
baseline:
  validation:
    trend_smoothness: 0.1               # Smooth trend changes
    macro_correlation_bounds: [0.1, 0.8] # Reasonable macro variable correlations
```

### Iterative Tuning Process
1. **Start conservative**: Tight priors, simple transformations
2. **Validate convergence**: Check all diagnostic metrics
3. **Expand complexity**: Add platform-specific parameters
4. **Business validation**: Review results with domain experts
5. **Production deployment**: Use stable, well-tested configuration

### Configuration Versioning
```yaml
project:
  name: "conjura_mmm_analysis_v2"    # Version your configs
  description: "Enhanced with Weibull adstock - Aug 2025"
```

Keep track of configuration changes and their impact on model performance for systematic optimization.
