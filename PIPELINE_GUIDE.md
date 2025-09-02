# MMM Pipeline Guide - Stage-by-Stage Configuration Guide

This guide walks you through each stage of the MMM pipeline and explains which configuration sections to adjust for your specific needs.

## 🚀 Quick Start Checklist

### Before You Begin
- [ ] Set `project.name` to your project identifier
- [ ] Configure `paths.*` for your environment
- [ ] Map your data columns in `data.channel_map`
- [ ] Review and adjust `validation.quality_thresholds` for your data quality standards

### Pipeline Execution Order
```bash
# Run individual stages
mmm ingest      # Load raw data
mmm validate    # Quality checks
mmm transform   # Data cleaning  
mmm features    # Feature engineering
mmm train       # Model training
mmm optimize    # Budget allocation
mmm evaluate    # Model validation
mmm report      # Generate insights

# Or run full pipeline
mmm run         # Execute all stages
```

---

## 📋 Stage 1: INGEST - Data Loading

### Configuration Sections
- `ingest.*` - Data connectors and sources
- `data.*` - Raw data structure and column mapping

### What to Configure
1. **Data Sources** (`ingest.connector_priority`)
   - Prioritize your data connectors (figshare, local_csv, etc.)
   - Configure source-specific settings

2. **Column Mapping** (`data.channel_map`)
   ```yaml
   data:
     channel_map:
       google_search: "YOUR_SEARCH_SPEND_COLUMN"
       meta_facebook: "YOUR_FACEBOOK_SPEND_COLUMN"
       # Add all your media channels
   ```

3. **Data Structure** (`data.keys`, `data.date_col`, `data.revenue_col`)
   - Map your date, revenue, and key columns
   - Set aggregation keys for your analysis level

### Success Criteria
- Raw data loaded successfully
- All required columns mapped
- Data covers minimum time period (52+ weeks recommended)

### Scripts Used
- `src/mmm/data/ingest.py`
- `src/mmm/cli.py` (ingest command)

---

## ✅ Stage 2: VALIDATE - Data Quality Checks

### Configuration Sections
- `validation.*` - All validation rules and thresholds

### What to Configure
1. **Coverage Requirements** (`validation.coverage_frequency`)
   ```yaml
   coverage_frequency:
     min_weeks_required: 52      # Adjust based on your needs
     preferred_weeks: 104        # 2+ years is ideal
     max_missing_weeks_pct: 2.0  # Tolerance for missing data
   ```

2. **Identifiability Checks** (`validation.identifiability`)
   - `max_channel_correlation: 0.9` - Lower if channels are too correlated
   - `min_zero_spend_weeks_pct: 10` - Ensure variation in spend

3. **Business Rules** (`validation.value_sanity`)
   - Set plausible ROI ranges
   - Configure metric tolerance (CPM, CPC)
   - Adjust outlier handling

### Common Adjustments
- **High correlation between channels?** → Lower `max_channel_correlation`
- **Strict data quality requirements?** → Increase `min_validation_score`
- **Different ROI expectations?** → Adjust `roi_range`

### Success Criteria
- Validation score > 0.8 (adjustable)
- No critical data quality issues
- Sufficient spend variation for identifiability

### Scripts Used
- `src/mmm/data/validate.py`

---

## 🔄 Stage 3: TRANSFORM - Data Cleaning

### Configuration Sections
- `enhanced_cleaning.*` - Data cleaning policies and rules

### What to Configure
1. **Outlier Handling** (`enhanced_cleaning.outliers`)
   ```yaml
   outliers:
     policy: "contextual_winsorization"
     exempt_promo_periods: true        # Don't cap during promotions
     winsorize_percentile: 99          # Adjust based on data quality
   ```

2. **Missing Data Strategy** (`enhanced_cleaning.missing_data`)
   - Choose between campaign-aware vs simple imputation
   - Configure pause detection vs outage detection

3. **Duplicate Resolution** (`enhanced_cleaning.duplicates`)
   - Set aggregation rules for conflicts
   - Define additive vs non-additive columns

### Business Context Adjustments
- **Frequent promotions?** → Enable `exempt_promo_periods`
- **Platform outages common?** → Enable outage detection
- **Conservative approach?** → Use higher winsorization percentiles

### Success Criteria
- Clean data with documented transformations
- Outliers handled appropriately
- Missing data strategy applied consistently

### Scripts Used
- `src/mmm/data/transform.py`

---

## ⚙️ Stage 4: FEATURES - Feature Engineering

### Configuration Sections
- `features.adstock.*` - Media carryover effects
- `features.saturation.*` - Diminishing returns
- `features.seasonality.*` - Time patterns
- `features.baseline.*` - Non-media factors

### What to Configure
1. **Adstock (Carryover) by Channel** (`features.adstock.platform_overrides`)
   ```yaml
   platform_overrides:
     google_search:
       decay: 0.7        # Fast decay for search (immediate intent)
     google_display:
       decay: 0.85       # Slower decay for brand awareness
   ```

2. **Saturation by Channel** (`features.saturation.channel_overrides`)
   ```yaml
   channel_overrides:
     google_search:
       k: 0.3           # Earlier saturation for high-intent
     google_display:
       k: 0.7           # Later saturation for awareness
   ```

3. **Seasonality** (`features.seasonality`)
   - Add business-relevant holidays
   - Configure calendar effects
   - Enable/disable weekly vs annual patterns

### Channel-Specific Tuning
- **Search/Shopping**: Fast adstock decay (0.5-0.7), early saturation (k=0.3-0.4)
- **Display/Video**: Slower decay (0.8-0.9), later saturation (k=0.6-0.8)
- **Social**: Medium decay (0.6-0.7), variable saturation based on platform

### Future Extensions (Currently Disabled)
- `features.attribution.*` - When conversion data is available
- `features.creative_fatigue.*` - When creative refresh data is available
- `features.custom_terms.*` - Business-specific features

### Success Criteria
- Features created for all media channels
- Seasonality patterns captured
- Baseline controls included

### Scripts Used
- `src/mmm/features/engineer.py`
- Individual feature modules (`adstock.py`, `saturation.py`, etc.)

---

## 🤖 Stage 5: TRAIN - Model Training

### Configuration Sections
- `model.*` - Model structure and priors
- `training.*` - Training configuration
- `complexity.*` - Convergence diagnostics

### What to Configure
1. **Business Priors** (`model.priors.roi_bounds`)
   ```yaml
   roi_bounds:
     google_search: [0.5, 8.0]     # Expected ROI range for search
     google_display: [0.3, 4.0]    # Lower expected ROI for display
   ```

2. **Training Setup** (`training.rolling_splits`)
   ```yaml
   rolling_splits:
     window_weeks: 104             # 2 years training window
     step_weeks: 13                # Quarterly validation
   ```

3. **Convergence Criteria** (`complexity.identifiability`)
   - `max_rhat: 1.1` - Tighten for better convergence
   - `min_ess: 400` - Increase for more robust estimates

### Backend Choice
- **Meridian** (Google): Production-ready, optimized for MMM
- **PyMC**: More flexible, research-oriented

### Convergence Troubleshooting
- **R-hat > 1.1?** → Increase training time or simplify model
- **Low ESS?** → Check for parameter correlations
- **Slow convergence?** → Review priors or reduce features

### Success Criteria
- Model converges (R-hat < 1.1)
- Sufficient effective samples (ESS > 400)
- Reasonable parameter estimates

### Scripts Used
- `src/mmm/models/meridian.py` or `src/mmm/models/pymc.py`
- `src/mmm/cli.py` (train command)

---

## 📈 Stage 6: OPTIMIZE - Budget Allocation

### Configuration Sections
- `optimization.*` - Optimization objectives and constraints

### What to Configure
1. **Business Constraints** (`optimization.platform_constraints`)
   ```yaml
   platform_constraints:
     min_spend_pct: 0.05          # Minimum 5% per channel
     max_spend_pct: 0.4           # Maximum 40% per channel
   ```

2. **Optimization Goal** (`optimization.objective`)
   - `"roas"` - Maximize return on ad spend
   - `"profit"` - Maximize profit (when cost data available)

3. **Scenario Planning** (`optimization.scenario_presets`)
   - Test different budget levels
   - Compare conservative vs aggressive strategies

### Business Rule Integration
- Set realistic min/max spend constraints
- Consider creative fatigue in optimization
- Account for audience overlap between platforms

### Success Criteria
- Optimal budget allocation generated
- Business constraints satisfied
- Sensitivity analysis completed

### Scripts Used
- `src/mmm/optimization/allocator.py`
- `src/mmm/cli.py` (optimize command)

---

## ✅ Stage 7: EVALUATE - Model Validation

### Configuration Sections
- `evaluation.*` - Validation strategies and thresholds

### What to Configure
1. **Performance Thresholds** (`evaluation.metrics`)
   ```yaml
   metrics:
     mape_threshold: 0.15         # 15% error tolerance
     coverage_threshold: 0.8      # 80% prediction intervals
   ```

2. **Digital Validation** (`evaluation.digital_checks`)
   - Compare to platform-reported attribution
   - Set incrementality thresholds

3. **Validation Strategies** (`evaluation.validation_strategies`)
   - Enable temporal holdout for time series
   - Consider brand/platform holdouts for robustness

### Quality Thresholds
- **MAPE < 15%**: Good model accuracy
- **Coverage ≥ 80%**: Well-calibrated uncertainty
- **Digital alignment**: Within reasonable bounds of platform data

### Success Criteria
- Model meets accuracy thresholds
- Passes business logic validation
- Digital metrics align with external data

### Scripts Used
- `src/mmm/evaluation/validator.py`
- `src/mmm/cli.py` (evaluate command)

---

## 📊 Stage 8: REPORT - Insights & Recommendations

### Configuration Sections
- `reports.*` - Report generation settings

### What to Configure
1. **Report Types** (`reports`)
   ```yaml
   reports:
     executive_deck: true         # Business summary
     dashboard_exports: true      # Data for dashboards
   ```

### Output Types
- **Executive Deck**: High-level business insights
- **Dashboard Data**: Detailed data for visualization tools
- **Attribution Reports**: Channel contribution analysis
- **Optimization Recommendations**: Budget reallocation suggestions

### Success Criteria
- Actionable business insights generated
- Clear recommendations provided
- Results communicated effectively

### Scripts Used
- `src/mmm/reporting/dashboard.py`
- `src/mmm/cli.py` (report command)

---

## 🔧 Infrastructure Configuration

### System Settings
- `logging.*` - Configure logging level and format
- `runtime.*` - Set compute resources (CPU/GPU, memory)
- `tracking.*` - MLflow experiment tracking

### Environment Profiles
- `profiles.local.*` - Local development settings
- `profiles.docker.*` - Container deployment
- `profiles.k8s.*` - Kubernetes cluster settings

---

## 🧪 Future Extensions (Currently Disabled)

### Advanced Features to Enable Later
1. **Attribution Modeling** (`features.attribution`)
   - Enable when conversion journey data is available
   - Configure attribution windows and models

2. **Creative Fatigue** (`features.creative_fatigue`)
   - Enable when creative refresh data is available
   - Configure fatigue detection methods

3. **External Data** (`external_data`)
   - Competitor intelligence
   - Economic indicators
   - Market research integration

4. **Privacy Features** (`privacy`)
   - Differential privacy
   - GDPR compliance
   - iOS 14.5+ attribution modeling

---

## 🚨 Common Issues & Solutions

### Data Quality Issues
- **High channel correlation** → Review media planning strategy
- **Insufficient spend variation** → Ensure test/learn approach
- **Missing data periods** → Investigate platform outages

### Model Convergence Issues
- **High R-hat** → Increase iterations or simplify model
- **Parameter correlations** → Review feature engineering
- **Implausible estimates** → Adjust priors based on business knowledge

### Performance Issues
- **Poor fit** → Check feature engineering and outlier handling
- **Unstable results** → Increase training window or regularization
- **Business logic failures** → Review constraints and validation rules

---

## 📈 Best Practices

### Data Requirements
- **Minimum**: 52 weeks of data with consistent spend
- **Recommended**: 104+ weeks (2+ years) for seasonal patterns
- **Ideal**: Multiple years with varied spend levels and campaign types

### Model Development
1. Start with simple model (default parameters)
2. Validate data quality thoroughly
3. Tune feature engineering based on business knowledge
4. Iterate on priors using domain expertise
5. Validate results against business intuition

### Production Deployment
- Use version control for configuration files
- Implement automated testing pipeline
- Monitor model performance over time
- Update regularly with new data

---

This guide should help you navigate the MMM pipeline configuration effectively. Remember to adjust parameters based on your specific business context and data characteristics!
