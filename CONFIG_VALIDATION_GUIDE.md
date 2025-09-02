# Configuration Validation Testing Guide

This document explains the comprehensive config validation test suite that validates every variable in `main.yaml` is actually used in the scripts and not hardcoded.

## Overview

The validation suite consists of two main test files:

1. **`test_config_usage_validation.py`** - Core validation engine
2. **`test_config_specific_validation.py`** - Specific domain tests

## Features

### ✅ What These Tests Validate

1. **Config Variable Usage**
   - Every config variable is used somewhere in the codebase
   - Variables are accessed through proper config patterns
   - No unused/orphaned config variables

2. **Hardcoding Detection**
   - File paths are not hardcoded (use `config.paths.*`)
   - Numeric thresholds come from config
   - Backend names use `config.model.backend`
   - Log levels use `config.logging.level`

3. **Access Pattern Consistency**
   - Consistent config access patterns within files
   - Prefer `_safe_config_get()` helper functions
   - Avoid mixing Pydantic and dict-style access

4. **Config Structure Validation**
   - All required config sections exist
   - Critical variables are present
   - Values are reasonable/valid

5. **Domain-Specific Validation**
   - Channel mapping usage
   - Path configuration
   - Backend consistency
   - Optimization parameters
   - Feature engineering config
   - Validation thresholds

### ❌ What's Excluded

As requested, these sections are excluded from validation:
- `attribution` (future extension)
- `creative_fatigue` (future extension)  
- `external_data` (future extension)
- `privacy` (future extension)
- `synthetic_data` (testing only)
- `competitors` (future extension)

## Running the Tests

### Quick Run

```bash
# Run all config validation tests
python scripts/validate_config_usage.py

# Verbose output
python scripts/validate_config_usage.py -v
```

### Using Pytest Directly

```bash
# Run specific test file
pytest tests/unit/test_config_usage_validation.py -v

# Run all config validation tests
pytest tests/unit/test_config_*validation*.py -v

# Run with markers
pytest -m config_validation -v

# Generate detailed report
pytest tests/unit/test_config_usage_validation.py::TestConfigUsageValidation::test_generate_config_usage_report -v
```

### CI/CD Integration

```bash
# For CI pipelines - fail fast with minimal output
pytest tests/unit/test_config_*validation*.py --tb=line -q
```

## Test Results and Reports

### Generated Reports

The tests generate a comprehensive report at:
```
artifacts/config_usage_report.md
```

This report includes:
- ✅ All config variables and where they're used
- ❌ Unused config variables  
- ⚠️ Potential hardcoded values
- 📁 Files with inconsistent access patterns

### Example Output

```
# Configuration Usage Validation Report
Generated on: 2025-09-02T10:30:00

## Summary
- Total config variables: 247
- Variables used: 203
- Usage ratio: 82.2%

## Config Variables Usage
✅ project.name
   └── src/mmm/cli.py
   └── src/mmm/config.py

✅ paths.raw  
   └── src/mmm/data/ingest.py
   └── src/mmm/cli.py

## Unused Config Variables
❌ features.creative_fatigue.enabled
❌ external_data.competitor_spend

## Potential Hardcoded Values
📁 src/mmm/optimization/allocator.py
   ⚠️ Line 45: hardcoded_threshold - 0.8
   ⚠️ Line 67: hardcoded_path - "artifacts/optimization"
```

## Test Breakdown

### Core Validation (`test_config_usage_validation.py`)

#### `test_all_config_variables_used`
- Validates that config variables are referenced in code
- Allows up to 20% unused variables
- Reports unused variables for cleanup

#### `test_no_hardcoded_paths`
- Ensures file paths use `config.paths.*`
- Allows max 5 hardcoded paths for tests/defaults
- Detects patterns like `"data/raw"` instead of `config.paths.raw`

#### `test_no_hardcoded_thresholds`  
- Validates numeric thresholds come from config
- Allows max 10 hardcoded for mathematical constants
- Detects patterns like `0.8`, `0.95` that should be configurable

#### `test_no_hardcoded_backends`
- Ensures backend names use `config.model.backend`
- Allows max 2 hardcoded for validation/fallbacks
- Detects `"meridian"`, `"pymc"` strings

#### `test_consistent_config_access_patterns`
- Validates consistent access patterns within files
- Prefers `_safe_config_get()` helper functions
- Allows max 3 files with mixed patterns

#### `test_config_structure_completeness`
- Validates all required top-level sections exist
- Ensures core MMM config sections are present

#### `test_critical_config_variables_exist`
- Validates critical variables like `project.name`, `model.backend`
- Ensures no missing essential configuration

### Specific Domain Tests (`test_config_specific_validation.py`)

#### Channel Configuration
- `test_channel_map_usage` - Channel names via config, not hardcoded
- Validates `data.channel_map` is properly used

#### Path Management  
- `test_path_configuration_usage` - Paths via `config.paths.*`
- Ensures no hardcoded directory paths

#### Backend Consistency
- `test_backend_configuration_consistency` - Backend via config
- Validates model backend access patterns

#### Logging Setup
- `test_logging_configuration_usage` - Log levels via config
- Ensures `config.logging.level` usage

#### Optimization Parameters
- `test_optimization_parameters_usage` - Optimization config usage
- Validates key parameters accessed via config

#### Feature Engineering  
- `test_feature_configuration_usage` - Feature config sections used
- Validates adstock, saturation, seasonality config usage

#### Validation Thresholds
- `test_validation_thresholds_usage` - Thresholds via config
- Ensures validation/evaluation thresholds configurable

## Configuration Access Patterns

### ✅ Recommended Patterns

```python
# Use safe helper function (preferred)
from ..utils.config import _safe_config_get
value = _safe_config_get(config.optimization, 'uncertainty_samples', 1000)

# Direct Pydantic access (good)
value = config.optimization.uncertainty_samples

# With fallback (good)
value = getattr(config.optimization, 'uncertainty_samples', 1000)
```

### ❌ Patterns to Avoid

```python
# Dictionary access on Pydantic objects (inconsistent)
value = config['optimization']['uncertainty_samples']

# Hardcoded values (bad)
uncertainty_samples = 1000  # Should be from config

# Mixed access patterns in same file (confusing)
config.optimization.objective  # Pydantic style
config.get('optimization', {}).get('samples', 1000)  # Dict style
```

## Customizing Tests

### Adding New Validations

To add domain-specific validations:

1. Add test method to `test_config_specific_validation.py`
2. Follow naming convention: `test_[domain]_configuration_usage`
3. Use similar patterns for detecting hardcoded values
4. Set reasonable thresholds for assertion failures

### Excluding Sections

To exclude additional config sections:

```python
# In ConfigUsageValidator.__init__
self.future_sections = {
    'attribution', 'creative_fatigue', 'external_data', 
    'privacy', 'synthetic_data', 'competitors',
    'your_new_section'  # Add here
}
```

### Adjusting Thresholds

Modify assertion thresholds based on project needs:

```python
# Allow more unused variables
assert usage_ratio >= 0.7  # Was 0.8

# Allow more hardcoded paths
assert len(path_issues) <= 10  # Was 5
```

## Integration with Development Workflow

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: config-validation
      name: Config Usage Validation
      entry: python scripts/validate_config_usage.py
      language: system
      pass_filenames: false
```

### GitHub Actions

Add to CI pipeline:

```yaml
- name: Validate Config Usage
  run: |
    python scripts/validate_config_usage.py
    pytest tests/unit/test_config_*validation*.py --junitxml=config-validation-results.xml
```

### Development Guidelines

1. **When adding new config variables:**
   - Add to appropriate section in `main.yaml`
   - Use the variable in at least one script
   - Follow consistent access patterns

2. **When using config in scripts:**
   - Import and use config objects
   - Avoid hardcoding values that should be configurable
   - Use `_safe_config_get()` for robust access

3. **When refactoring:**
   - Run config validation tests
   - Fix any new hardcoding issues
   - Update config structure if needed

## Troubleshooting

### Common Issues

1. **False Positive Unused Variables**
   - Variable used but not detected by regex patterns
   - Add more specific usage detection patterns
   - Check if variable name is too generic

2. **False Positive Hardcoded Values**
   - Common values appearing in multiple contexts
   - Add to exclusion list in `_is_likely_hardcoded()`
   - Improve context detection

3. **Test Failures Due to Project Changes**
   - Update test thresholds as project evolves
   - Adjust patterns for new config access styles
   - Update excluded sections if needed

### Debugging Test Issues

```bash
# Run single test with full output
pytest tests/unit/test_config_usage_validation.py::TestConfigUsageValidation::test_all_config_variables_used -v -s

# Generate report and examine
pytest tests/unit/test_config_usage_validation.py::TestConfigUsageValidation::test_generate_config_usage_report -v
cat artifacts/config_usage_report.md

# Run validator directly for debugging
cd /path/to/project
python -c "
from tests.unit.test_config_usage_validation import ConfigUsageValidator
v = ConfigUsageValidator()
usage = v.find_config_usage()
print(f'Found {len(usage)} used variables')
"
```

## Success Criteria

The validation suite considers the configuration well-managed when:

- ✅ 80%+ of config variables are used
- ✅ <5 hardcoded file paths
- ✅ <10 hardcoded numeric thresholds  
- ✅ <8 hardcoded backend references
- ✅ <3 files with inconsistent access patterns
- ✅ All critical config variables exist
- ✅ Config structure is complete

This ensures your MMM project maintains clean, configurable, and maintainable code!
