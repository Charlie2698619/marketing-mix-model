"""
Additional validation tests for specific config usage patterns
and edge cases in the MMM project.
"""

import pytest
import yaml
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


class SpecificConfigValidationTests:
    """Additional specific tests for config validation."""
    
    @pytest.fixture(scope="class")
    def config_data(self):
        """Load the main config file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "main.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @pytest.fixture(scope="class") 
    def src_files(self):
        """Get all source files."""
        src_dir = Path(__file__).parent.parent.parent / "src" / "mmm"
        return list(src_dir.rglob("*.py"))

    def test_channel_map_usage(self, config_data, src_files):
        """Test that data.channel_map is properly used and not hardcoded."""
        channel_map = config_data.get('data', {}).get('channel_map', {})
        
        # Check that channel names are used via config, not hardcoded
        hardcoded_channels = []
        
        for py_file in src_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Look for hardcoded channel names
                for channel_name in channel_map.keys():
                    # Skip very common words that might appear naturally
                    if len(channel_name) < 5:
                        continue
                        
                    # Look for quoted channel names that might be hardcoded
                    pattern = f'["\\\']\\s*{re.escape(channel_name)}\\s*["\\\']'
                    if re.search(pattern, content):
                        # Check if it's not accessing via config
                        if not re.search(f'channel_map.*{re.escape(channel_name)}', content):
                            hardcoded_channels.append(f"{py_file.name}: {channel_name}")
                            
            except Exception:
                continue
        
        # Allow some hardcoded usage for testing/examples
        assert len(hardcoded_channels) <= 5, f"Too many hardcoded channel names: {hardcoded_channels}"

    def test_path_configuration_usage(self, config_data, src_files):
        """Test that paths are accessed via config.paths."""
        paths_config = config_data.get('paths', {})
        
        path_access_issues = []
        
        for py_file in src_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Look for potential hardcoded paths
                for path_key, path_value in paths_config.items():
                    if isinstance(path_value, str) and len(path_value) > 3:
                        # Check if path is hardcoded instead of using config.paths
                        pattern = f'["\\\']\\s*{re.escape(path_value)}\\s*["\\\']'
                        if re.search(pattern, content):
                            # Check if it's properly accessed via config
                            if not re.search(f'config\\.paths\\.{path_key}|paths\\.{path_key}', content):
                                path_access_issues.append(f"{py_file.name}: {path_value}")
                                
            except Exception:
                continue
        
        # Allow some hardcoded paths for default values
        assert len(path_access_issues) <= 3, f"Paths not accessed via config: {path_access_issues}"

    def test_backend_configuration_consistency(self, config_data, src_files):
        """Test that model backend is consistently accessed via config."""
        backend = config_data.get('model', {}).get('backend', 'meridian')
        
        backend_issues = []
        
        for py_file in src_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Look for hardcoded backend references
                backend_patterns = [
                    r'["\\\']meridian["\\\']',
                    r'["\\\']pymc["\\\']'
                ]
                
                for pattern in backend_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Check if it's not accessed via config
                        line_start = content.rfind('\n', 0, match.start()) + 1
                        line_end = content.find('\n', match.end())
                        if line_end == -1:
                            line_end = len(content)
                        line_content = content[line_start:line_end]
                        
                        # Skip if it's accessing via config or in a comment
                        if not (re.search(r'config\.model\.backend|backend.*config', line_content) or 
                               line_content.strip().startswith('#')):
                            line_num = content[:match.start()].count('\n') + 1
                            backend_issues.append(f"{py_file.name}:{line_num}")
                            
            except Exception:
                continue
        
        # Allow some hardcoded backend references for validation/defaults
        assert len(backend_issues) <= 8, f"Backend not accessed via config: {backend_issues}"

    def test_logging_configuration_usage(self, config_data, src_files):
        """Test that logging configuration is properly used."""
        logging_config = config_data.get('logging', {})
        
        logging_issues = []
        
        for py_file in src_files:
            if 'logging' in py_file.name or 'log' in py_file.name:
                continue  # Skip logging utility files
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Look for hardcoded log levels
                log_level_pattern = r'level\s*=\s*["\\\'](?:DEBUG|INFO|WARNING|ERROR)["\\\']'
                matches = re.finditer(log_level_pattern, content)
                
                for match in matches:
                    # Check if it's not using config
                    if not re.search(r'config\.logging\.level', content):
                        line_num = content[:match.start()].count('\n') + 1
                        logging_issues.append(f"{py_file.name}:{line_num}")
                        
            except Exception:
                continue
        
        # Allow some hardcoded logging for fallbacks
        assert len(logging_issues) <= 5, f"Logging not configured via config: {logging_issues}"

    def test_optimization_parameters_usage(self, config_data, src_files):
        """Test that optimization parameters are accessed via config."""
        opt_config = config_data.get('optimization', {})
        
        # Key optimization parameters that should come from config
        opt_params = [
            'uncertainty_propagation',
            'uncertainty_samples', 
            'max_iterations',
            'convergence_tolerance'
        ]
        
        missing_usage = []
        
        for param in opt_params:
            param_used = False
            
            for py_file in src_files:
                if 'optimization' not in str(py_file):
                    continue
                    
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    # Check if parameter is accessed via config
                    patterns = [
                        f'config.*{param}',
                        f'opt_config.*{param}',
                        f'_safe_config_get.*{param}'
                    ]
                    
                    if any(re.search(pattern, content) for pattern in patterns):
                        param_used = True
                        break
                        
                except Exception:
                    continue
            
            if not param_used:
                missing_usage.append(param)
        
        assert not missing_usage, f"Optimization parameters not accessed via config: {missing_usage}"

    def test_feature_configuration_usage(self, config_data, src_files):
        """Test that feature engineering config is properly used."""
        features_config = config_data.get('features', {})
        
        # Key feature sections
        feature_sections = ['adstock', 'saturation', 'seasonality', 'baseline']
        
        section_usage = {}
        
        for section in feature_sections:
            section_usage[section] = False
            
            # Find corresponding feature files
            for py_file in src_files:
                if section in py_file.name:
                    try:
                        with open(py_file, 'r') as f:
                            content = f.read()
                        
                        # Check if section config is accessed
                        patterns = [
                            f'config\\.features\\.{section}',
                            f'features\\.{section}',
                            f'{section}_config'
                        ]
                        
                        if any(re.search(pattern, content) for pattern in patterns):
                            section_usage[section] = True
                            break
                            
                    except Exception:
                        continue
        
        unused_sections = [section for section, used in section_usage.items() if not used]
        
        # Allow some unused sections for future features
        assert len(unused_sections) <= 2, f"Feature sections not accessed via config: {unused_sections}"

    def test_validation_thresholds_usage(self, config_data, src_files):
        """Test that validation thresholds are accessed via config."""
        validation_config = config_data.get('validation', {})
        evaluation_config = config_data.get('evaluation', {})
        
        # Common threshold patterns that should be configurable
        threshold_patterns = [
            r'\b0\.8\b',   # 80% thresholds
            r'\b0\.9\b',   # 90% thresholds  
            r'\b0\.95\b',  # 95% thresholds
            r'\b0\.15\b',  # 15% thresholds (MAPE)
        ]
        
        hardcoded_thresholds = []
        
        for py_file in src_files:
            if 'validate' in py_file.name or 'evaluation' in py_file.name:
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    for pattern in threshold_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            # Check if threshold is not accessed via config
                            line_start = content.rfind('\n', 0, match.start()) + 1
                            line_end = content.find('\n', match.end())
                            if line_end == -1:
                                line_end = len(content)
                            line_content = content[line_start:line_end]
                            
                            # Skip if it's from config or in comment
                            if not (re.search(r'config\.|threshold|_config', line_content) or
                                   line_content.strip().startswith('#')):
                                line_num = content[:match.start()].count('\n') + 1
                                hardcoded_thresholds.append(f"{py_file.name}:{line_num}")
                                
                except Exception:
                    continue
        
        # Allow some hardcoded thresholds for mathematical constants
        assert len(hardcoded_thresholds) <= 10, f"Thresholds not from config: {hardcoded_thresholds}"


class ConfigStructureTests:
    """Test the overall structure and completeness of the config."""
    
    @pytest.fixture
    def config_data(self):
        """Load the main config file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "main.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_config_sections_complete(self, config_data):
        """Test that all required config sections exist."""
        required_sections = [
            'project', 'paths', 'data', 'validation', 'enhanced_cleaning',
            'features', 'model', 'training', 'complexity', 'optimization', 
            'evaluation', 'reports', 'orchestration', 'logging', 'tracking',
            'runtime', 'profiles'
        ]
        
        missing_sections = [section for section in required_sections 
                          if section not in config_data]
        
        assert not missing_sections, f"Missing config sections: {missing_sections}"
    
    def test_data_section_completeness(self, config_data):
        """Test that data section has all required fields."""
        data_config = config_data.get('data', {})
        
        required_fields = [
            'frequency', 'timezone', 'outcome', 'brands', 'regions',
            'keys', 'brand_key', 'region_key', 'date_col', 'revenue_col',
            'volume_col', 'channel_map'
        ]
        
        missing_fields = [field for field in required_fields 
                         if field not in data_config]
        
        assert not missing_fields, f"Missing data config fields: {missing_fields}"
    
    def test_channel_map_completeness(self, config_data):
        """Test that channel_map has reasonable channels."""
        channel_map = config_data.get('data', {}).get('channel_map', {})
        
        # Should have at least some major channels
        expected_channel_types = ['google', 'meta', 'search', 'social']
        
        found_types = []
        for channel_name in channel_map.keys():
            for channel_type in expected_channel_types:
                if channel_type in channel_name.lower():
                    found_types.append(channel_type)
                    break
        
        assert len(found_types) >= 2, f"Channel map should include major channel types: {found_types}"
        assert len(channel_map) >= 5, f"Channel map should have at least 5 channels, found {len(channel_map)}"
    
    def test_optimization_scenarios_complete(self, config_data):
        """Test that optimization scenarios are properly defined."""
        opt_config = config_data.get('optimization', {})
        scenario_presets = opt_config.get('scenario_presets', {})
        
        # Should have basic scenarios
        expected_scenarios = ['conservative', 'current', 'aggressive']
        
        missing_scenarios = [scenario for scenario in expected_scenarios 
                           if scenario not in scenario_presets]
        
        assert not missing_scenarios, f"Missing optimization scenarios: {missing_scenarios}"
        
        # Each scenario should have required fields
        for scenario_name, scenario_config in scenario_presets.items():
            if isinstance(scenario_config, dict):
                assert 'budget_multiplier' in scenario_config, f"Scenario {scenario_name} missing budget_multiplier"
                assert isinstance(scenario_config['budget_multiplier'], (int, float)), f"Invalid budget_multiplier type in {scenario_name}"
    
    def test_model_backend_configuration(self, config_data):
        """Test that model backend configuration is complete."""
        model_config = config_data.get('model', {})
        
        assert 'backend' in model_config, "Model backend not specified"
        assert model_config['backend'] in ['meridian', 'pymc'], f"Invalid backend: {model_config['backend']}"
        
        # Check backend-specific params exist
        backend_params = model_config.get('backend_params', {})
        current_backend = model_config['backend']
        
        assert current_backend in backend_params, f"Backend params missing for {current_backend}"
        
        # Check required params for current backend
        current_params = backend_params[current_backend]
        required_params = ['draws', 'chains', 'seed']
        
        missing_params = [param for param in required_params if param not in current_params]
        assert not missing_params, f"Missing {current_backend} params: {missing_params}"


if __name__ == "__main__":
    # Run with pytest when called directly
    pytest.main([__file__, "-v"])
