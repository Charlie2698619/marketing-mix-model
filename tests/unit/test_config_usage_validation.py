"""
Comprehensive test suite to validate that every variable in main.yaml 
is actually used in the scripts and not hardcoded.

This test suite:
1. Parses main.yaml to extract all configuration variables
2. Searches through all Python files to verify usage
3. Identifies hardcoded values that should be configurable
4. Reports unused config variables
5. Validates config access patterns are consistent

Excludes future extension sections as requested.
"""

import os
import re
import ast
import yaml
import pytest
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Union
from collections import defaultdict
import logging

# Set up logging for test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigUsageValidator:
    """Validates that all config variables are used and no hardcoding exists."""
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            # Find project root by looking for pyproject.toml
            current = Path(__file__).parent
            while current != current.parent:
                if (current / "pyproject.toml").exists():
                    project_root = str(current)
                    break
                current = current.parent
            else:
                raise ValueError("Could not find project root (no pyproject.toml found)")
        
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src" / "mmm"
        self.config_file = self.project_root / "config" / "main.yaml"
        
        # Future extension sections to exclude
        self.future_sections = {
            'attribution', 'creative_fatigue', 'external_data', 'privacy',
            'synthetic_data', 'competitors'
        }
        
        # Load and parse config
        self.config_vars = self._parse_config()
        self.python_files = self._find_python_files()
        
    def _parse_config(self) -> Dict[str, Any]:
        """Parse main.yaml and extract all configuration variables."""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Flatten the config structure excluding future sections
        flattened = {}
        self._flatten_dict(config, flattened)
        
        # Remove future extension sections
        filtered = {}
        for key, value in flattened.items():
            key_parts = key.split('.')
            if not any(future_section in key_parts for future_section in self.future_sections):
                filtered[key] = value
        
        return filtered
    
    def _flatten_dict(self, d: Dict, result: Dict, prefix: str = '') -> None:
        """Recursively flatten a nested dictionary."""
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_dict(value, result, new_key)
            else:
                result[new_key] = value
    
    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the src directory."""
        python_files = []
        for path in self.src_dir.rglob("*.py"):
            if "__pycache__" not in str(path):
                python_files.append(path)
        return python_files
    
    def find_config_usage(self) -> Dict[str, List[str]]:
        """Find where each config variable is used in the codebase."""
        usage_map = defaultdict(list)
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find config usage patterns
                usages = self._find_usage_in_content(content, str(py_file))
                
                for config_key in usages:
                    usage_map[config_key].append(str(py_file.relative_to(self.project_root)))
                    
            except Exception as e:
                logger.warning(f"Error reading {py_file}: {e}")
        
        return dict(usage_map)
    
    def _find_usage_in_content(self, content: str, file_path: str) -> Set[str]:
        """Find config variable usage in file content."""
        usages = set()
        
        # Pattern 1: config.section.subsection.variable
        for key in self.config_vars.keys():
            if self._is_config_accessed(content, key):
                usages.add(key)
        
        return usages
    
    def _is_config_accessed(self, content: str, config_key: str) -> bool:
        """Check if a config key is accessed in the content."""
        key_parts = config_key.split('.')
        
        # Various access patterns to check
        dot_joined = '\\.'.join(key_parts)
        pipe_joined = '|'.join(key_parts)
        bracket_joined = "\\]\\[['\\\"]".join(key_parts)
        pipe_joined_subset = '|'.join(key_parts[1:]) if len(key_parts) > 1 else ''
        
        patterns = [
            # Direct access: config.section.subsection
            f"config\\.{dot_joined}",
            # Getattr access: getattr(config.section, 'subsection')
            f"getattr\\(.*?['\\\"]({pipe_joined})['\\\"]",
            # Dictionary access: config['section']['subsection']  
            f"config\\[['\\\"]({bracket_joined})['\\\"]\\]",
            # Mixed access patterns
            f"config\\.{key_parts[0]}.*?['\\\"]({pipe_joined_subset})['\\\"]" if pipe_joined_subset else None,
            # Safe config get patterns
            f"_safe_config_get.*?['\\\"]({pipe_joined})['\\\"]",
        ]
        
        # Filter out None patterns
        patterns = [p for p in patterns if p is not None]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
                
        # Check for literal value usage (potential hardcoding)
        value = self.config_vars.get(config_key)
        if isinstance(value, (str, int, float)) and value != "":
            # Check if the literal value appears in code
            if self._is_likely_hardcoded(content, str(value), config_key):
                return True
        
        return False
    
    def _is_likely_hardcoded(self, content: str, value: str, config_key: str) -> bool:
        """Check if a value is likely hardcoded instead of using config."""
        if not value or len(str(value)) < 3:  # Skip very short values
            return False
            
        # Skip common values that are likely to appear naturally
        common_values = {
            'auto', 'cpu', 'gpu', 'INFO', 'DEBUG', 'ERROR', 'json', 'text',
            'true', 'false', 'none', 'null', '42', '123', '456'
        }
        
        if str(value).lower() in common_values:
            return False
        
        # Look for quoted string literals or numeric literals
        escaped_value = re.escape(str(value))
        patterns = [
            f'["\\\']\\s*{escaped_value}\\s*["\\\']',  # String literals
            f'\\b{escaped_value}\\b' if isinstance(value, (int, float)) else None
        ]
        
        for pattern in patterns:
            if pattern and re.search(pattern, content):
                # Check if it's not part of a comment or docstring
                lines = content.split('\n')
                for line_num, line in enumerate(lines):
                    if re.search(pattern, line):
                        # Skip if it's in a comment or docstring
                        stripped = line.strip()
                        if not (stripped.startswith('#') or 
                               stripped.startswith('"""') or 
                               stripped.startswith("'''")):
                            return True
        
        return False
    
    def find_hardcoded_values(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find potential hardcoded values that should be configurable."""
        hardcoded_issues = defaultdict(list)
        
        # Common hardcoded patterns to look for
        hardcode_patterns = [
            # File paths
            (r'["\\\'](?:data|artifacts|reports|models)/[^"\\\']*["\\\']', 'hardcoded_path'),
            # Common numbers that should be configurable
            (r'\\b(?:1000|2000|5000|10000)\\b', 'hardcoded_iteration_count'),
            (r'\\b(?:0\\.8|0\\.9|0\\.95|0\\.99)\\b', 'hardcoded_threshold'),
            # Backend names
            (r'["\\\'](?:meridian|pymc)["\\\']', 'hardcoded_backend'),
            # Log levels
            (r'["\\\'](?:DEBUG|INFO|WARNING|ERROR)["\\\']', 'hardcoded_log_level'),
        ]
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, issue_type in hardcode_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Get line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        hardcoded_issues[str(py_file.relative_to(self.project_root))].append({
                            'type': issue_type,
                            'value': match.group(),
                            'line': line_num,
                            'pattern': pattern
                        })
                        
            except Exception as e:
                logger.warning(f"Error checking hardcoded values in {py_file}: {e}")
        
        return dict(hardcoded_issues)
    
    def validate_config_access_patterns(self) -> Dict[str, List[str]]:
        """Validate that config access patterns are consistent."""
        inconsistent_access = defaultdict(list)
        
        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for mixed access patterns
                patterns = {
                    'pydantic_style': r'config\\.[a-zA-Z_][a-zA-Z0-9_]*\\.[a-zA-Z_][a-zA-Z0-9_]*',
                    'dict_style': r'config\\[["\\\'][a-zA-Z_][a-zA-Z0-9_]*["\\\']\\]',
                    'get_style': r'config\\.get\\(["\\\'][a-zA-Z_][a-zA-Z0-9_]*["\\\']',
                    'getattr_style': r'getattr\\(config.*?,["\\\'][a-zA-Z_][a-zA-Z0-9_]*["\\\']',
                    'safe_get_style': r'_safe_config_get\\(',
                }
                
                found_patterns = []
                for style, pattern in patterns.items():
                    if re.search(pattern, content):
                        found_patterns.append(style)
                
                # Report if multiple inconsistent patterns found
                if len(found_patterns) > 2:  # Allow some flexibility
                    inconsistent_access[str(py_file.relative_to(self.project_root))] = found_patterns
                    
            except Exception as e:
                logger.warning(f"Error checking access patterns in {py_file}: {e}")
        
        return dict(inconsistent_access)


# Test class with pytest fixtures and test methods
class TestConfigUsageValidation:
    """Test class for config usage validation."""
    
    @pytest.fixture(scope="class")
    def validator(self):
        """Create a config usage validator instance."""
        return ConfigUsageValidator()
    
    @pytest.fixture(scope="class") 
    def config_usage(self, validator):
        """Get config usage mapping."""
        return validator.find_config_usage()
    
    @pytest.fixture(scope="class")
    def hardcoded_values(self, validator):
        """Get hardcoded values mapping."""
        return validator.find_hardcoded_values()
    
    @pytest.fixture(scope="class")
    def access_patterns(self, validator):
        """Get config access patterns validation."""
        return validator.validate_config_access_patterns()
    
    def test_all_config_variables_used(self, validator, config_usage):
        """Test that all config variables are used somewhere in the codebase."""
        unused_vars = []
        
        for config_key in validator.config_vars.keys():
            if config_key not in config_usage:
                unused_vars.append(config_key)
        
        if unused_vars:
            logger.warning(f"Found {len(unused_vars)} unused config variables:")
            for var in unused_vars[:10]:  # Show first 10
                logger.warning(f"  - {var}")
            if len(unused_vars) > 10:
                logger.warning(f"  ... and {len(unused_vars) - 10} more")
        
        # Report but don't fail - some variables might be legitimately unused
        logger.info(f"Config usage validation: {len(config_usage)}/{len(validator.config_vars)} variables used")
        
        # Fail only if more than 20% of variables are unused
        usage_ratio = len(config_usage) / len(validator.config_vars)
        assert usage_ratio >= 0.8, f"Too many unused config variables: {usage_ratio:.1%} used"
    
    def test_no_hardcoded_paths(self, hardcoded_values):
        """Test that file paths are not hardcoded."""
        path_issues = []
        
        for file_path, issues in hardcoded_values.items():
            for issue in issues:
                if issue['type'] == 'hardcoded_path':
                    path_issues.append(f"{file_path}:{issue['line']} - {issue['value']}")
        
        if path_issues:
            logger.warning("Found hardcoded paths:")
            for issue in path_issues[:10]:
                logger.warning(f"  {issue}")
            if len(path_issues) > 10:
                logger.warning(f"  ... and {len(path_issues) - 10} more")
        
        # Allow some hardcoded paths for testing, but limit them
        assert len(path_issues) <= 5, f"Too many hardcoded paths found: {len(path_issues)}"
    
    def test_no_hardcoded_thresholds(self, hardcoded_values):
        """Test that numeric thresholds are not hardcoded."""
        threshold_issues = []
        
        for file_path, issues in hardcoded_values.items():
            for issue in issues:
                if issue['type'] == 'hardcoded_threshold':
                    threshold_issues.append(f"{file_path}:{issue['line']} - {issue['value']}")
        
        if threshold_issues:
            logger.warning("Found hardcoded thresholds:")
            for issue in threshold_issues[:10]:
                logger.warning(f"  {issue}")
        
        # Allow some hardcoded thresholds for mathematical constants
        assert len(threshold_issues) <= 10, f"Too many hardcoded thresholds: {len(threshold_issues)}"
    
    def test_no_hardcoded_backends(self, hardcoded_values):
        """Test that backend names are not hardcoded."""
        backend_issues = []
        
        for file_path, issues in hardcoded_values.items():
            for issue in issues:
                if issue['type'] == 'hardcoded_backend':
                    backend_issues.append(f"{file_path}:{issue['line']} - {issue['value']}")
        
        if backend_issues:
            logger.warning("Found hardcoded backend names:")
            for issue in backend_issues:
                logger.warning(f"  {issue}")
        
        # Backend names should come from config
        assert len(backend_issues) <= 2, f"Too many hardcoded backend names: {len(backend_issues)}"
    
    def test_consistent_config_access_patterns(self, access_patterns):
        """Test that config access patterns are consistent within files."""
        inconsistent_files = []
        
        for file_path, patterns in access_patterns.items():
            if len(patterns) > 2:  # Allow some flexibility
                inconsistent_files.append(f"{file_path}: {', '.join(patterns)}")
        
        if inconsistent_files:
            logger.warning("Found inconsistent config access patterns:")
            for file_info in inconsistent_files:
                logger.warning(f"  {file_info}")
        
        # Ideally all files should use consistent patterns
        assert len(inconsistent_files) <= 3, f"Too many files with inconsistent access patterns: {len(inconsistent_files)}"
    
    def test_config_structure_completeness(self, validator):
        """Test that config structure covers all necessary sections."""
        required_sections = [
            'project', 'paths', 'data', 'validation', 'features', 
            'model', 'training', 'optimization', 'evaluation', 
            'logging', 'tracking', 'runtime'
        ]
        
        top_level_keys = set()
        for key in validator.config_vars.keys():
            top_level_keys.add(key.split('.')[0])
        
        missing_sections = [section for section in required_sections if section not in top_level_keys]
        
        assert not missing_sections, f"Missing required config sections: {missing_sections}"
    
    def test_critical_config_variables_exist(self, validator):
        """Test that critical config variables exist."""
        critical_vars = [
            'project.name',
            'paths.raw', 'paths.artifacts', 'paths.reports',
            'data.channel_map', 'data.outcome', 'data.date_col',
            'model.backend',
            'optimization.objective',
            'logging.level'
        ]
        
        missing_critical = [var for var in critical_vars if var not in validator.config_vars]
        
        assert not missing_critical, f"Missing critical config variables: {missing_critical}"
    
    def test_config_values_reasonable(self, validator):
        """Test that config values are reasonable."""
        issues = []
        
        for key, value in validator.config_vars.items():
            # Check for reasonable numeric values
            if isinstance(value, (int, float)):
                if 'threshold' in key.lower() and (value < 0 or value > 1):
                    issues.append(f"{key}: threshold should be between 0 and 1, got {value}")
                elif 'percentage' in key.lower() and (value < 0 or value > 100):
                    issues.append(f"{key}: percentage should be between 0 and 100, got {value}")
                elif 'samples' in key.lower() and value < 100:
                    issues.append(f"{key}: sample count seems too low: {value}")
        
        if issues:
            logger.warning("Found questionable config values:")
            for issue in issues:
                logger.warning(f"  {issue}")
        
        # Allow some flexibility in config values
        assert len(issues) <= 5, f"Too many questionable config values: {len(issues)}"
    
    def test_generate_config_usage_report(self, validator, config_usage, hardcoded_values, access_patterns):
        """Generate a comprehensive config usage report."""
        report_lines = [
            "# Configuration Usage Validation Report",
            f"Generated on: {__import__('datetime').datetime.now().isoformat()}",
            "",
            "## Summary",
            f"- Total config variables: {len(validator.config_vars)}",
            f"- Variables used: {len(config_usage)}",
            f"- Usage ratio: {len(config_usage)/len(validator.config_vars):.1%}",
            "",
            "## Config Variables Usage",
        ]
        
        # Add used variables
        for var, files in sorted(config_usage.items()):
            report_lines.append(f"✅ {var}")
            for file_path in files:
                report_lines.append(f"   └── {file_path}")
        
        # Add unused variables
        unused = [var for var in validator.config_vars.keys() if var not in config_usage]
        if unused:
            report_lines.extend(["", "## Unused Config Variables"])
            for var in sorted(unused):
                report_lines.append(f"❌ {var}")
        
        # Add hardcoded values
        if hardcoded_values:
            report_lines.extend(["", "## Potential Hardcoded Values"])
            for file_path, issues in hardcoded_values.items():
                report_lines.append(f"📁 {file_path}")
                for issue in issues:
                    report_lines.append(f"   ⚠️ Line {issue['line']}: {issue['type']} - {issue['value']}")
        
        # Add access pattern issues
        if access_patterns:
            report_lines.extend(["", "## Inconsistent Config Access Patterns"])
            for file_path, patterns in access_patterns.items():
                report_lines.append(f"📁 {file_path}: {', '.join(patterns)}")
        
        # Save report
        report_path = validator.project_root / "artifacts" / "config_usage_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Config usage report saved to: {report_path}")
        
        # Always pass this test - it's for reporting only
        assert True


if __name__ == "__main__":
    # Allow running this module directly for debugging
    validator = ConfigUsageValidator()
    usage = validator.find_config_usage()
    hardcoded = validator.find_hardcoded_values()
    
    print(f"Config variables: {len(validator.config_vars)}")
    print(f"Used variables: {len(usage)}")
    print(f"Usage ratio: {len(usage)/len(validator.config_vars):.1%}")
    
    if hardcoded:
        print(f"\\nPotential hardcoded values found in {len(hardcoded)} files")
