#!/usr/bin/env python3
"""
Configuration Validation Runner

Runs comprehensive config validation tests and generates detailed reports.
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse


def run_config_validation_tests(verbose=False, generate_report=True):
    """Run all config validation tests."""
    
    print("🔍 Starting Configuration Validation Tests...")
    print("=" * 60)
    
    # Find project root
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            project_root = current
            break
        current = current.parent
    else:
        print("❌ Could not find project root (no pyproject.toml found)")
        return False
    
    print(f"📁 Project root: {project_root}")
    
    # Test files to run
    test_files = [
        "tests/unit/test_config_usage_validation.py",
        "tests/unit/test_config_specific_validation.py"
    ]
    
    # Check test files exist
    missing_tests = []
    for test_file in test_files:
        if not (project_root / test_file).exists():
            missing_tests.append(test_file)
    
    if missing_tests:
        print(f"❌ Missing test files: {missing_tests}")
        return False
    
    # Run tests
    all_passed = True
    results = {}
    
    for test_file in test_files:
        print(f"\n🧪 Running {test_file}...")
        print("-" * 40)
        
        cmd = [
            sys.executable, "-m", "pytest", 
            str(project_root / test_file),
            "-v" if verbose else "-q",
            "--tb=short",
            "--color=yes"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            results[test_file] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
                if verbose:
                    print(result.stdout)
            else:
                print(f"❌ {test_file} - FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                all_passed = False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} - TIMEOUT")
            all_passed = False
        except Exception as e:
            print(f"💥 {test_file} - ERROR: {e}")
            all_passed = False
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for r in results.values() if r['returncode'] == 0)
    total_tests = len(results)
    
    print(f"Tests run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if all_passed:
        print("\n🎉 All configuration validation tests PASSED!")
    else:
        print("\n⚠️  Some configuration validation tests FAILED!")
        print("Review the output above for details.")
    
    # Check for generated reports
    artifacts_dir = project_root / "artifacts"
    if artifacts_dir.exists():
        reports = list(artifacts_dir.glob("*config*report*"))
        if reports:
            print(f"\n📄 Generated reports:")
            for report in reports:
                print(f"   📋 {report}")
    
    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive config validation tests"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose test output"
    )
    parser.add_argument(
        "--no-report",
        action="store_true", 
        help="Skip generating detailed reports"
    )
    
    args = parser.parse_args()
    
    success = run_config_validation_tests(
        verbose=args.verbose,
        generate_report=not args.no_report
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
