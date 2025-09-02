#!/usr/bin/env python3
"""
Pipeline Testing Launcher

Convenience script to run various tests from the project root.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Main entry point for running tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MMM Pipeline Tests")
    parser.add_argument("test_type", choices=[
        "pipeline", "config", "unit", "integration", "all"
    ], help="Type of test to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--profile", default="local", help="Config profile for pipeline tests")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    
    if args.test_type == "pipeline":
        # Run end-to-end pipeline test
        cmd = [
            sys.executable, 
            str(project_root / "tests" / "integration" / "test_pipeline.py"),
            "--profile", args.profile
        ]
        if args.verbose:
            cmd.append("--verbose")
            
    elif args.test_type == "config":
        # Run config validation tests
        cmd = [
            sys.executable,
            str(project_root / "tests" / "validation" / "validate_config_usage.py")
        ]
        if args.verbose:
            cmd.append("--verbose")
            
    elif args.test_type == "unit":
        # Run unit tests
        cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v" if args.verbose else "-q"]
        
    elif args.test_type == "integration":
        # Run integration tests
        cmd = [sys.executable, "-m", "pytest", "tests/integration/", "-v" if args.verbose else "-q"]
        
    elif args.test_type == "all":
        # Run all tests
        print("🧪 Running all tests...")
        
        # Config validation
        print("\n1. Running config validation...")
        subprocess.run([
            sys.executable,
            str(project_root / "tests" / "validation" / "validate_config_usage.py")
        ])
        
        # Unit tests
        print("\n2. Running unit tests...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/unit/", "-v" if args.verbose else "-q"])
        
        # Integration tests
        print("\n3. Running integration tests...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/integration/", "-v" if args.verbose else "-q"])
        
        return
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
