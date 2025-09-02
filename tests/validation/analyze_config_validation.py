#!/usr/bin/env python3
"""
Configuration Validation Summary

Provides a summary of the config validation test results and
actionable recommendations for improving config consistency.
"""

import subprocess
import sys
from pathlib import Path
import json
import re


def analyze_validation_report():
    """Analyze the generated validation report and provide summary."""
    
    project_root = Path(__file__).parent.parent.parent
    report_path = project_root / "artifacts" / "config_usage_report.md"
    
    if not report_path.exists():
        print("❌ No validation report found. Run config validation tests first.")
        return False
    
    print("🔍 CONFIGURATION VALIDATION ANALYSIS")
    print("=" * 60)
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract summary stats
    summary_match = re.search(r'Total config variables: (\d+)', content)
    used_match = re.search(r'Variables used: (\d+)', content)
    ratio_match = re.search(r'Usage ratio: ([\d.]+)%', content)
    
    if summary_match and used_match and ratio_match:
        total = int(summary_match.group(1))
        used = int(used_match.group(1))
        ratio = float(ratio_match.group(1))
        
        print(f"📊 SUMMARY STATISTICS:")
        print(f"   Total config variables: {total}")
        print(f"   Variables used: {used}")
        print(f"   Usage ratio: {ratio:.1f}%")
        print(f"   Unused variables: {total - used}")
        
        # Grade the configuration
        if ratio >= 95:
            grade = "🏆 EXCELLENT"
        elif ratio >= 90:
            grade = "✅ VERY GOOD"
        elif ratio >= 80:
            grade = "👍 GOOD"
        elif ratio >= 70:
            grade = "⚠️ NEEDS IMPROVEMENT"
        else:
            grade = "❌ POOR"
        
        print(f"   Overall grade: {grade}")
        print()
    
    # Count issues
    unused_section = content.find("## Unused Config Variables")
    hardcoded_section = content.find("## Potential Hardcoded Values")
    
    if unused_section >= 0:
        unused_content = content[unused_section:hardcoded_section if hardcoded_section >= 0 else len(content)]
        unused_count = len(re.findall(r'^❌', unused_content, re.MULTILINE))
        print(f"⚠️ UNUSED VARIABLES: {unused_count}")
        
        # Show first few unused variables
        unused_vars = re.findall(r'^❌ (.+)$', unused_content, re.MULTILINE)
        if unused_vars:
            print("   Examples:")
            for var in unused_vars[:5]:
                print(f"     • {var}")
            if len(unused_vars) > 5:
                print(f"     ... and {len(unused_vars) - 5} more")
        print()
    
    if hardcoded_section >= 0:
        hardcoded_content = content[hardcoded_section:]
        
        # Count by file
        file_sections = re.findall(r'^📁 (.+)$', hardcoded_content, re.MULTILINE)
        issue_counts = {}
        
        for section in file_sections:
            file_content = hardcoded_content[hardcoded_content.find(f"📁 {section}"):]
            next_file = hardcoded_content.find("📁", hardcoded_content.find(f"📁 {section}") + 1)
            if next_file >= 0:
                file_content = file_content[:next_file - hardcoded_content.find(f"📁 {section}")]
            
            issues = len(re.findall(r'^\s*⚠️', file_content, re.MULTILINE))
            if issues > 0:
                issue_counts[section] = issues
        
        if issue_counts:
            total_hardcoded = sum(issue_counts.values())
            print(f"🚨 HARDCODED VALUES: {total_hardcoded}")
            print("   Files with most issues:")
            for file_path, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                short_path = file_path.replace("src/mmm/", "")
                print(f"     • {short_path}: {count} issues")
            print()
    
    return True


def provide_recommendations():
    """Provide actionable recommendations based on test results."""
    
    print("🎯 ACTIONABLE RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = [
        {
            "priority": "HIGH",
            "emoji": "🔥",
            "title": "Fix Hardcoded Backend References",
            "description": "Replace hardcoded 'meridian'/'pymc' strings with config.model.backend",
            "action": "Search for backend strings and replace with config access"
        },
        {
            "priority": "HIGH", 
            "emoji": "📁",
            "title": "Use Config for File Paths",
            "description": "Replace hardcoded paths like 'data/features' with config.paths.*",
            "action": "Update path references to use config.paths.features, etc."
        },
        {
            "priority": "MEDIUM",
            "emoji": "🧹",
            "title": "Remove Unused Config Variables",
            "description": "Clean up config variables that aren't used anywhere",
            "action": "Review unused variables and either use them or remove them"
        },
        {
            "priority": "MEDIUM",
            "emoji": "🔧",
            "title": "Standardize Config Access Patterns", 
            "description": "Use consistent _safe_config_get() patterns across files",
            "action": "Implement safe config access helpers in all modules"
        },
        {
            "priority": "LOW",
            "emoji": "📋",
            "title": "Add Missing Config Variables",
            "description": "Some critical variables like data.channel_map detected as missing",
            "action": "Verify config structure has all required sections"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['emoji']} {rec['title']} ({rec['priority']} Priority)")
        print(f"   Description: {rec['description']}")
        print(f"   Action: {rec['action']}")
        print()
    
    print("🛠️ QUICK FIXES")
    print("-" * 30)
    print("1. Run: grep -r '\"meridian\"\\|\"pymc\"' src/mmm/ --include='*.py'")
    print("2. Replace with: config.model.backend")
    print("3. Run: grep -r '\"data/' src/mmm/ --include='*.py'") 
    print("4. Replace with: config.paths.*")
    print("5. Re-run validation: python scripts/validate_config_usage.py")
    print()


def main():
    """Main entry point."""
    
    # Run validation analysis
    if not analyze_validation_report():
        return 1
    
    # Provide recommendations
    provide_recommendations()
    
    print("✨ NEXT STEPS")
    print("=" * 60)
    print("1. 📖 Review the full report: artifacts/config_usage_report.md")
    print("2. 🔧 Fix high-priority issues first")
    print("3. 🧪 Re-run tests: python scripts/validate_config_usage.py")
    print("4. 🎯 Aim for >95% config usage ratio")
    print("5. 🔄 Integrate validation into CI/CD pipeline")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
