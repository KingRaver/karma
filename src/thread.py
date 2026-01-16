#!/usr/bin/env python3
"""
🔍 NUMBA THREAD CONFLICT DIAGNOSTIC TOOL 🔍
===========================================

Comprehensive diagnostic tool to identify root cause of NUMBA thread conflicts.
Designed to work with numba_thread_manager.py global system.

This tool will:
1. Scan all Python files for thread configuration references
2. Identify where NUMBA_NUM_THREADS=10 is being set
3. Find hardcoded thread counts in calculations
4. Detect runtime thread configuration conflicts
5. Generate detailed report for industry-standard fix

Author: Thread Conflict Resolution System
Version: 1.0 - Root Cause Analysis Edition
"""

import os
import re
import ast
import sys
import time
import inspect
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
import logging

class NumbaThreadDiagnostic:
    """🔍 Comprehensive NUMBA thread conflict diagnostic system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.scan_results = {
            'thread_count_references': [],
            'numba_imports': [],
            'environment_settings': [],
            'hardcoded_values': [],
            'calculation_conflicts': [],
            'runtime_conflicts': [],
            'file_analysis': {}
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Setup diagnostic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | 🔍 DIAGNOSTIC | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger('NumbaThreadDiagnostic')
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """🎯 Run comprehensive diagnostic scan"""
        self.logger.info("🚀 Starting NUMBA thread conflict diagnostic...")
        start_time = time.time()
        
        try:
            # 1. Scan all Python files
            self.scan_python_files()
            
            # 2. Check environment variables
            self.check_environment_variables()
            
            # 3. Analyze runtime imports
            self.analyze_runtime_imports()
            
            # 4. Check for calculation-specific conflicts
            self.check_calculation_conflicts()
            
            # 5. Detect hardcoded thread values
            self.detect_hardcoded_threads()
            
            # 6. Generate comprehensive report
            report = self.generate_diagnostic_report()
            
            duration = time.time() - start_time
            self.logger.info(f"✅ Diagnostic completed in {duration:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Diagnostic failed: {e}")
            return {'error': str(e), 'scan_results': self.scan_results}
    
    def scan_python_files(self):
        """📁 Scan all Python files for thread-related code"""
        self.logger.info("📁 Scanning Python files...")
        
        python_files = list(self.project_root.rglob("*.py"))
        self.logger.info(f"Found {len(python_files)} Python files to scan")
        
        for file_path in python_files:
            if self.should_scan_file(file_path):
                self.scan_file(file_path)
    
    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned"""
        # Skip certain directories and files
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            '.env',
            'node_modules',
            'build',
            'dist'
        ]
        
        return not any(pattern in str(file_path) for pattern in skip_patterns)
    
    def scan_file(self, file_path: Path):
        """🔍 Scan individual file for thread-related patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_results = {
                'path': str(file_path),
                'thread_references': [],
                'numba_imports': [],
                'hardcoded_values': [],
                'suspicious_patterns': []
            }
            
            # Search for thread-related patterns
            self.find_thread_patterns(content, file_results)
            self.find_numba_patterns(content, file_results)
            self.find_hardcoded_values(content, file_results)
            self.find_suspicious_patterns(content, file_results)
            
            # Only store results if we found something
            if any(file_results[key] for key in ['thread_references', 'numba_imports', 'hardcoded_values', 'suspicious_patterns']):
                self.scan_results['file_analysis'][str(file_path)] = file_results
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not scan {file_path}: {e}")
    
    def find_thread_patterns(self, content: str, file_results: dict):
        """Find thread-related patterns in file content"""
        patterns = [
            r'NUMBA_NUM_THREADS.*?=.*?(\d+)',
            r'thread_count.*?=.*?(\d+)',
            r'num_threads.*?=.*?(\d+)',
            r'workers.*?=.*?(\d+)',
            r'OPTIMAL_WORKERS.*?=.*?(\d+)',
            r'performance_cores.*?=.*?(\d+)',
            r'max_workers.*?=.*?(\d+)',
            r'ThreadPoolExecutor.*?max_workers.*?=.*?(\d+)',
            r'os\.environ\[.*?THREAD.*?\].*?=.*?["\'](\d+)["\']',
            r'set_num_threads\((\d+)\)',
            r'initialize.*?thread_count.*?=.*?(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                thread_count = match.group(1) if match.groups() else 'unknown'
                
                file_results['thread_references'].append({
                    'pattern': pattern,
                    'match': match.group(0),
                    'thread_count': thread_count,
                    'line_number': line_num,
                    'context': self.get_line_context(content, match.start())
                })
                
                # Flag potential conflicts
                if thread_count == '10':
                    self.scan_results['calculation_conflicts'].append({
                        'file': file_results['path'],
                        'line': line_num,
                        'thread_count': thread_count,
                        'context': match.group(0)
                    })
    
    def find_numba_patterns(self, content: str, file_results: dict):
        """Find NUMBA-related imports and configurations"""
        patterns = [
            r'from numba import.*',
            r'import numba.*',
            r'@njit.*',
            r'@jit.*',
            r'numba\.njit.*',
            r'numba\.jit.*',
            r'numba\.set_num_threads.*',
            r'numba\.config\.THREADING_LAYER.*',
            r'NUMBA_THREADING_LAYER.*',
            r'get_num_threads\(\)',
            r'threading\.active_count\(\)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                file_results['numba_imports'].append({
                    'pattern': pattern,
                    'match': match.group(0),
                    'line_number': line_num,
                    'context': self.get_line_context(content, match.start())
                })
    
    def find_hardcoded_values(self, content: str, file_results: dict):
        """Find hardcoded thread count values"""
        # Look specifically for the number 10 in thread-related contexts
        patterns = [
            r'.*(?:thread|worker|core|cpu).*?(?:=|:|\()\s*10\b',
            r'10\s*(?:thread|worker|core|cpu)',
            r'range\(10\)',
            r'ThreadPoolExecutor.*?10',
            r'max_workers\s*=\s*10',
            r'thread_count\s*=\s*10',
            r'num_threads\s*=\s*10'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                file_results['hardcoded_values'].append({
                    'pattern': pattern,
                    'match': match.group(0),
                    'line_number': line_num,
                    'context': self.get_line_context(content, match.start(), context_lines=2),
                    'suspicious_level': 'HIGH' if '10' in match.group(0) else 'MEDIUM'
                })
                
                self.scan_results['hardcoded_values'].append({
                    'file': file_results['path'],
                    'line': line_num,
                    'value': match.group(0),
                    'context': self.get_line_context(content, match.start())
                })
    
    def find_suspicious_patterns(self, content: str, file_results: dict):
        """Find suspicious patterns that might cause conflicts"""
        patterns = [
            r'os\.environ.*NUMBA.*10',
            r'multiprocessing\.cpu_count\(\)',
            r'threading\.active_count\(\)',
            r'psutil\.cpu_count\(\)',
            r'get_optimal_worker_count\(\)',
            r'performance_cores.*efficiency_cores',
            r'trying to set \d+',
            r'currently have \d+',
            r'Cannot set NUMBA_NUM_THREADS'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                file_results['suspicious_patterns'].append({
                    'pattern': pattern,
                    'match': match.group(0),
                    'line_number': line_num,
                    'context': self.get_line_context(content, match.start()),
                    'risk_level': self.assess_risk_level(match.group(0))
                })
    
    def get_line_context(self, content: str, position: int, context_lines: int = 1) -> List[str]:
        """Get surrounding lines for context"""
        lines = content.split('\n')
        line_num = content[:position].count('\n')
        
        start_line = max(0, line_num - context_lines)
        end_line = min(len(lines), line_num + context_lines + 1)
        
        context = []
        for i in range(start_line, end_line):
            prefix = ">>>" if i == line_num else "   "
            context.append(f"{prefix} {i+1:3d}: {lines[i]}")
        
        return context
    
    def assess_risk_level(self, pattern: str) -> str:
        """Assess risk level of suspicious pattern"""
        if 'Cannot set NUMBA_NUM_THREADS' in pattern:
            return 'CRITICAL'
        elif '10' in pattern and 'thread' in pattern.lower():
            return 'HIGH'
        elif 'cpu_count' in pattern.lower():
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def check_environment_variables(self):
        """🌍 Check environment variables for thread settings"""
        self.logger.info("🌍 Checking environment variables...")
        
        env_vars = [
            'NUMBA_NUM_THREADS',
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS'
        ]
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                self.scan_results['environment_settings'].append({
                    'variable': var,
                    'value': value,
                    'conflict_risk': 'HIGH' if value == '10' and var == 'NUMBA_NUM_THREADS' else 'MEDIUM'
                })
    
    def analyze_runtime_imports(self):
        """🔍 Analyze runtime imports for NUMBA configuration"""
        self.logger.info("🔍 Analyzing runtime imports...")
        
        try:
            # Check if NUMBA is already imported
            if 'numba' in sys.modules:
                numba = sys.modules['numba']
                self.scan_results['runtime_conflicts'].append({
                    'type': 'NUMBA_ALREADY_IMPORTED',
                    'module': 'numba',
                    'status': 'LOADED',
                    'risk': 'HIGH'
                })
                
                # Try to get thread count if possible
                try:
                    if hasattr(numba, 'get_num_threads'):
                        current_threads = numba.get_num_threads()
                        self.scan_results['runtime_conflicts'].append({
                            'type': 'CURRENT_THREAD_COUNT',
                            'value': current_threads,
                            'risk': 'HIGH' if current_threads != 8 else 'LOW'
                        })
                except:
                    pass
        except Exception as e:
            self.logger.warning(f"Could not analyze runtime imports: {e}")
    
    def check_calculation_conflicts(self):
        """🧮 Check specific calculation methods for thread conflicts"""
        self.logger.info("🧮 Checking calculation conflicts...")
        
        # Target files that might have calculation conflicts
        target_files = [
            'technical_calculations.py',
            'prediction_engine.py',
            'technical_indicators.py',
            'macd.py',
            'bollinger.py',
            'rsi.py'
        ]
        
        for target_file in target_files:
            file_path = self.project_root / target_file
            if file_path.exists():
                self.check_calculation_file(file_path)
    
    def check_calculation_file(self, file_path: Path):
        """Check specific calculation file for conflicts"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for specific conflict patterns in calculation files
            conflict_patterns = [
                r'def.*(?:calculate_macd|calculate_rsi|calculate_bollinger).*:.*?njit.*?10',
                r'@njit.*\n.*def.*(?:macd|rsi|bollinger)',
                r'ThreadPoolExecutor.*10.*(?:macd|rsi|bollinger)',
                r'NUMBA_NUM_THREADS.*10.*(?:calculation|indicator)'
            ]
            
            for pattern in conflict_patterns:
                matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.scan_results['calculation_conflicts'].append({
                        'file': str(file_path),
                        'line': line_num,
                        'type': 'CALCULATION_THREAD_CONFLICT',
                        'pattern': pattern,
                        'match': match.group(0)[:100] + '...' if len(match.group(0)) > 100 else match.group(0),
                        'risk': 'CRITICAL'
                    })
                    
        except Exception as e:
            self.logger.warning(f"Could not check calculation file {file_path}: {e}")
    
    def detect_hardcoded_threads(self):
        """🔢 Detect hardcoded thread values of 10"""
        self.logger.info("🔢 Detecting hardcoded thread values...")
        
        # Already handled in scan_file method
        # This method aggregates and analyzes the results
        
        high_risk_files = []
        for file_path, results in self.scan_results['file_analysis'].items():
            thread_10_count = sum(1 for hv in results['hardcoded_values'] 
                                if '10' in hv['match'] and 'thread' in hv['match'].lower())
            
            if thread_10_count > 0:
                high_risk_files.append({
                    'file': file_path,
                    'thread_10_references': thread_10_count,
                    'risk': 'CRITICAL' if thread_10_count > 2 else 'HIGH'
                })
        
        self.scan_results['high_risk_files'] = high_risk_files
    
    def generate_diagnostic_report(self) -> Dict[str, Any]:
        """📋 Generate comprehensive diagnostic report"""
        self.logger.info("📋 Generating diagnostic report...")
        
        # Analyze findings
        total_files_scanned = len(self.scan_results['file_analysis'])
        files_with_issues = len([f for f in self.scan_results['file_analysis'].values() 
                               if any(f[key] for key in f.keys())])
        
        # Count critical issues
        critical_issues = []
        critical_issues.extend([c for c in self.scan_results.get('calculation_conflicts', []) 
                              if c.get('risk') == 'CRITICAL'])
        critical_issues.extend([h for h in self.scan_results.get('hardcoded_values', []) 
                              if '10' in str(h.get('value', ''))])
        
        # Generate summary
        report = {
            'diagnostic_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_files_scanned': total_files_scanned,
                'files_with_issues': files_with_issues,
                'critical_issues_count': len(critical_issues),
                'environment_conflicts': len(self.scan_results.get('environment_settings', [])),
                'runtime_conflicts': len(self.scan_results.get('runtime_conflicts', []))
            },
            'critical_findings': {
                'thread_count_10_references': [
                    issue for issue in critical_issues 
                    if '10' in str(issue)
                ],
                'calculation_conflicts': self.scan_results.get('calculation_conflicts', []),
                'environment_conflicts': [
                    env for env in self.scan_results.get('environment_settings', [])
                    if env.get('value') == '10'
                ]
            },
            'detailed_analysis': self.scan_results,
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[Dict[str, str]]:
        """💡 Generate fix recommendations"""
        recommendations = []
        
        # Check for thread count 10 issues
        if any('10' in str(issue) for issue in self.scan_results.get('hardcoded_values', [])):
            recommendations.append({
                'priority': 'CRITICAL',
                'issue': 'Hardcoded thread count of 10 found',
                'solution': 'Replace all hardcoded thread counts of 10 with OPTIMAL_WORKERS from numba_thread_manager',
                'action': 'Search and replace hardcoded "10" with dynamic thread count from global manager'
            })
        
        # Check for calculation conflicts
        if self.scan_results.get('calculation_conflicts'):
            recommendations.append({
                'priority': 'CRITICAL', 
                'issue': 'MACD/RSI/Bollinger calculations have thread conflicts',
                'solution': 'Integrate calculations with numba_thread_manager.py global system',
                'action': 'Modify calculation methods to use get_global_manager().get_njit()'
            })
        
        # Check for environment conflicts
        if any(env.get('value') == '10' for env in self.scan_results.get('environment_settings', [])):
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Environment variable NUMBA_NUM_THREADS set to 10',
                'solution': 'Set NUMBA_NUM_THREADS=8 to match thread manager',
                'action': 'Update environment configuration or startup scripts'
            })
        
        if not recommendations:
            recommendations.append({
                'priority': 'INFO',
                'issue': 'No critical thread conflicts detected',
                'solution': 'System appears to be configured correctly',
                'action': 'Monitor for runtime conflicts during execution'
            })
        
        return recommendations

    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """💾 Save diagnostic report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"numba_thread_diagnostic_{timestamp}.json"
        
        import json
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"📄 Report saved to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return None

def main():
    """🎯 Main diagnostic execution"""
    print("🔍 NUMBA Thread Conflict Diagnostic Tool")
    print("=" * 50)
    
    # Initialize diagnostic
    diagnostic = NumbaThreadDiagnostic()
    
    # Run full diagnostic
    report = diagnostic.run_full_diagnostic()
    
    # Display summary
    print(f"\n📋 DIAGNOSTIC SUMMARY:")
    print(f"Files Scanned: {report['diagnostic_summary']['total_files_scanned']}")
    print(f"Files with Issues: {report['diagnostic_summary']['files_with_issues']}")
    print(f"Critical Issues: {report['diagnostic_summary']['critical_issues_count']}")
    
    print(f"\n🚨 CRITICAL FINDINGS:")
    critical_findings = report['critical_findings']
    
    if critical_findings['thread_count_10_references']:
        print("❌ Thread Count 10 References Found:")
        for ref in critical_findings['thread_count_10_references'][:5]:  # Show first 5
            print(f"   - {ref.get('file', 'unknown')}:{ref.get('line', '?')}")
    
    if critical_findings['calculation_conflicts']:
        print("❌ Calculation Conflicts Found:")
        for conflict in critical_findings['calculation_conflicts'][:3]:  # Show first 3
            print(f"   - {conflict.get('file', 'unknown')}:{conflict.get('line', '?')} - {conflict.get('type', 'unknown')}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"🔧 [{rec['priority']}] {rec['issue']}")
        print(f"   Solution: {rec['solution']}")
    
    # Save detailed report
    filename = diagnostic.save_report(report)
    if filename:
        print(f"\n📄 Detailed report saved to: {filename}")
    
    return report

if __name__ == "__main__":
    report = main()