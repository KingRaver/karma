#!/usr/bin/env python3
"""
🔍 CPU THREAD COUNT DIAGNOSTIC TOOL 🔍
=====================================

Targeted diagnostic to identify where raw CPU counts are being used
instead of the managed OPTIMAL_WORKERS value from numba_thread_manager.py

This tool specifically hunts for:
1. Direct multiprocessing.cpu_count() usage for threading
2. Raw CPU count references in thread configuration 
3. Hardcoded thread counts that don't match OPTIMAL_WORKERS
4. M4SystemDetector inconsistencies
5. Environment variable conflicts

Author: CPU Thread Management Diagnostic System
Version: 1.0 - Root Cause Hunter Edition
"""

import os
import re
import ast
import sys
import time
import multiprocessing
import platform
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
import logging
import json

class CPUThreadDiagnostic:
    """🔍 Specialized CPU thread count conflict diagnostic"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.system_info = self.get_system_info()
        self.scan_results = {
            'cpu_count_references': [],
            'thread_inconsistencies': [],
            'hardcoded_conflicts': [],
            'environment_issues': [],
            'foundation_analysis': {},
            'critical_locations': []
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Setup diagnostic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | 🔍 CPU DIAG | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger('CPUThreadDiagnostic')
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'cpu_count': multiprocessing.cpu_count(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'system': platform.system(),
            'is_apple_silicon': platform.machine() in ['arm64', 'aarch64'],
            'python_version': sys.version
        }
        
        # Try to get more detailed CPU info
        try:
            if info['system'] == 'Darwin':  # macOS
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['physical_cores'] = int(result.stdout.strip())
                
                result = subprocess.run(['sysctl', '-n', 'hw.logicalcpu'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['logical_cores'] = int(result.stdout.strip())
                    
                # Check for efficiency cores (M1/M2/M3/M4)
                result = subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['performance_cores'] = int(result.stdout.strip())
                    
                result = subprocess.run(['sysctl', '-n', 'hw.perflevel1.logicalcpu'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    info['efficiency_cores'] = int(result.stdout.strip())
                    
        except Exception as e:
            info['cpu_detection_error'] = str(e)
        
        return info
    
    def run_cpu_diagnostic(self) -> Dict[str, Any]:
        """🎯 Run comprehensive CPU thread diagnostic"""
        self.logger.info("🚀 Starting CPU thread count diagnostic...")
        self.logger.info(f"💻 System: {self.system_info['cpu_count']} cores detected")
        
        start_time = time.time()
        
        try:
            # 1. Analyze foundation configuration
            self.analyze_foundation_config()
            
            # 2. Find raw CPU count usage
            self.find_raw_cpu_usage()
            
            # 3. Identify thread inconsistencies
            self.find_thread_inconsistencies()
            
            # 4. Check environment conflicts
            self.check_environment_conflicts()
            
            # 5. Analyze M4SystemDetector logic
            self.analyze_m4_detector()
            
            # 6. Find critical locations
            self.identify_critical_locations()
            
            # 7. Generate targeted report
            report = self.generate_cpu_report()
            
            duration = time.time() - start_time
            self.logger.info(f"✅ CPU diagnostic completed in {duration:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ CPU diagnostic failed: {e}")
            return {'error': str(e), 'scan_results': self.scan_results}
    
    def analyze_foundation_config(self):
        """🏗️ Analyze technical_foundation.py configuration"""
        self.logger.info("🏗️ Analyzing foundation configuration...")
        
        foundation_file = self.project_root / "src" / "technical_foundation.py"
        if not foundation_file.exists():
            foundation_file = self.project_root / "technical_foundation.py"
        
        if foundation_file.exists():
            try:
                with open(foundation_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract OPTIMAL_WORKERS logic
                patterns = [
                    r'OPTIMAL_WORKERS\s*=\s*(.+)',
                    r'get_optimal_worker_count\(\)\s*->\s*int:(.*?)def\s+',
                    r'performance_cores.*?(\d+)',
                    r'cpu_count.*?(\d+)',
                    r'min\(.*?(\d+).*?\)',
                    r'multiprocessing\.cpu_count\(\)'
                ]
                
                foundation_analysis = {
                    'file_path': str(foundation_file),
                    'optimal_workers_definition': [],
                    'cpu_references': [],
                    'detector_logic': [],
                    'potential_conflicts': []
                }
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        context = self.get_line_context(content, match.start())
                        
                        foundation_analysis['optimal_workers_definition'].append({
                            'pattern': pattern,
                            'match': match.group(0)[:200] + ('...' if len(match.group(0)) > 200 else ''),
                            'line': line_num,
                            'context': context
                        })
                        
                        # Check for potential conflicts
                        if 'cpu_count' in match.group(0) and 'multiprocessing' in match.group(0):
                            foundation_analysis['potential_conflicts'].append({
                                'type': 'RAW_CPU_COUNT_USAGE',
                                'line': line_num,
                                'match': match.group(0),
                                'risk': 'HIGH'
                            })
                
                self.scan_results['foundation_analysis'] = foundation_analysis
                
            except Exception as e:
                self.logger.warning(f"Could not analyze foundation: {e}")
    
    def find_raw_cpu_usage(self):
        """🔍 Find raw CPU count usage patterns"""
        self.logger.info("🔍 Scanning for raw CPU count usage...")
        
        # Target patterns that indicate raw CPU usage for threading
        cpu_patterns = [
            r'multiprocessing\.cpu_count\(\)',
            r'os\.cpu_count\(\)',
            r'psutil\.cpu_count\(\)',
            r'threading\.active_count\(\)',
            r'cpu_count\(\)',
            r'ThreadPoolExecutor\s*\(\s*max_workers\s*=\s*(?!OPTIMAL_WORKERS)([^)]+)\)',
            r'max_workers\s*=\s*(?!OPTIMAL_WORKERS)(\d+|[a-zA-Z_]+\.cpu_count\(\))',
            r'thread_count\s*=\s*(?!OPTIMAL_WORKERS)(\d+|[a-zA-Z_]+\.cpu_count\(\))',
            r'num_threads\s*=\s*(?!OPTIMAL_WORKERS)(\d+|[a-zA-Z_]+\.cpu_count\(\))'
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if self.should_scan_file(file_path):
                self.scan_file_for_cpu_patterns(file_path, cpu_patterns)
    
    def scan_file_for_cpu_patterns(self, file_path: Path, patterns: List[str]):
        """Scan individual file for CPU-related patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    context = self.get_line_context(content, match.start())
                    
                    cpu_ref = {
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': pattern,
                        'match': match.group(0),
                        'context': context,
                        'risk_level': self.assess_cpu_risk(match.group(0), str(file_path))
                    }
                    
                    self.scan_results['cpu_count_references'].append(cpu_ref)
                    
                    # Flag high-risk cases
                    if cpu_ref['risk_level'] == 'CRITICAL':
                        self.scan_results['critical_locations'].append(cpu_ref)
                        
        except Exception as e:
            self.logger.warning(f"Could not scan {file_path}: {e}")
    
    def assess_cpu_risk(self, match: str, file_path: str) -> str:
        """Assess risk level of CPU count usage"""
        # Critical files that should use OPTIMAL_WORKERS
        critical_files = [
            'technical_calculations.py',
            'prediction_engine.py', 
            'technical_indicators.py',
            'numba_thread_manager.py',
            'technical_foundation.py'
        ]
        
        # High-risk patterns
        high_risk_patterns = [
            'multiprocessing.cpu_count()',
            'ThreadPoolExecutor',
            'max_workers',
            'thread_count',
            'num_threads'
        ]
        
        if any(critical_file in file_path for critical_file in critical_files):
            if any(pattern in match for pattern in high_risk_patterns):
                return 'CRITICAL'
            else:
                return 'HIGH'
        
        if 'multiprocessing.cpu_count()' in match:
            return 'HIGH'
        elif 'ThreadPoolExecutor' in match and 'max_workers' in match:
            return 'HIGH'
        elif any(pattern in match for pattern in ['thread_count', 'num_threads']):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def find_thread_inconsistencies(self):
        """🔍 Find inconsistencies between different thread counts"""
        self.logger.info("🔍 Analyzing thread count inconsistencies...")
        
        # Look for files that use multiple different thread counts
        file_thread_counts = {}
        
        for cpu_ref in self.scan_results['cpu_count_references']:
            file_path = cpu_ref['file']
            match = cpu_ref['match']
            
            # Extract numeric values
            numbers = re.findall(r'\d+', match)
            
            if file_path not in file_thread_counts:
                file_thread_counts[file_path] = set()
            
            for num in numbers:
                if int(num) > 1 and int(num) <= 64:  # Reasonable thread count range
                    file_thread_counts[file_path].add(int(num))
        
        # Find files with inconsistent thread counts
        for file_path, thread_counts in file_thread_counts.items():
            if len(thread_counts) > 1:
                # Check if one of them is 8 (expected OPTIMAL_WORKERS)
                has_eight = 8 in thread_counts
                has_ten = 10 in thread_counts
                has_cpu_count = self.system_info['cpu_count'] in thread_counts
                
                inconsistency = {
                    'file': file_path,
                    'thread_counts': sorted(list(thread_counts)),
                    'has_optimal_8': has_eight,
                    'has_cpu_count_10': has_ten,
                    'has_system_cpu_count': has_cpu_count,
                    'risk': 'CRITICAL' if (has_eight and (has_ten or has_cpu_count)) else 'HIGH'
                }
                
                self.scan_results['thread_inconsistencies'].append(inconsistency)
    
    def check_environment_conflicts(self):
        """🌍 Check for environment variable conflicts"""
        self.logger.info("🌍 Checking environment conflicts...")
        
        env_vars = [
            'NUMBA_NUM_THREADS',
            'OMP_NUM_THREADS', 
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'NUMEXPR_NUM_THREADS'
        ]
        
        conflicts = []
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                try:
                    thread_count = int(value)
                    conflict = {
                        'variable': var,
                        'value': thread_count,
                        'matches_cpu_count': thread_count == self.system_info['cpu_count'],
                        'matches_optimal_8': thread_count == 8,
                        'risk': 'CRITICAL' if var == 'NUMBA_NUM_THREADS' and thread_count != 8 else 'MEDIUM'
                    }
                    conflicts.append(conflict)
                except ValueError:
                    conflicts.append({
                        'variable': var,
                        'value': value,
                        'error': 'non_numeric_value',
                        'risk': 'MEDIUM'
                    })
        
        self.scan_results['environment_issues'] = conflicts
    
    def analyze_m4_detector(self):
        """🔍 Analyze M4SystemDetector logic specifically"""
        self.logger.info("🔍 Analyzing M4SystemDetector logic...")
        
        detector_patterns = [
            r'def get_optimal_worker_count.*?return.*?(\d+)',
            r'performance_cores.*?=.*?(\d+)',
            r'efficiency_cores.*?=.*?(\d+)', 
            r'cpu_count.*?>=.*?(\d+)',
            r'min\(.*?(\d+).*?\)',
            r'max\(.*?(\d+).*?\)'
        ]
        
        foundation_file = self.project_root / "src" / "technical_foundation.py"
        if not foundation_file.exists():
            foundation_file = self.project_root / "technical_foundation.py"
        
        if foundation_file.exists():
            try:
                with open(foundation_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                detector_analysis = []
                
                # Find the get_optimal_worker_count method
                method_match = re.search(
                    r'def get_optimal_worker_count\(self\)\s*->\s*int:(.*?)(?=def|\Z)',
                    content, re.DOTALL
                )
                
                if method_match:
                    method_body = method_match.group(1)
                    method_start = method_match.start()
                    method_line = content[:method_start].count('\n') + 1
                    
                    # Analyze the logic
                    logic_analysis = {
                        'method_line': method_line,
                        'method_body': method_body[:500] + ('...' if len(method_body) > 500 else ''),
                        'potential_issues': []
                    }
                    
                    # Check for problematic patterns
                    if 'total_cores' in method_body and 'min(' in method_body:
                        # This could be the source of using 10 instead of 8
                        if 'min(total_cores, 6)' in method_body:
                            logic_analysis['potential_issues'].append({
                                'issue': 'FALLBACK_LOGIC_USES_TOTAL_CORES',
                                'description': 'Fallback uses min(total_cores, 6) which could return more than 8',
                                'risk': 'CRITICAL'
                            })
                    
                    if '10' in method_body:
                        logic_analysis['potential_issues'].append({
                            'issue': 'HARDCODED_10_IN_DETECTOR',
                            'description': 'Hardcoded 10 found in detector logic',
                            'risk': 'CRITICAL'
                        })
                    
                    detector_analysis.append(logic_analysis)
                
                self.scan_results['m4_detector_analysis'] = detector_analysis
                
            except Exception as e:
                self.logger.warning(f"Could not analyze M4SystemDetector: {e}")
    
    def identify_critical_locations(self):
        """🎯 Identify the most critical locations for fixes"""
        self.logger.info("🎯 Identifying critical fix locations...")
        
        # Priority files that must use OPTIMAL_WORKERS consistently
        critical_files = [
            'technical_calculations.py',
            'prediction_engine.py',
            'technical_indicators.py', 
            'technical_foundation.py',
            'numba_thread_manager.py'
        ]
        
        for cpu_ref in self.scan_results['cpu_count_references']:
            file_path = cpu_ref['file']
            
            # Check if it's in a critical file
            if any(critical_file in file_path for critical_file in critical_files):
                # Check if it's using raw CPU count instead of OPTIMAL_WORKERS
                if ('multiprocessing.cpu_count()' in cpu_ref['match'] or 
                    'cpu_count()' in cpu_ref['match'] or
                    ('ThreadPoolExecutor' in cpu_ref['match'] and 'OPTIMAL_WORKERS' not in cpu_ref['match'])):
                    
                    critical_location = {
                        **cpu_ref,
                        'fix_priority': 'IMMEDIATE',
                        'recommended_fix': 'Replace with OPTIMAL_WORKERS from technical_foundation',
                        'impact': 'HIGH - Causes NUMBA thread conflicts'
                    }
                    
                    if critical_location not in self.scan_results['critical_locations']:
                        self.scan_results['critical_locations'].append(critical_location)
    
    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'venv',
            '.env',
            'node_modules',
            'build',
            'dist',
            '.pytest_cache'
        ]
        
        return not any(pattern in str(file_path) for pattern in skip_patterns)
    
    def get_line_context(self, content: str, position: int, context_lines: int = 2) -> List[str]:
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
    
    def generate_cpu_report(self) -> Dict[str, Any]:
        """📋 Generate comprehensive CPU diagnostic report"""
        self.logger.info("📋 Generating CPU diagnostic report...")
        
        # Count issues by severity
        critical_count = len([ref for ref in self.scan_results['cpu_count_references'] 
                            if ref.get('risk_level') == 'CRITICAL'])
        high_count = len([ref for ref in self.scan_results['cpu_count_references'] 
                        if ref.get('risk_level') == 'HIGH'])
        
        # Analyze inconsistencies
        inconsistency_count = len(self.scan_results['thread_inconsistencies'])
        critical_inconsistencies = len([inc for inc in self.scan_results['thread_inconsistencies']
                                      if inc.get('risk') == 'CRITICAL'])
        
        report = {
            'cpu_diagnostic_summary': {
                'timestamp': datetime.now().isoformat(),
                'system_cpu_count': self.system_info['cpu_count'],
                'expected_optimal_workers': 8,
                'total_cpu_references': len(self.scan_results['cpu_count_references']),
                'critical_issues': critical_count,
                'high_risk_issues': high_count,
                'thread_inconsistencies': inconsistency_count,
                'critical_inconsistencies': critical_inconsistencies,
                'environment_conflicts': len(self.scan_results['environment_issues']),
                'immediate_fix_locations': len(self.scan_results['critical_locations'])
            },
            'system_information': self.system_info,
            'root_cause_analysis': self.analyze_root_cause(),
            'critical_findings': {
                'immediate_fixes_required': self.scan_results['critical_locations'][:10],
                'thread_inconsistencies': self.scan_results['thread_inconsistencies'],
                'environment_conflicts': [env for env in self.scan_results['environment_issues'] 
                                        if env.get('risk') == 'CRITICAL'],
                'foundation_issues': self.scan_results.get('foundation_analysis', {}),
                'detector_problems': self.scan_results.get('m4_detector_analysis', [])
            },
            'detailed_scan_results': self.scan_results,
            'fix_recommendations': self.generate_fix_recommendations()
        }
        
        return report
    
    def analyze_root_cause(self) -> Dict[str, Any]:
        """🎯 Analyze the root cause of thread conflicts"""
        root_causes = []
        
        # Check if system CPU count is being used directly
        cpu_count_usage = [ref for ref in self.scan_results['cpu_count_references']
                          if 'multiprocessing.cpu_count()' in ref['match'] or 
                             'cpu_count()' in ref['match']]
        
        if cpu_count_usage:
            root_causes.append({
                'cause': 'DIRECT_CPU_COUNT_USAGE',
                'description': f'Raw CPU count ({self.system_info["cpu_count"]}) being used instead of OPTIMAL_WORKERS (8)',
                'evidence_count': len(cpu_count_usage),
                'severity': 'CRITICAL',
                'files_affected': list(set(ref['file'] for ref in cpu_count_usage))
            })
        
        # Check for hardcoded 10s
        hardcoded_10s = [ref for ref in self.scan_results['cpu_count_references']
                        if '10' in ref['match']]
        
        if hardcoded_10s:
            root_causes.append({
                'cause': 'HARDCODED_THREAD_COUNT_10',
                'description': 'Hardcoded thread count of 10 conflicts with OPTIMAL_WORKERS (8)',
                'evidence_count': len(hardcoded_10s),
                'severity': 'CRITICAL',
                'files_affected': list(set(ref['file'] for ref in hardcoded_10s))
            })
        
        # Check environment variables
        env_conflicts = [env for env in self.scan_results['environment_issues']
                        if env.get('matches_cpu_count', False) and not env.get('matches_optimal_8', False)]
        
        if env_conflicts:
            root_causes.append({
                'cause': 'ENVIRONMENT_VARIABLE_CONFLICT',
                'description': 'Environment variables set to CPU count instead of OPTIMAL_WORKERS',
                'evidence_count': len(env_conflicts),
                'severity': 'HIGH',
                'variables_affected': [env['variable'] for env in env_conflicts]
            })
        
        return {
            'primary_root_causes': root_causes,
            'conflict_pattern': 'NUMBA thread manager locked to 8, but other code trying to use 10',
            'fix_strategy': 'Replace all raw CPU count usage with OPTIMAL_WORKERS from technical_foundation'
        }
    
    def generate_fix_recommendations(self) -> List[Dict[str, str]]:
        """💡 Generate targeted fix recommendations"""
        recommendations = []
        
        # Critical fixes
        if self.scan_results['critical_locations']:
            recommendations.append({
                'priority': 'IMMEDIATE',
                'issue': 'Raw CPU count usage in critical files',
                'solution': 'Replace multiprocessing.cpu_count() with OPTIMAL_WORKERS',
                'action': 'Import OPTIMAL_WORKERS from technical_foundation and use consistently',
                'files_to_fix': len(set(loc['file'] for loc in self.scan_results['critical_locations']))
            })
        
        # Thread inconsistencies
        if self.scan_results['thread_inconsistencies']:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Multiple different thread counts in same files',
                'solution': 'Standardize all thread counts to use OPTIMAL_WORKERS',
                'action': 'Update ThreadPoolExecutor and thread_count parameters',
                'files_to_fix': len(self.scan_results['thread_inconsistencies'])
            })
        
        # Environment conflicts
        critical_env = [env for env in self.scan_results['environment_issues'] 
                       if env.get('risk') == 'CRITICAL']
        if critical_env:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Environment variables conflict with thread manager',
                'solution': 'Set NUMBA_NUM_THREADS=8 in environment',
                'action': 'Update startup scripts and environment configuration',
                'variables_to_fix': len(critical_env)
            })
        
        # M4 detector issues
        if self.scan_results.get('m4_detector_analysis'):
            detector_issues = []
            for analysis in self.scan_results['m4_detector_analysis']:
                detector_issues.extend(analysis.get('potential_issues', []))
            
            if detector_issues:
                recommendations.append({
                    'priority': 'HIGH',
                    'issue': 'M4SystemDetector logic may return inconsistent values',
                    'solution': 'Fix detector to always return consistent thread count',
                    'action': 'Update get_optimal_worker_count() method logic',
                    'detector_issues': len(detector_issues)
                })
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """💾 Save CPU diagnostic report to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cpu_thread_diagnostic_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"📄 Report saved to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return None

def main():
    """🎯 Main CPU diagnostic execution"""
    print("🔍 CPU Thread Count Diagnostic Tool")
    print("=" * 50)
    
    # Initialize diagnostic
    diagnostic = CPUThreadDiagnostic()
    
    print(f"💻 System Detected: {diagnostic.system_info['cpu_count']} cores")
    print(f"🎯 Expected Optimal: 8 threads (OPTIMAL_WORKERS)")
    
    # Run diagnostic
    report = diagnostic.run_cpu_diagnostic()
    
    # Display key findings
    summary = report['cpu_diagnostic_summary']
    print(f"\n📋 CPU DIAGNOSTIC SUMMARY:")
    print(f"Total CPU References: {summary['total_cpu_references']}")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"Thread Inconsistencies: {summary['thread_inconsistencies']}")
    print(f"Environment Conflicts: {summary['environment_conflicts']}")
    
    print(f"\n🚨 ROOT CAUSE ANALYSIS:")
    root_cause = report['root_cause_analysis']
    for cause in root_cause['primary_root_causes']:
        print(f"❌ {cause['cause']}: {cause['description']}")
        print(f"   Files Affected: {cause.get('files_affected', cause.get('variables_affected', []))}")
    
    print(f"\n🎯 IMMEDIATE FIXES REQUIRED:")
    critical_findings = report['critical_findings']
    for fix in critical_findings['immediate_fixes_required'][:5]:
        print(f"🔧 {fix['file']}:{fix['line']} - {fix['match']}")
        print(f"   Fix: {fix.get('recommended_fix', 'Update to use OPTIMAL_WORKERS')}")
    
    print(f"\n💡 FIX RECOMMENDATIONS:")
    for rec in report['fix_recommendations']:
        print(f"🔧 [{rec['priority']}] {rec['issue']}")
        print(f"   Action: {rec['action']}")
    
    # Save detailed report
    filename = diagnostic.save_report(report)
    if filename:
        print(f"\n📄 Detailed report saved to: {filename}")
    
    return report

if __name__ == "__main__":
    report = main()