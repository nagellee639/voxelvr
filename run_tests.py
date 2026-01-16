#!/usr/bin/env python3
"""
VoxelVR Test Runner

Runs tests with different configurations and generates reports.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --fast             # Skip slow tests
    python run_tests.py --benchmark        # Run only benchmarks
    python run_tests.py --platform-check   # Check platform compatibility
"""

import subprocess
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime


def run_tests(args: list) -> int:
    """Run pytest with given arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def get_system_info() -> dict:
    """Collect system information."""
    import platform
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
    }
    
    # ONNX Runtime
    try:
        import onnxruntime as ort
        info['onnxruntime'] = {
            'version': ort.__version__,
            'providers': ort.get_available_providers(),
        }
    except ImportError:
        info['onnxruntime'] = 'not installed'
    
    # OpenCV
    try:
        import cv2
        info['opencv'] = cv2.__version__
    except ImportError:
        info['opencv'] = 'not installed'
    
    # NumPy
    try:
        import numpy as np
        info['numpy'] = np.__version__
    except ImportError:
        info['numpy'] = 'not installed'
    
    return info


def main():
    parser = argparse.ArgumentParser(description="VoxelVR Test Runner")
    parser.add_argument('--fast', action='store_true', 
                       help='Skip slow tests')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run only benchmark tests')
    parser.add_argument('--platform-check', action='store_true',
                       help='Run platform compatibility tests')
    parser.add_argument('--overhead', action='store_true',
                       help='Run Python overhead analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--report', action='store_true',
                       help='Generate JSON report')
    parser.add_argument('--gpu', action='store_true',
                       help='Run GPU-specific tests')
    
    args = parser.parse_args()
    
    # Print system info
    print("="*60)
    print("VOXELVR TEST SUITE")
    print("="*60)
    
    info = get_system_info()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {info['platform']}")
    if isinstance(info.get('onnxruntime'), dict):
        print(f"ONNX Runtime: {info['onnxruntime']['version']}")
        print(f"Providers: {info['onnxruntime']['providers']}")
    print("="*60 + "\n")
    
    # Build pytest arguments
    pytest_args = []
    
    if args.verbose:
        pytest_args.append('-v')
    
    if args.fast:
        pytest_args.extend(['-m', 'not slow'])
    
    if args.benchmark:
        pytest_args.extend(['-m', 'benchmark', '-v', '-s'])
    
    if args.platform_check:
        pytest_args.extend(['tests/test_cross_platform.py', '-v'])
    
    if args.overhead:
        pytest_args.extend(['tests/test_python_overhead.py', '-v', '-s'])
    
    if args.gpu:
        pytest_args.extend(['-m', 'gpu'])
    
    if not any([args.benchmark, args.platform_check, args.overhead]):
        # Default: run all tests except slow and benchmark
        if not args.fast:
            pytest_args.extend(['-m', 'not benchmark'])
    
    # Run tests
    result = run_tests(pytest_args)
    
    # Generate report if requested
    if args.report:
        report_path = Path("test_report.json")
        report = {
            'system_info': info,
            'exit_code': result,
            'args': vars(args),
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nReport saved to: {report_path}")
    
    return result


if __name__ == "__main__":
    sys.exit(main())
