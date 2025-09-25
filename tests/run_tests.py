"""
Comprehensive test runner for the QuantAI Trading Platform.

This script runs all tests and provides detailed reporting.
"""

import pytest
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_unit_tests():
    """Run all unit tests."""
    print("🧪 Running Unit Tests...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run unit tests with verbose output
    result = pytest.main([
        "tests/unit/",
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Unit Tests completed in {duration:.2f} seconds")
    return result == 0


def run_integration_tests():
    """Run all integration tests."""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run integration tests
    result = pytest.main([
        "tests/integration/",
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Integration Tests completed in {duration:.2f} seconds")
    return result == 0


def run_performance_tests():
    """Run performance tests."""
    print("\n⚡ Running Performance Tests...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run performance tests
    result = pytest.main([
        "tests/unit/test_risk_management.py::TestStressTesting",
        "tests/integration/test_trading_pipeline.py::TestPerformanceMonitoringIntegration",
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  Performance Tests completed in {duration:.2f} seconds")
    return result == 0


def run_coverage_report():
    """Run tests with coverage reporting."""
    print("\n📊 Running Coverage Report...")
    print("=" * 50)
    
    try:
        import coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run all tests
        pytest.main([
            "tests/",
            "-v",
            "--tb=short"
        ])
        
        cov.stop()
        cov.save()
        
        # Generate coverage report
        cov.report()
        
        # Generate HTML coverage report
        cov.html_report(directory='htmlcov')
        print("\n📁 HTML coverage report generated in 'htmlcov/' directory")
        
        return True
        
    except ImportError:
        print("❌ Coverage package not installed. Install with: pip install coverage")
        return False


def run_specific_test_suite(suite_name):
    """Run a specific test suite."""
    test_suites = {
        'data': 'tests/unit/test_data_sources.py',
        'risk': 'tests/unit/test_risk_management.py',
        'sentiment': 'tests/unit/test_sentiment_analysis.py',
        'integration': 'tests/integration/test_trading_pipeline.py',
        'all': 'tests/'
    }
    
    if suite_name not in test_suites:
        print(f"❌ Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(test_suites.keys())}")
        return False
    
    print(f"\n🎯 Running {suite_name.upper()} Test Suite...")
    print("=" * 50)
    
    start_time = time.time()
    
    result = pytest.main([
        test_suites[suite_name],
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n⏱️  {suite_name.upper()} Tests completed in {duration:.2f} seconds")
    return result == 0


def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n📋 Generating Test Report...")
    print("=" * 50)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': {},
        'summary': {}
    }
    
    # Run different test categories
    test_categories = [
        ('unit', run_unit_tests),
        ('integration', run_integration_tests),
        ('performance', run_performance_tests)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category, test_func in test_categories:
        try:
            result = test_func()
            report['test_results'][category] = {
                'status': 'PASSED' if result else 'FAILED',
                'timestamp': datetime.now().isoformat()
            }
            if result:
                passed_tests += 1
            total_tests += 1
        except Exception as e:
            report['test_results'][category] = {
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            total_tests += 1
    
    # Generate summary
    report['summary'] = {
        'total_categories': total_tests,
        'passed_categories': passed_tests,
        'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    }
    
    # Save report
    import json
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Test Summary:")
    print(f"   Total Categories: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"   Report saved to: test_report.json")
    
    return report


def main():
    """Main test runner function."""
    print("🚀 QuantAI Trading Platform - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'unit':
            success = run_unit_tests()
        elif command == 'integration':
            success = run_integration_tests()
        elif command == 'performance':
            success = run_performance_tests()
        elif command == 'coverage':
            success = run_coverage_report()
        elif command in ['data', 'risk', 'sentiment', 'all']:
            success = run_specific_test_suite(command)
        elif command == 'report':
            report = generate_test_report()
            success = report['summary']['success_rate'] > 80
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: unit, integration, performance, coverage, data, risk, sentiment, all, report")
            success = False
    else:
        # Run all tests by default
        print("Running all test suites...")
        
        unit_success = run_unit_tests()
        integration_success = run_integration_tests()
        performance_success = run_performance_tests()
        
        success = unit_success and integration_success and performance_success
        
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed!")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
