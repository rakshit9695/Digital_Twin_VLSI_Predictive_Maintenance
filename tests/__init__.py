"""
Digital Twin Tests Module - Initialization
===========================================

Comprehensive unit and integration tests for the Digital Twin system.

Test Organization:
------------------
1. test_filters.py    - Core filter implementations (EKF, UKF, ParticleFilter)
2. test_inference.py  - End-to-end inference pipeline tests
3. test_physics.py    - Physics model validation and accuracy
4. test_telemetry.py  - Telemetry simulation and preprocessing

Test Categories:
----------------
Unit Tests:
- Individual component functionality
- Edge cases and boundary conditions
- Error handling and recovery

Integration Tests:
- Multi-component interactions
- Full pipeline execution
- Data flow validation

Validation Tests:
- Physics accuracy
- Filter convergence
- State estimation quality

Stress Tests:
- Long-running simulations (1+ years)
- High-frequency sampling
- Outlier robustness

Example Test Run:
-----------------
>>> import unittest
>>> from digital_twin.tests import test_filters, test_inference
>>> 
>>> # Create test suite
>>> loader = unittest.TestLoader()
>>> suite = unittest.TestSuite()
>>> 
>>> # Add test modules
>>> suite.addTests(loader.loadTestsFromModule(test_filters))
>>> suite.addTests(loader.loadTestsFromModule(test_inference))
>>> 
>>> # Run tests
>>> runner = unittest.TextTestRunner(verbosity=2)
>>> result = runner.run(suite)

Test Coverage:
--------------
Core (Filters):
- [ ] EKF filter implementation
- [ ] UKF filter implementation
- [ ] ParticleFilter implementation
- [ ] Filter convergence
- [ ] Covariance propagation

Physics:
- [ ] BTI model accuracy
- [ ] HCI model accuracy
- [ ] EM model accuracy
- [ ] Temperature scaling
- [ ] Stress dependencies

Inference:
- [ ] State estimation accuracy
- [ ] Measurement fusion
- [ ] Constraint enforcement
- [ ] History tracking
- [ ] Diagnostic validity

Telemetry:
- [ ] Synthetic data realism
- [ ] Outlier detection
- [ ] Preprocessing pipeline
- [ ] Quality metrics
- [ ] Calibration accuracy

Version: 1.0.0
Author: Digital Twin Team
Date: December 24, 2025
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules
from . import test_filters
from . import test_inference
from . import test_physics

__all__ = [
    "test_filters",
    "test_inference",
    "test_physics",
]

__version__ = "1.0.0"
__author__ = "Digital Twin Team"
__date__ = "2025-12-24"


def create_test_suite():
    """
    Create comprehensive test suite.
    
    Returns:
        unittest.TestSuite with all tests
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules
    suite.addTests(loader.loadTestsFromModule(test_filters))
    suite.addTests(loader.loadTestsFromModule(test_inference))
    suite.addTests(loader.loadTestsFromModule(test_physics))
    
    return suite


def run_tests(verbosity: int = 2):
    """
    Run all tests.
    
    Args:
        verbosity: Output verbosity level
        
    Returns:
        unittest.TestResult
    """
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)