"""
Digital Twin Core Module - Initialization
==========================================

Core module provides the foundational components for Bayesian state-space
filtering and constraint enforcement in CMOS aging digital twins.

Components:
-----------
1. state_space.py  - State-space model definitions (SSM, LinearSSM, NonlinearSSM)
2. filters.py      - Bayesian filters (EKF, UKF, ParticleFilter)
3. constraints.py  - Physics constraints (monotonic degradation enforcement)

Usage:
------
from digital_twin.core import StateSpaceModel, ExtendedKalmanFilter
from digital_twin.core import MonotonicProjectionFilter

# Create a nonlinear state-space model
ssm = NonlinearSSM(
    n_states=6,
    n_observables=3,
    dt=0.05
)

# Apply EKF for inference
ekf = ExtendedKalmanFilter(ssm)
state_estimate = ekf.predict()
state_estimate = ekf.update(measurement)

# Enforce monotonicity (aging never decreases)
proj_filter = MonotonicProjectionFilter()
state_estimate = proj_filter.project(state_estimate)

Version: 1.0.0
Author: Digital Twin Team
Date: December 24, 2025
"""

# Import core classes
from .state_space import (
    StateSpaceModel,
    LinearSSM,
    NonlinearSSM,
)

from .filters import (
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter,
)

from .constraints import (
    MonotonicProjectionFilter,
    ArrheniusValidation,
)

__all__ = [
    # State-space models
    "StateSpaceModel",
    "LinearSSM",
    "NonlinearSSM",
    # Filters
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    # Constraints
    "MonotonicProjectionFilter",
    "ArrheniusValidation",
]

__version__ = "1.0.0"
__author__ = "Digital Twin Team"
__date__ = "2025-12-24"