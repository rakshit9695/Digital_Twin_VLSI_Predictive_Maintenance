"""
CMOS Digital Twin System
========================

Complete semiconductor device reliability digital twin framework.

Modules:
--------
- config: Technology node configurations (28nm-7nm)
- core: Inference engines (EKF, UKF, ParticleFilter)
- physics: Degradation models (BTI, HCI, EM)
- pipeline: End-to-end inference orchestration
- telemetry: Synthetic data generation & preprocessing
- tests: 56+ test cases for validation
- utils: Configuration & logging utilities

Features:
---------
✅ State-space physics modeling
✅ Bayesian state estimation (EKF/UKF/PF)
✅ Real-time aging prediction
✅ Mechanism attribution (NBTI/HCI/EM)
✅ Lifetime estimation (MTTF)
✅ Constraint enforcement
✅ Telemetry preprocessing
✅ Production-ready logging

Quick Start:
-----------
from digital_twin.utils import load_config, setup_logging
from digital_twin.pipeline import ReliabilityPipeline

# Initialize
setup_logging("logs/")
config = load_config("config/7nm.yaml")

# Create pipeline
pipeline = ReliabilityPipeline(config)

# Run inference
for measurement in data_stream:
    state = pipeline.step(measurement, T_k=373.15, activity=0.4)

Version: 1.0.0
Author: BITS Pilani Electronics & VLSI Lab
Date: December 24, 2025
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Digital Twin Team"
__date__ = "2025-12-24"
__all__ = [
    "config",
    "core",
    "physics",
    "pipeline",
    "telemetry",
    "tests",
    "utils",
]