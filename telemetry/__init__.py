"""
Digital Twin Telemetry Module - Initialization
===============================================

Telemetry module handles measurement acquisition and preprocessing.

Components:
-----------
1. simulator.py   - Synthetic telemetry generation (testing/validation)
2. preprocessor.py - Real measurement processing and normalization
3. __init__.py     - Module initialization and exports

Measurement Types:
------------------
1. Ring Oscillator Frequency (f_RO)
   - Primary aging indicator
   - Decreases with V_th shift
   - Temperature dependent

2. Leakage Current (I_leak)
   - Secondary aging indicator
   - Exponential increase with V_th
   - Temperature sensitive

3. Critical Path Delay (D_crit)
   - Path-specific aging effect
   - Increases with V_th and EM
   - Activity-dependent sampling

Telemetry Pipeline:
-------------------
Raw Measurement
    ↓
[Sensor Noise Removal]
    ↓
[Calibration]
    ↓
[Normalization]
    ↓
[Outlier Detection]
    ↓
Clean Measurement → Pipeline

Simulation:
-----------
- Synthetic physics-based aging
- Realistic sensor noise (Gaussian, quantization)
- Temperature effects
- Activity-dependent patterns
- Multi-sample aggregation

Usage:
------
from digital_twin.telemetry import TelemeterySensor, TelemetrySimulator
from digital_twin.utils import load_config

# Real sensors
sensor = TelemeterySensor()
f_ro, i_leak, d_crit = sensor.read()

# Simulation
config = load_config("config/7nm.yaml")
sim = TelemetrySimulator(config)
for z in sim.generate_samples(n_samples=8760):
    pipeline.step(z, T_k=373.15)

Version: 1.0.0
Author: Digital Twin Team
Date: December 24, 2025
"""

# Import telemetry classes
from .simulator import (
    TelemetrySimulator,
    SyntheticAging,
    NoiseModel,
)

from .preprocessor import (
    TelemetryPreprocessor,
    Calibrator,
    OutlierDetector,
    NormalizationFilter,
)

__all__ = [
    # Simulator
    "TelemetrySimulator",
    "SyntheticAging",
    "NoiseModel",
    # Preprocessor
    "TelemetryPreprocessor",
    "Calibrator",
    "OutlierDetector",
    "NormalizationFilter",
]

__version__ = "1.0.0"
__author__ = "Digital Twin Team"
__date__ = "2025-12-24"