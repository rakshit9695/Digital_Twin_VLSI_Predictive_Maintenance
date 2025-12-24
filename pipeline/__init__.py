"""
Digital Twin Pipeline Module - Initialization
==============================================

Pipeline module orchestrates end-to-end inference for CMOS reliability.

Components:
-----------
1. inference.py  - Main inference engine (physics → state-space → constraints)
2. attribution.py - Mechanism attribution and diagnostics
3. __init__.py    - Module initialization and exports

Inference Pipeline:
-------------------
Input: Raw telemetry {frequency, leakage, delay}
    ↓
[Physics] Compute degradation rates (BTI, HCI, EM)
    ↓
[State-Space] Predict state evolution (6D degradation vector)
    ↓
[Filter] Bayesian update (EKF/UKF/PF)
    ↓
[Constraints] Enforce physics validity (monotonicity, Arrhenius)
    ↓
[Attribution] Breakdown degradation by mechanism
    ↓
Output: Constrained state + reliability predictions + diagnostics

End-to-End Flow:
----------------
1. Load configuration (tech node, filter type)
2. Create models (physics, state-space, filter)
3. Run inference loop (hourly/daily samples)
4. Accumulate damage (BTI, HCI, EM)
5. Predict lifetime (margins, MTTF)
6. Attribute degradation (percentages)
7. Export results (JSON, CSV, plots)

Usage:
------
from digital_twin.pipeline import ReliabilityPipeline
from digital_twin.utils import load_config

# Setup
config = load_config("config/7nm.yaml")
pipeline = ReliabilityPipeline(config, filter_type="UKF")

# Run inference
for t, telemetry in enumerate(telemetry_stream):
    state_est = pipeline.step(telemetry, T_k=373.15, activity=0.4)
    
    # Get results
    if t % 100 == 0:
        lifetime = pipeline.get_lifetime_remaining()
        print(f"Hour {t}: Lifetime = {lifetime:.1f} years")

# Analyze
attribution = pipeline.get_attribution()
diagnostics = pipeline.get_diagnostics()

Version: 1.0.0
Author: Digital Twin Team
Date: December 24, 2025
"""

# Import pipeline classes
from .inference import (
    ReliabilityPipeline,
    InferenceEngine,
    StateTransition,
)

from .attribution import (
    AttributionAnalyzer,
    LifetimePredictor,
    MechanismBreakdown,
)

__all__ = [
    # Inference
    "ReliabilityPipeline",
    "InferenceEngine",
    "StateTransition",
    # Attribution
    "AttributionAnalyzer",
    "LifetimePredictor",
    "MechanismBreakdown",
]

__version__ = "1.0.0"
__author__ = "Digital Twin Team"
__date__ = "2025-12-24"