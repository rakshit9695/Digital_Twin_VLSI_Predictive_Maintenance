"""
Digital Twin Physics Module - Initialization
=============================================

Physics module provides degradation mechanisms and thermal models
for CMOS aging simulation.

Components:
-----------
1. models.py      - Degradation models (BTI, HCI, EM compact models)
2. degradation.py - Unified degradation interface and mechanisms
3. thermal.py     - Thermal effects and Arrhenius scaling

Degradation Mechanisms:
-----------------------
1. BTI (Bias Temperature Instability)
   - NBTI: PMOS high |V_GS| stress
   - PBTI: NMOS high |V_GS| stress
   - Causes threshold voltage shift

2. HCI (Hot-Carrier Injection)
   - High |V_DS| × high switching activity
   - Creates interface traps
   - Stronger in short-channel devices

3. EM (Electromigration)
   - High current density in narrow wires
   - Temperature accelerated
   - Interconnect lifetime limit

Thermal Physics:
----------------
- Arrhenius scaling: A(T) = A_ref × exp(E_a/k × (1/T_ref - 1/T))
- Temperature-dependent reaction rates
- Activation energy: 0.08-0.15 eV typical
- Boltzmann constant: k = 8.617e-5 eV/K

Usage:
------
from digital_twin.physics import BTIDegradation, HCIDegradation
from digital_twin.physics import ThermalModel, DegradationComposer

# Create degradation mechanisms
bti = BTIDegradation(E_a=0.13, prefactor=1.1e-5)
hci = HCIDegradation(E_a=0.08, prefactor=0.8e-7)

# Apply thermal scaling
thermal = ThermalModel(reference_temp_k=373.15)
bti_rate = thermal.accelerate(bti_rate_ref, T_k=343.15)

# Compose all mechanisms
composer = DegradationComposer([bti, hci])
total_damage = composer.compute_damage(...)

Version: 1.0.0
Author: Digital Twin Team
Date: December 24, 2025
"""

# Import physics classes
from .models import (
    CompactModel,
    BTIModel,
    HCIModel,
    EMModel,
)

from .degradation import (
    DegradationMechanism,
    BTIDegradation,
    PBTIDegradation,
    HCIDegradation,
    EMDegradation,
    DegradationComposer,
)

from .thermal import (
    ThermalModel,
    ArrheniusScaling,
    TemperatureDependence,
)

__all__ = [
    # Models
    "CompactModel",
    "BTIModel",
    "HCIModel",
    "EMModel",
    # Degradation mechanisms
    "DegradationMechanism",
    "BTIDegradation",
    "PBTIDegradation",
    "HCIDegradation",
    "EMDegradation",
    "DegradationComposer",
    # Thermal
    "ThermalModel",
    "ArrheniusScaling",
    "TemperatureDependence",
]

__version__ = "1.0.0"
__author__ = "Digital Twin Team"
__date__ = "2025-12-24"