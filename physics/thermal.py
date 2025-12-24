"""
Digital Twin Physics - Thermal Models
======================================

Implements temperature-dependent effects and Arrhenius scaling.

Thermal Physics:
----------------
All semiconductor degradation follows Arrhenius law:

    Rate(T) = A × exp(-E_a / (k_B × T))

where:
- A: Pre-exponential factor (mechanism-dependent)
- E_a: Activation energy [eV] (0.08-0.15 eV typical)
- k_B: Boltzmann constant = 8.617e-5 eV/K
- T: Absolute temperature [K]

Acceleration Factor:
    A(T) / A(T_ref) = exp((E_a / k_B) × (1/T_ref - 1/T))

Key Insights:
- E_a determines temperature sensitivity
- Lower E_a → less temperature dependent
- E_a ≈ 0.1 eV → 2-3x acceleration per 60°C
- E_a ≈ 0.2 eV → 10x acceleration per 60°C

Temperature Effects in CMOS:
1. Degradation acceleration (Arrhenius)
2. Leakage increase: I_leak ∝ exp(ΔV_th/nV_T)
3. Frequency decrease: f ∝ sqrt(V_DD - V_th) / sqrt(μ)
4. Power increase: P ∝ α × f × V^2 + I_leak × V
5. Thermal runaway: Feedback loop

Thermal Management:
-------------------
- Max junction temperature: 140°C (413K) typical
- Thermal time constant: 0.1-1.0 seconds
- Heatsink/cooling required in 7nm
- DVFS/DFS essential for thermal control

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ThermalModel:
    """
    Thermal model for Arrhenius-based acceleration.
    
    Computes temperature-dependent acceleration factors for
    all degradation mechanisms.
    
    Physics:
    --------
    Arrhenius acceleration factor:
        A(T) = exp((E_a / k_B) × (1/T_ref - 1/T))
    
    Relative acceleration at different temperatures:
    - 313K (40°C):  0.5-1.0x (baseline or less)
    - 373K (100°C): 1.0x (reference)
    - 423K (150°C): 5-20x (extreme stress)
    
    Example:
    --------
    >>> thermal = ThermalModel(reference_temp_k=373.15)
    >>> accel_313k = thermal.acceleration_factor(E_a=0.13, T_k=313.15)
    >>> accel_373k = thermal.acceleration_factor(E_a=0.13, T_k=373.15)
    >>> accel_423k = thermal.acceleration_factor(E_a=0.13, T_k=423.15)
    # Returns relative to 373K reference
    """
    
    def __init__(self, reference_temp_k: float = 373.15):
        """
        Initialize thermal model.
        
        Args:
            reference_temp_k: Reference temperature for acceleration [K]
                             Default: 373.15K (100°C)
        """
        self.reference_temp_k = reference_temp_k
        self.k_B = 8.617e-5  # Boltzmann constant [eV/K]
        
    def acceleration_factor(self,
                           E_a: float,
                           T_k: float) -> float:
        """
        Compute Arrhenius acceleration factor.
        
        Args:
            E_a: Activation energy [eV]
            T_k: Temperature [K]
            
        Returns:
            Acceleration factor relative to reference temperature
        """
        exponent = (E_a / self.k_B) * (1/self.reference_temp_k - 1/T_k)
        return np.exp(exponent)
    
    def temperature_for_acceleration(self,
                                    E_a: float,
                                    target_accel: float) -> float:
        """
        Find temperature needed for target acceleration.
        
        Args:
            E_a: Activation energy [eV]
            target_accel: Desired acceleration factor
            
        Returns:
            Temperature [K] to achieve target acceleration
        """
        exponent = np.log(target_accel) * self.k_B / E_a
        return 1 / (1/self.reference_temp_k + exponent)
    
    def accelerate_rate(self,
                       rate_ref: float,
                       E_a: float,
                       T_k: float) -> float:
        """
        Apply Arrhenius acceleration to a reference rate.
        
        Args:
            rate_ref: Reference rate at T_ref
            E_a: Activation energy [eV]
            T_k: Temperature [K]
            
        Returns:
            Accelerated rate at temperature T_k
        """
        accel = self.acceleration_factor(E_a, T_k)
        return rate_ref * accel


class ArrheniusScaling:
    """
    Arrhenius scaling helper for degradation mechanisms.
    
    Pre-computes and caches acceleration factors for
    efficient computation during inference.
    
    Example:
    --------
    >>> scaling = ArrheniusScaling(
    ...     E_a_dict={
    ...         "BTI": 0.13,
    ...         "HCI": 0.08,
    ...         "EM": 0.09
    ...     }
    ... )
    >>> factors = scaling.get_factors(T_k=343.15)
    # Returns acceleration factors for each mechanism
    """
    
    def __init__(self,
                 E_a_dict: Dict[str, float],
                 reference_temp_k: float = 373.15):
        """
        Initialize Arrhenius scaling.
        
        Args:
            E_a_dict: Dictionary {mechanism: E_a [eV]}
            reference_temp_k: Reference temperature [K]
        """
        self.E_a_dict = E_a_dict
        self.thermal = ThermalModel(reference_temp_k)
        self.k_B = 8.617e-5
        self.cache = {}
        
    def get_factors(self, T_k: float) -> Dict[str, float]:
        """
        Get acceleration factors for all mechanisms at temperature.
        
        Args:
            T_k: Temperature [K]
            
        Returns:
            Dictionary {mechanism: acceleration_factor}
        """
        if T_k in self.cache:
            return self.cache[T_k]
        
        factors = {}
        for mech, E_a in self.E_a_dict.items():
            factors[mech] = self.thermal.acceleration_factor(E_a, T_k)
        
        self.cache[T_k] = factors
        return factors
    
    def get_factor(self, mechanism: str, T_k: float) -> float:
        """Get acceleration factor for single mechanism."""
        factors = self.get_factors(T_k)
        return factors.get(mechanism, 1.0)
    
    def clear_cache(self):
        """Clear acceleration factor cache."""
        self.cache.clear()


class TemperatureDependence:
    """
    Models temperature effects on circuit parameters.
    
    Includes:
    1. Leakage temperature sensitivity
    2. Frequency temperature sensitivity
    3. Thermal runaway feedback
    4. Heat dissipation and spreading
    
    Physics:
    --------
    Leakage Temperature Dependence:
        I_leak(T) = I_leak,0 × exp(α × (T - T_0))
        where α ≈ 0.5-1.0 %/K (subthreshold)
    
    Frequency Temperature Dependence:
        f(T) = f_0 × (1 - β × (T - T_0))
        where β ≈ 0.1-0.2 %/K
    
    Power Temperature Dependence:
        P(T) = P_0 × (1 + γ × (T - T_0))
        where γ ≈ 1-2% (from leakage increase)
    
    Example:
    --------
    >>> temp_dep = TemperatureDependence(
    ...     T_nominal_k=313.15,
    ...     leakage_sensitivity=0.07,  # 7%/K
    ...     frequency_sensitivity=-0.1  # -0.1%/K
    ... )
    >>> leakage_hot = temp_dep.get_leakage_ratio(T_k=343.15)
    >>> freq_hot = temp_dep.get_frequency_ratio(T_k=343.15)
    """
    
    def __init__(self,
                 T_nominal_k: float = 313.15,
                 leakage_sensitivity: float = 0.07,
                 frequency_sensitivity: float = -0.1,
                 power_sensitivity: float = 0.015):
        """
        Initialize temperature dependence model.
        
        Args:
            T_nominal_k: Nominal operating temperature [K]
            leakage_sensitivity: d(ln(I_leak))/dT [1/K]
            frequency_sensitivity: df/f/dT [1/K]
            power_sensitivity: dP/P/dT [1/K]
        """
        self.T_nominal_k = T_nominal_k
        self.leakage_sensitivity = leakage_sensitivity
        self.frequency_sensitivity = frequency_sensitivity
        self.power_sensitivity = power_sensitivity
        
    def get_leakage_ratio(self, T_k: float) -> float:
        """
        Compute leakage current ratio at temperature.
        
        I_leak(T) / I_leak(T_nominal)
        
        Args:
            T_k: Temperature [K]
            
        Returns:
            Leakage ratio (relative to nominal)
        """
        dT = T_k - self.T_nominal_k
        return np.exp(self.leakage_sensitivity * dT)
    
    def get_frequency_ratio(self, T_k: float) -> float:
        """
        Compute frequency ratio at temperature.
        
        f(T) / f(T_nominal)
        
        Args:
            T_k: Temperature [K]
            
        Returns:
            Frequency ratio (relative to nominal)
        """
        dT = T_k - self.T_nominal_k
        return 1 + self.frequency_sensitivity * dT
    
    def get_power_ratio(self, T_k: float) -> float:
        """
        Compute power ratio at temperature (static leakage only).
        
        P(T) / P(T_nominal)
        
        Args:
            T_k: Temperature [K]
            
        Returns:
            Power ratio (relative to nominal)
        """
        dT = T_k - self.T_nominal_k
        return 1 + self.power_sensitivity * dT


class ThermalTimeconstant:
    """
    Models thermal time constant for transient analysis.
    
    Transfer function:
        T_j(s) = G(s) × P(s)
        where G(s) = 1 / (τ_th × s + 1)
        τ_th = thermal time constant [seconds]
    
    Differential equation:
        τ_th × dT_j/dt + T_j = T_ambient + R_th × P_dyn
        where R_th = thermal resistance [K/W]
    
    Physics:
    --------
    For isolated die:
        τ_th = C_th × R_th
        C_th = thermal capacitance [J/K]
        R_th = thermal resistance [K/W]
    
    Typical values:
    - 28nm: 0.5-1.0 second
    - 7nm: 0.1-0.5 second (smaller die, higher density)
    - Package effects can add 1-10x
    
    Example:
    --------
    >>> thermal_rc = ThermalTimeconstant(
    ...     tau_th=0.5,
    ...     T_ambient_k=313.15,
    ...     R_th=0.1
    ... )
    >>> T_j_new = thermal_rc.step_response(T_j=323.15, P_dyn=100, dt=0.1)
    """
    
    def __init__(self,
                 tau_th: float = 0.5,
                 T_ambient_k: float = 313.15,
                 R_th: float = 0.1):
        """
        Initialize thermal time constant model.
        
        Args:
            tau_th: Thermal time constant [seconds]
            T_ambient_k: Ambient temperature [K]
            R_th: Thermal resistance [K/W]
        """
        self.tau_th = tau_th
        self.T_ambient_k = T_ambient_k
        self.R_th = R_th
        
    def step_response(self,
                     T_j: float,
                     P_dyn: float,
                     dt: float) -> float:
        """
        Compute junction temperature after time step.
        
        Discrete approximation of:
            dT_j/dt = (T_ambient + R_th × P - T_j) / τ_th
        
        Args:
            T_j: Current junction temperature [K]
            P_dyn: Power dissipation [W]
            dt: Time step [seconds]
            
        Returns:
            New junction temperature [K]
        """
        if self.tau_th == 0:
            # Instantaneous response
            return self.T_ambient_k + self.R_th * P_dyn
        
        # First-order linear ODE solution
        T_steady = self.T_ambient_k + self.R_th * P_dyn
        alpha = np.exp(-dt / self.tau_th)
        T_new = alpha * T_j + (1 - alpha) * T_steady
        
        return T_new
    
    def settling_time(self, percent_error: float = 5) -> float:
        """
        Compute settling time to within percent_error of steady-state.
        
        Args:
            percent_error: Error percentage [%]
            
        Returns:
            Time to settle [seconds]
        """
        # For first-order system: t_settle ≈ -τ × ln(percent_error/100)
        return -self.tau_th * np.log(percent_error / 100)


class ThermalFeedback:
    """
    Models thermal feedback loop.
    
    Feedback mechanisms:
    1. Leakage → Temperature → More leakage (thermal runaway)
    2. Leakage → Power → Temperature
    3. Temperature → Degradation → Leakage
    4. Degradation → Frequency loss → Lower power (competing effect)
    
    Stability:
    - Positive feedback (runaway) if leakage sensitivity high
    - Can be stabilized by frequency/voltage scaling
    - Critical above ~120°C
    
    Example:
    --------
    >>> feedback = ThermalFeedback(
    ...     tau_th=0.5,
    ...     leakage_coeff=5e-3,  # 5mA/K
    ...     V_dd=0.75
    ... )
    >>> T_new = feedback.iterate(T_old=323.15, P_dyn=100, iterations=10)
    """
    
    def __init__(self,
                 tau_th: float = 0.5,
                 leakage_coeff: float = 5e-3,
                 V_dd: float = 0.75,
                 T_ambient_k: float = 313.15,
                 R_th: float = 0.1):
        """
        Initialize thermal feedback model.
        
        Args:
            tau_th: Thermal time constant [seconds]
            leakage_coeff: dI_leak/dT [A/K]
            V_dd: Supply voltage [V]
            T_ambient_k: Ambient temperature [K]
            R_th: Thermal resistance [K/W]
        """
        self.thermal_rc = ThermalTimeconstant(tau_th, T_ambient_k, R_th)
        self.leakage_coeff = leakage_coeff
        self.V_dd = V_dd
        
    def iterate(self,
                T_old: float,
                P_dyn: float,
                iterations: int = 5) -> float:
        """
        Iterate thermal feedback loop.
        
        Args:
            T_old: Previous junction temperature [K]
            P_dyn: Dynamic power [W]
            iterations: Number of feedback iterations
            
        Returns:
            Converged junction temperature [K]
        """
        T = T_old
        
        for _ in range(iterations):
            # Leakage power from thermal feedback
            I_leak = self.leakage_coeff * (T - self.thermal_rc.T_ambient_k)
            P_leak = I_leak * self.V_dd
            
            # Total power
            P_total = P_dyn + P_leak
            
            # New temperature from RC model
            T_new = self.thermal_rc.T_ambient_k + self.thermal_rc.R_th * P_total
            
            # Check convergence
            if abs(T_new - T) < 0.01:  # 0.01K tolerance
                return T_new
            
            T = T_new
        
        return T
    
    def is_stable(self) -> bool:
        """
        Check if thermal feedback is stable.
        
        Runaway occurs if dP/dT × R_th > 1
        
        Returns:
            True if system is stable
        """
        # Power sensitivity to temperature
        dP_dT = self.leakage_coeff * self.V_dd
        
        # Stability condition
        loop_gain = dP_dT * self.thermal_rc.R_th
        
        return loop_gain < 0.99  # Leave margin