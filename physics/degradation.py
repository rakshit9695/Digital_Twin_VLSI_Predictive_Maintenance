# ============================================================================
# CMOS DIGITAL TWIN - PHYSICS MODULE: DEGRADATION MECHANISMS
# ============================================================================
# degradation.py
# Complete Unified Degradation Interface with All Mechanisms
# ============================================================================

"""
Digital Twin Physics - Unified Degradation Interface

Provides unified interface for degradation mechanisms and composition.

Degradation Mechanism Hierarchy:
- DegradationMechanism (abstract base class)
  ├── BTIDegradation (NBTI - Negative BTI)
  ├── PBTIDegradation (PBTI - Positive BTI)
  ├── HCIDegradation (HCI - Hot Carrier Injection)
  └── EMDegradation (EM - Electromigration)

Each mechanism maps stress conditions → damage rate → observable effects

Stress Conditions:
1. Temperature T [K]: Arrhenius acceleration (thermal)
2. Voltage V [V]: Determines stress intensity
3. Activity α [0-1]: Switching frequency impact
4. Current J [A/cm²]: For EM concerns
5. Time t [s]: Stress duration

Damage Outputs:
1. ΔV_th [V]: Threshold voltage shift
2. μ_degrade [fraction]: Mobility loss
3. I_leak_increase [fraction]: Leakage current growth
4. f_decrease [fraction]: Frequency degradation
5. Delay_increase [fraction]: Path delay growth

Composition:
Individual mechanisms combined to compute:
- Total damage
- Attribution percentages
- Field reliability estimates
- Lifetime prediction

Author: Digital Twin Team
Date: December 24, 2025
Version: 2.0 (Production Ready)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DegradationMechanism(ABC):
    """
    Abstract base class for degradation mechanisms.
    
    Defines interface for stress-dependent degradation computation.
    All derived classes must implement compute_damage_rate() and compute_effects()
    """

    def __init__(self, name: str):
        """
        Initialize degradation mechanism.
        
        Args:
            name: Mechanism name (NBTI, PBTI, HCI, EM, etc.)
        """
        self.name = name
        self.accumulated_damage = 0.0
        self.stress_history = []

    @abstractmethod
    def compute_damage_rate(self, **stress_conditions) -> float:
        """
        Compute instantaneous damage rate.
        
        Args:
            **stress_conditions: Temperature, voltage, activity, etc.
        
        Returns:
            Damage rate [fraction/second]
        """
        pass

    @abstractmethod
    def compute_effects(self, damage: float) -> Dict[str, float]:
        """
        Compute observable effects from damage.
        
        Maps accumulated damage to measurable effects:
        - V_th shift
        - Mobility degradation
        - Leakage increase
        - Frequency decrease
        - Delay increase
        
        Args:
            damage: Accumulated damage fraction [0, 1]
        
        Returns:
            Dictionary of effects {effect_name: magnitude}
        """
        pass

    def accumulate(self, rate: float, dt: float):
        """
        Accumulate damage over time step.
        
        Args:
            rate: Damage rate [fraction/second]
            dt: Time step [seconds]
        """
        damage_increment = rate * dt
        self.accumulated_damage = min(self.accumulated_damage + damage_increment, 1.0)
        self.stress_history.append({
            "damage": self.accumulated_damage,
            "rate": rate,
            "dt": dt
        })

    def reset(self):
        """Reset accumulated damage and history."""
        self.accumulated_damage = 0.0
        self.stress_history = []

    def get_statistics(self) -> Dict:
        """Get degradation statistics."""
        if not self.stress_history:
            return {}
        rates = [h["rate"] for h in self.stress_history]
        return {
            "total_damage": self.accumulated_damage,
            "mean_rate": np.mean(rates),
            "max_rate": np.max(rates),
            "min_rate": np.min(rates),
            "n_steps": len(self.stress_history),
        }


# ============================================================================
# NBTI - NEGATIVE BIAS TEMPERATURE INSTABILITY
# ============================================================================

class BTIDegradation(DegradationMechanism):
    """
    NBTI (Negative Bias Temperature Instability) for PMOS.
    
    Physics:
    - Gate oxide stressed with negative V_GS
    - Interface traps generated (N_it)
    - Causes V_th shift and leakage increase
    - Some recovery occurs when stress removed
    
    Stress Dependence:
    - Strong temperature (Arrhenius): ~3-5x per 60°C
    - Moderate voltage: V^0.25
    - Time: power-law with recovery
    
    Observable Effects:
    - ΔV_th: 2-5 mV per 1000 hours (28nm)
    - Leakage increase: 30% per year
    - Frequency decrease: 2-3% per year
    
    Example:
    >>> nbti = BTIDegradation(E_a=0.13, prefactor=1.1e-5)
    >>> rate = nbti.compute_damage_rate(T_k=373.15, V_gs=-0.65)
    >>> effects = nbti.compute_effects(damage=0.005)
    """

    def __init__(self,
                 E_a: float = 0.13,
                 prefactor: float = 1.1e-5,
                 voltage_exponent: float = 0.25,
                 recovery_factor: float = 0.01):
        """
        Initialize NBTI mechanism.
        
        Args:
            E_a: Activation energy [eV]
            prefactor: Pre-exponential factor
            voltage_exponent: V^n stress dependence
            recovery_factor: Recovery coefficient during off-stress
        """
        super().__init__("NBTI")
        self.E_a = E_a
        self.prefactor = prefactor
        self.voltage_exponent = voltage_exponent
        self.recovery_factor = recovery_factor
        self.k_B = 8.617e-5  # Boltzmann constant [eV/K]

    def compute_damage_rate(self,
                           T_k: float,
                           V_gs: float = -0.65,
                           V_th0: float = -0.45,
                           t_stress: float = 1000,
                           is_stressed: bool = True) -> float:
        """
        Compute NBTI damage rate.
        
        Args:
            T_k: Temperature [K]
            V_gs: Gate-source voltage [V] (negative for PMOS stress)
            V_th0: Nominal threshold voltage [V]
            t_stress: Cumulative stress time [hours]
            is_stressed: True if under stress, False if recovery
        
        Returns:
            Damage rate [fraction/hour]
        """
        # Arrhenius acceleration
        arr_factor = np.exp((self.E_a / self.k_B) * (1/373.15 - 1/T_k))
        
        # Voltage stress
        V_stress = abs(V_gs) / abs(V_th0)  # Normalized stress
        voltage_factor = V_stress ** self.voltage_exponent
        
        # Time dependence (power-law)
        time_factor = (1 + t_stress) ** (-0.5)
        
        # Base rate
        rate = self.prefactor * arr_factor * voltage_factor * time_factor
        
        # Apply recovery if not stressed
        if not is_stressed:
            rate *= self.recovery_factor
        
        return rate

    def compute_effects(self, damage: float) -> Dict[str, float]:
        """
        Compute NBTI effects from accumulated damage.
        
        Args:
            damage: Accumulated damage fraction [0, 1]
        
        Returns:
            Dictionary of effects
        """
        # V_th shift: ~0.18 mV per unit damage (28nm)
        delta_vth = damage * 0.018e-3  # V
        
        # Leakage increase: exponential from V_th shift
        i_leak_increase = np.exp(delta_vth / 0.026) - 1  # V_T = 0.026V
        
        # Mobility degradation: ~0.15 × damage
        mu_degrade = damage * 0.15
        
        # Frequency decrease
        f_decrease = delta_vth / 0.45 * 0.5  # 50% of V_th effect
        
        # Delay increase (inverse of frequency)
        delay_increase = f_decrease / (1 - f_decrease) if f_decrease < 0.9 else 0.5
        
        return {
            "delta_vth": delta_vth,
            "i_leak_increase": i_leak_increase,
            "mu_degrade": mu_degrade,
            "f_decrease": -f_decrease,
            "delay_increase": delay_increase,
        }


# ============================================================================
# PBTI - POSITIVE BIAS TEMPERATURE INSTABILITY
# ============================================================================

class PBTIDegradation(BTIDegradation):
    """
    PBTI (Positive Bias Temperature Instability) for NMOS.
    
    Similar to NBTI but:
    - Positive V_GS stress (N-channel)
    - Lower degradation rate (~10% of NBTI)
    - Different activation energy often observed
    - Increasing concern in advanced nodes
    
    Example:
    >>> pbti = PBTIDegradation(E_a=0.12, prefactor=1.0e-6)
    """

    def __init__(self,
                 E_a: float = 0.12,
                 prefactor: float = 1.0e-6,
                 voltage_exponent: float = 0.25,
                 recovery_factor: float = 0.02):
        """Initialize PBTI mechanism."""
        super().__init__(E_a, prefactor, voltage_exponent, recovery_factor)
        self.name = "PBTI"


# ============================================================================
# HCI - HOT CARRIER INJECTION
# ============================================================================

class HCIDegradation(DegradationMechanism):
    """
    HCI (Hot-Carrier Injection) Degradation.
    
    Physics:
    - High drain voltage + switching activity
    - Hot carriers generate interface traps
    - Activity-dependent (unlike BTI)
    - Worse in short-channel devices (7nm critical)
    
    Stress Dependence:
    - Weak temperature (lower E_a): 2-3x per 60°C
    - Strong voltage: V_DS^1.5-1.8
    - Strong activity: proportional to f×duty
    
    Observable Effects:
    - ΔV_th: Higher than BTI (~5-10 mV in 7nm)
    - Less leakage effect than BTI
    - Direct impact on delay
    
    Example:
    >>> hci = HCIDegradation(E_a=0.08, prefactor=2.0e-7)
    >>> rate = hci.compute_damage_rate(T_k=343.15, V_ds=0.75, activity=0.8)
    """

    def __init__(self,
                 E_a: float = 0.08,
                 prefactor: float = 2.0e-7,
                 power_exponent: float = 1.8,
                 min_v_ds: float = 0.2,
                 hci_alpha: float = 1.0):
        """
        Initialize HCI mechanism.
        
        Args:
            E_a: Activation energy [eV]
            prefactor: Pre-exponential factor
            power_exponent: V_DS^p stress dependence
            min_v_ds: Minimum drain voltage for HCI [V]
            hci_alpha: Voltage scaling factor (CORRECTED PHYSICS)
        """
        super().__init__("HCI")
        self.E_a = E_a
        self.prefactor = prefactor
        self.power_exponent = power_exponent
        self.min_v_ds = min_v_ds
        self.hci_alpha = hci_alpha  # CRITICAL FIX
        self.k_B = 8.617e-5

    def compute_damage_rate(self,
                           T_k: float,
                           V_dd: float = 0.75,
                           V_ds: float = 0.75,
                           activity: float = 0.4,
                           V_dsat: float = 0.2) -> float:
        """
        Compute HCI damage rate with CORRECTED voltage formula.
        
        CORRECTED PHYSICS: Higher voltage = Higher HCI damage
        Formula: exp(-alpha / V_dd) where alpha > 0
        
        Args:
            T_k: Temperature [K]
            V_dd: Supply voltage [V]
            V_ds: Drain-source voltage [V]
            activity: Switching activity [0, 1]
            V_dsat: Saturation voltage [V]
        
        Returns:
            Damage rate [fraction/second]
        """
        # Check minimum voltage
        if V_ds < self.min_v_ds:
            return 0.0

        # Arrhenius factor
        arr_factor = np.exp((self.E_a / self.k_B) * (1/373.15 - 1/T_k))

        # CORRECTED Voltage stress formula
        # Higher V_dd INCREASES HCI (exponential dependence)
        alpha = self.hci_alpha
        voltage_factor = np.exp(-alpha / V_dd)

        # Activity dependence (key difference from BTI)
        activity_factor = activity

        # Base rate
        rate = self.prefactor * arr_factor * voltage_factor * activity_factor

        return rate

    def compute_effects(self, damage: float) -> Dict[str, float]:
        """
        Compute HCI effects.
        
        Args:
            damage: Accumulated damage fraction [0, 1]
        
        Returns:
            Dictionary of effects
        """
        # V_th shift: ~0.12 mV per unit damage (higher in 7nm)
        delta_vth = damage * 0.012e-3  # V

        # Less leakage increase than BTI
        i_leak_increase = np.exp(delta_vth / 0.05) - 1  # Weaker dependence

        # Mobility degradation: ~0.10 × damage
        mu_degrade = damage * 0.10

        # Frequency decrease
        f_decrease = delta_vth / 0.45 * 0.7

        # Delay increase
        delay_increase = f_decrease / (1 - f_decrease) if f_decrease < 0.9 else 0.5

        return {
            "delta_vth": delta_vth,
            "i_leak_increase": i_leak_increase,
            "mu_degrade": mu_degrade,
            "f_decrease": -f_decrease,
            "delay_increase": delay_increase,
        }


# ============================================================================
# EM - ELECTROMIGRATION
# ============================================================================

class EMDegradation(DegradationMechanism):
    """
    EM (Electromigration) Degradation.
    
    Physics:
    - Void formation in narrow wires
    - Current density driven
    - Temperature accelerated (very high E_a)
    - Critical in 7nm interconnects
    
    Stress Dependence:
    - Very strong current: J^2
    - Strong temperature: ~3x per 10°C
    - Time-to-failure: Lognormal distribution
    
    Observable Effects:
    - Resistance increase: 5-20% significant
    - Delay increase: path-specific
    - Complete failure possible
    
    Example:
    >>> em = EMDegradation(E_a=0.09, prefactor=2.5e-8)
    >>> rate = em.compute_damage_rate(T_k=373.15, J_a=1.2e6)
    """

    def __init__(self,
                 E_a: float = 0.09,
                 prefactor: float = 2.5e-8,
                 current_exponent: float = 2.0,
                 J_crit: float = 1.0e6):
        """
        Initialize EM mechanism.
        
        Args:
            E_a: Activation energy [eV]
            prefactor: Pre-exponential factor
            current_exponent: J^q exponent
            J_crit: Critical current density [A/cm²]
        """
        super().__init__("EM")
        self.E_a = E_a
        self.prefactor = prefactor
        self.current_exponent = current_exponent
        self.J_crit = J_crit
        self.k_B = 8.617e-5

    def compute_damage_rate(self,
                           T_k: float,
                           J_a: float,
                           J_threshold: float = 0.5e6) -> float:
        """
        Compute EM damage rate (void growth).
        
        Args:
            T_k: Temperature [K]
            J_a: Current density [A/cm²]
            J_threshold: Threshold for significant EM [A/cm²]
        
        Returns:
            Damage rate [fraction/second]
        """
        # Below threshold, no significant EM
        if J_a < J_threshold:
            return 0.0

        # Arrhenius factor (very temperature sensitive)
        arr_factor = np.exp((self.E_a / self.k_B) * (1/373.15 - 1/T_k))

        # Black's law: J^q
        J_ratio = J_a / J_threshold
        current_factor = J_ratio ** self.current_exponent

        # Base rate
        rate = self.prefactor * arr_factor * current_factor

        return rate

    def compute_effects(self, damage: float) -> Dict[str, float]:
        """
        Compute EM effects.
        
        Args:
            damage: Void volume fraction [0, 1]
        
        Returns:
            Dictionary of effects
        """
        # Resistance increase: ~15% per unit damage
        r_increase = damage * 0.15

        # Delay increase proportional to resistance
        delay_increase = r_increase * 1.2

        # Minimal direct V_th and leakage effect (resistance-driven)
        return {
            "delta_vth": 0.0,
            "i_leak_increase": 0.0,
            "mu_degrade": 0.0,
            "r_increase": r_increase,
            "delay_increase": delay_increase,
        }

    def compute_mtf(self, J_a: float, T_k: float,
                    mtf_ref: float = 1000) -> float:
        """
        Compute mean time to failure using Black's law.
        
        Args:
            J_a: Current density [A/cm²]
            T_k: Temperature [K]
            mtf_ref: Reference MTF at J=1e6 A/cm², T=373K [hours]
        
        Returns:
            Mean time to failure [hours]
        """
        J_ref = 1.0e6
        T_ref = 373.15
        arr_factor = np.exp((self.E_a / self.k_B) * (1/T_ref - 1/T_k))
        J_factor = (J_ref / J_a) ** self.current_exponent
        mtf = mtf_ref * J_factor * arr_factor
        return mtf


# ============================================================================
# DEGRADATION COMPOSER - MULTI-MECHANISM COMPOSITION
# ============================================================================

class DegradationComposer:
    """
    Compose multiple degradation mechanisms.
    
    Combines BTI, HCI, EM effects into total system degradation,
    with attribution analysis.
    
    Example:
    >>> mechanisms = [BTIDegradation(), HCIDegradation(), EMDegradation()]
    >>> composer = DegradationComposer(mechanisms)
    >>>
    >>> # One time step
    >>> rates = composer.compute_rates(T_k=373.15, V_ds=0.75, activity=0.4)
    >>> composer.accumulate(rates, dt=3600)
    >>>
    >>> # Get total effects
    >>> effects = composer.get_total_effects()
    >>> attribution = composer.get_attribution()
    """

    def __init__(self, mechanisms: List[DegradationMechanism]):
        """
        Initialize degradation composer.
        
        Args:
            mechanisms: List of degradation mechanisms
        """
        self.mechanisms = mechanisms

    def compute_rates(self, **stress_conditions) -> Dict[str, float]:
        """
        Compute damage rates for all mechanisms.
        
        Args:
            **stress_conditions: Temperature, voltage, activity, etc.
        
        Returns:
            Dictionary {mechanism_name: rate}
        """
        rates = {}
        for mech in self.mechanisms:
            rates[mech.name] = mech.compute_damage_rate(**stress_conditions)
        return rates

    def accumulate(self, rates: Dict[str, float], dt: float):
        """
        Accumulate damage for all mechanisms.
        
        Args:
            rates: Dictionary of rates from compute_rates()
            dt: Time step [seconds]
        """
        for mech in self.mechanisms:
            if mech.name in rates:
                mech.accumulate(rates[mech.name], dt)

    def get_total_effects(self) -> Dict[str, float]:
        """
        Get total effects from all mechanisms.
        
        Returns:
            Dictionary of cumulative effects
        """
        total = {}
        for mech in self.mechanisms:
            effects = mech.compute_effects(mech.accumulated_damage)
            for key, val in effects.items():
                total[key] = total.get(key, 0) + val
        return total

    def get_attribution(self) -> Dict[str, float]:
        """
        Get mechanism attribution as percentages.
        
        Returns:
            Dictionary {mechanism_name: percentage}
        """
        total_damage = sum(m.accumulated_damage for m in self.mechanisms)
        
        if total_damage == 0:
            return {m.name: 0.0 for m in self.mechanisms}
        
        return {
            m.name: 100 * m.accumulated_damage / total_damage
            for m in self.mechanisms
        }

    def reset(self):
        """Reset all mechanisms."""
        for mech in self.mechanisms:
            mech.reset()

    def get_statistics(self) -> Dict:
        """Get statistics for all mechanisms."""
        return {
            mech.name: mech.get_statistics()
            for mech in self.mechanisms
        }


# ============================================================================
# END OF FILE
# =======================================================================