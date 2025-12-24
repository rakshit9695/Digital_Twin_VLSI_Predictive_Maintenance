"""
Digital Twin Physics - Degradation Models
==========================================

Implements compact models for CMOS aging mechanisms using physical equations.

Models Implemented:
-------------------
1. BTI (Bias Temperature Instability)
   - Gate oxide: {Si, SiO2, HfO2} variants
   - Time dependence: t^m (power-law)
   - Voltage dependence: V^n (exponential)

2. HCI (Hot-Carrier Injection)
   - Channel hot carrier generation
   - Drain voltage and activity dependent
   - Short-channel effects

3. EM (Electromigration)
   - Void formation and growth
   - Current density exponent: J^q (typically q=2)
   - Black's law: MTF = A × J^(-q) × exp(E_a/kT)

Physics Equations:
------------------
BTI (Reaction-Diffusion):
    dN_it/dt = A × (1 - θ(f_recovery)) × exp(-E_a/kT) × (V_stress/V_0)^n × t^(-m)
    where N_it = interface trap density

HCI (Lucky Electron):
    dN_it/dt = A × exp(-E_a/kT) × (V_DS/V_c)^p × I_D × t^(-m)
    where I_D = drain current

EM (Void Growth):
    dρ_void/dt = A × exp(-E_a/kT) × (J/J_crit)^q
    MTF = τ_0 × (J_0/J)^q × exp(E_a/kT)
    where J = current density, q ≈ 2

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Optional, Dict, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class CompactModel(ABC):
    """
    Abstract base class for compact degradation models.
    
    Provides interface for computing degradation rates and mechanisms.
    """
    
    def __init__(self,
                 name: str,
                 E_a: float = 0.12,
                 prefactor: float = 1.0e-5,
                 k_B: float = 8.617e-5):
        """
        Initialize compact model.
        
        Args:
            name: Model name for logging
            E_a: Activation energy [eV]
            prefactor: Pre-exponential factor [various units]
            k_B: Boltzmann constant [eV/K]
        """
        self.name = name
        self.E_a = E_a
        self.prefactor = prefactor
        self.k_B = k_B
        
    @abstractmethod
    def compute_rate(self, **kwargs) -> float:
        """
        Compute degradation rate.
        
        Args:
            **kwargs: Stress conditions (temperature, voltage, activity, etc.)
            
        Returns:
            Degradation rate [fraction/second or other units]
        """
        pass
    
    def arrhenius_factor(self, T_k: float, T_ref_k: float = 373.15) -> float:
        """
        Compute Arrhenius temperature acceleration factor.
        
        Args:
            T_k: Temperature [K]
            T_ref_k: Reference temperature [K]
            
        Returns:
            Acceleration factor A(T) / A(T_ref)
        """
        exponent = (self.E_a / self.k_B) * (1/T_ref_k - 1/T_k)
        return np.exp(exponent)


class BTIModel(CompactModel):
    """
    BTI (Bias Temperature Instability) Compact Model.
    
    Reaction-diffusion model for interface trap generation:
    
    dN_it/dt = A × exp(-E_a/kT) × (V_GS/V_0)^n × t^(-m)
    
    Physics:
    --------
    1. Interface trap creation at Si-SiO2 interface
    2. Depends on:
       - Temperature (Arrhenius)
       - Gate voltage (exponential)
       - Time (power-law recovery)
    3. Different for NBTI (PMOS) and PBTI (NMOS)
    4. Recovery during stress-free periods
    
    Parameters:
    -----------
    - Voltage exponent n: 0.2-0.3 typical
    - Time exponent m: 0.5 (power-law)
    - Activation energy E_a: 0.12-0.13 eV
    - Prefactor varies by technology
    
    Example:
    --------
    >>> bti = BTIModel(E_a=0.13, prefactor=1.1e-5)
    >>> rate = bti.compute_rate(T_k=373.15, V_gs=0.65, t_stress=1000)
    """
    
    def __init__(self,
                 E_a: float = 0.13,
                 prefactor: float = 1.1e-5,
                 voltage_exponent: float = 0.25,
                 time_exponent: float = 0.50,
                 recovery_factor: float = 0.01):
        """
        Initialize BTI model.
        
        Args:
            E_a: Activation energy [eV]
            prefactor: Pre-exponential factor
            voltage_exponent: V^n dependence
            time_exponent: t^(-m) dependence (recovery)
            recovery_factor: Recovery coefficient (0-1)
        """
        super().__init__("BTI", E_a, prefactor)
        self.voltage_exponent = voltage_exponent
        self.time_exponent = time_exponent
        self.recovery_factor = recovery_factor
        
    def compute_rate(self,
                     T_k: float,
                     V_gs: float,
                     V_th0: float = 0.45,
                     t_stress: float = 1.0,
                     is_stress: bool = True) -> float:
        """
        Compute BTI degradation rate.
        
        Args:
            T_k: Temperature [K]
            V_gs: Gate-source voltage [V]
            V_th0: Nominal threshold voltage [V]
            t_stress: Stress duration [seconds]
            is_stress: True if under stress, False if recovery
            
        Returns:
            Degradation rate [1/s]
        """
        # Arrhenius factor
        arr_factor = self.arrhenius_factor(T_k)
        
        # Voltage scaling (stress-dependent)
        V_stress = abs(V_gs)  # Use absolute value for voltage stress
        voltage_factor = (V_stress / V_th0) ** self.voltage_exponent
        
        # Time dependence (power-law)
        if t_stress > 0:
            time_factor = (1 + t_stress) ** (-self.time_exponent)
        else:
            time_factor = 1.0
        
        # Base rate
        rate = self.prefactor * arr_factor * voltage_factor * time_factor
        
        # Apply recovery if not stressing
        if not is_stress:
            rate *= self.recovery_factor
        
        return rate
    
    def compute_vth_shift(self,
                          rate: float,
                          dt: float,
                          vth_shift_coefficient: float = 0.18) -> float:
        """
        Map degradation rate to V_th shift.
        
        Args:
            rate: Degradation rate [1/s]
            dt: Time step [s]
            vth_shift_coefficient: ΔV_th per unit damage [mV]
            
        Returns:
            Threshold voltage shift [V]
        """
        damage = rate * dt
        return damage * vth_shift_coefficient / 1000  # Convert mV to V


class HCIModel(CompactModel):
    """
    HCI (Hot-Carrier Injection) Compact Model.
    
    Lucky electron model for hot-carrier trap generation:
    
    dN_it/dt = A × exp(-E_a/kT) × (V_DS/V_c)^p × I_D × α
    
    Physics:
    --------
    1. Carriers gain energy from high drain voltage
    2. Some reach sufficient energy to create traps
    3. Depends on:
       - Drain voltage (lucky electron threshold)
       - Drain current (carrier flux)
       - Switching activity α (HCI only during transitions)
    4. More severe in short-channel devices
    5. Activity-dependent (unlike BTI)
    
    Parameters:
    -----------
    - Power exponent p: 1.5-2.0
    - Activity dependence: proportional to f × duty cycle
    - Activation energy E_a: 0.08-0.10 eV
    - Min drain voltage: ~0.2V (lower in 7nm)
    
    Example:
    --------
    >>> hci = HCIModel(E_a=0.08, prefactor=0.8e-7)
    >>> rate = hci.compute_rate(T_k=313.15, V_ds=0.75, 
    ...                          I_d=1e-3, activity=0.4)
    """
    
    def __init__(self,
                 E_a: float = 0.08,
                 prefactor: float = 0.8e-7,
                 power_exponent: float = 1.5,
                 min_v_ds: float = 0.2):
        """
        Initialize HCI model.
        
        Args:
            E_a: Activation energy [eV]
            prefactor: Pre-exponential factor
            power_exponent: (V_DS)^p dependence
            min_v_ds: Minimum drain voltage for HCI [V]
        """
        super().__init__("HCI", E_a, prefactor)
        self.power_exponent = power_exponent
        self.min_v_ds = min_v_ds
        
    def compute_rate(self,
                     T_k: float,
                     V_ds: float,
                     I_d: float = 1e-3,
                     activity: float = 0.4,
                     V_dsat: float = 0.2) -> float:
        """
        Compute HCI degradation rate.
        
        Args:
            T_k: Temperature [K]
            V_ds: Drain-source voltage [V]
            I_d: Drain current [A]
            activity: Switching activity [0-1]
            V_dsat: Saturation voltage [V]
            
        Returns:
            Degradation rate [1/s]
        """
        # Check minimum voltage for HCI
        if V_ds < self.min_v_ds:
            return 0.0
        
        # Arrhenius factor
        arr_factor = self.arrhenius_factor(T_k)
        
        # Voltage scaling (lucky electron threshold)
        V_stress = max(V_ds - V_dsat, 0.01)  # Effective stress voltage
        voltage_factor = (V_stress / V_dsat) ** self.power_exponent
        
        # Current dependence (carrier flux)
        current_factor = I_d  # Normalized to some reference
        
        # Activity dependence (only harmful during switching)
        activity_factor = activity
        
        # Base rate
        rate = self.prefactor * arr_factor * voltage_factor * current_factor * activity_factor
        
        return rate
    
    def compute_vth_shift(self,
                          rate: float,
                          dt: float,
                          vth_shift_coefficient: float = 0.12) -> float:
        """
        Map HCI rate to V_th shift.
        
        Args:
            rate: Degradation rate [1/s]
            dt: Time step [s]
            vth_shift_coefficient: ΔV_th per unit damage [mV]
            
        Returns:
            Threshold voltage shift [V]
        """
        damage = rate * dt
        return damage * vth_shift_coefficient / 1000


class EMModel(CompactModel):
    """
    EM (Electromigration) Compact Model.
    
    Black's law for electromigration failure:
    
    MTF = τ_0 × (J_0/J)^q × exp(E_a/kT)
    
    Void growth:
    dρ_void/dt = A × exp(-E_a/kT) × (J/J_crit)^q
    
    Physics:
    --------
    1. Atoms drift in direction of electron flow
    2. Creates voids and hillocks
    3. Void growth accelerates failure
    4. Depends on:
       - Current density (drift force)
       - Temperature (Arrhenius)
       - Wire geometry (cross-section)
       - Material (Cu vs Al, with/without barrier)
    5. Very high temperature sensitivity
    6. Critical in narrow interconnects (7nm severe)
    
    Parameters:
    -----------
    - Current exponent q: 1.5-2.5, typically 2
    - Current threshold J_crit: 0.5e6 (7nm) to 1.5e6 (28nm) A/cm²
    - Activation energy E_a: 0.08-0.10 eV
    - Mean time to failure (MTF) ~ 1000 hours at J_crit
    
    Example:
    --------
    >>> em = EMModel(E_a=0.09, prefactor=2.5e-8)
    >>> rate = em.compute_rate(T_k=373.15, J_a=1.0e6)
    """
    
    def __init__(self,
                 E_a: float = 0.09,
                 prefactor: float = 2.5e-8,
                 current_exponent: float = 2.0,
                 J_crit: float = 1.0e6):
        """
        Initialize EM model.
        
        Args:
            E_a: Activation energy [eV]
            prefactor: Pre-exponential factor
            current_exponent: J^q dependence
            J_crit: Critical current density [A/cm²]
        """
        super().__init__("EM", E_a, prefactor)
        self.current_exponent = current_exponent
        self.J_crit = J_crit
        
    def compute_rate(self,
                     T_k: float,
                     J_a: float,
                     J_threshold: float = 0.5e6) -> float:
        """
        Compute EM degradation rate (void growth).
        
        Args:
            T_k: Temperature [K]
            J_a: Actual current density [A/cm²]
            J_threshold: Threshold density for significant EM [A/cm²]
            
        Returns:
            Void growth rate [nm/s or dimensionless]
        """
        # Check if above critical threshold
        if J_a < J_threshold:
            return 0.0
        
        # Arrhenius factor
        arr_factor = self.arrhenius_factor(T_k)
        
        # Current density scaling (Black's law)
        J_ratio = J_a / J_threshold
        current_factor = J_ratio ** self.current_exponent
        
        # Base rate
        rate = self.prefactor * arr_factor * current_factor
        
        return rate
    
    def compute_mtf(self,
                    T_k: float,
                    J_a: float,
                    mtf_ref: float = 1000) -> float:
        """
        Compute Mean Time To Failure using Black's law.
        
        MTF = MTF_0 × (J_0/J)^q × exp(E_a/kT)
        
        Args:
            T_k: Temperature [K]
            J_a: Current density [A/cm²]
            mtf_ref: Reference MTF at 1e6 A/cm², 373K [hours]
            
        Returns:
            Mean time to failure [hours]
        """
        J_ref = 1.0e6  # Reference current density [A/cm²]
        T_ref = 373.15  # Reference temperature [K]
        
        # Black's law
        arr_factor = np.exp((self.E_a / self.k_B) * (1/T_ref - 1/T_k))
        J_factor = (J_ref / J_a) ** self.current_exponent
        
        mtf = mtf_ref * J_factor * arr_factor
        
        return mtf
    
    def compute_delay_increase(self,
                               rate: float,
                               dt: float,
                               delay_coefficient: float = 0.15) -> float:
        """
        Map void growth to delay increase.
        
        Args:
            rate: Void growth rate [dimensionless]
            dt: Time step [s]
            delay_coefficient: Delay increase per unit damage
            
        Returns:
            Relative delay increase [fraction]
        """
        damage = rate * dt
        return damage * delay_coefficient


class ModelFactory:
    """
    Factory for creating appropriate degradation models.
    
    Simplifies model instantiation with preset parameters for
    different technology nodes.
    """
    
    # Preset parameters for technology nodes
    PRESETS = {
        "28nm": {
            "bti": {"E_a": 0.12, "prefactor": 1.1e-5},
            "pbti": {"E_a": 0.12, "prefactor": 1.0e-6},
            "hci": {"E_a": 0.08, "prefactor": 0.8e-7},
            "em": {"E_a": 0.08, "prefactor": 1.2e-8, "J_crit": 1.5e6},
        },
        "7nm": {
            "bti": {"E_a": 0.13, "prefactor": 1.5e-5},
            "pbti": {"E_a": 0.12, "prefactor": 1.8e-6},
            "hci": {"E_a": 0.08, "prefactor": 2.0e-7},
            "em": {"E_a": 0.09, "prefactor": 2.5e-8, "J_crit": 0.5e6},
        },
    }
    
    @classmethod
    def create_bti_model(cls, tech_node: str = "28nm", **kwargs) -> BTIModel:
        """Create BTI model for specified technology."""
        params = cls.PRESETS.get(tech_node, {}).get("bti", {})
        params.update(kwargs)
        return BTIModel(**params)
    
    @classmethod
    def create_hci_model(cls, tech_node: str = "28nm", **kwargs) -> HCIModel:
        """Create HCI model for specified technology."""
        params = cls.PRESETS.get(tech_node, {}).get("hci", {})
        params.update(kwargs)
        return HCIModel(**params)
    
    @classmethod
    def create_em_model(cls, tech_node: str = "28nm", **kwargs) -> EMModel:
        """Create EM model for specified technology."""
        params = cls.PRESETS.get(tech_node, {}).get("em", {})
        params.update(kwargs)
        return EMModel(**params)
    
    @classmethod
    def create_all_models(cls, tech_node: str = "28nm") -> Dict[str, CompactModel]:
        """Create all degradation models for a technology node."""
        return {
            "bti": cls.create_bti_model(tech_node),
            "pbti": cls.create_pbti_model(tech_node),
            "hci": cls.create_hci_model(tech_node),
            "em": cls.create_em_model(tech_node),
        }
    
    @classmethod
    def create_pbti_model(cls, tech_node: str = "28nm", **kwargs) -> BTIModel:
        """Create PBTI model for specified technology."""
        params = cls.PRESETS.get(tech_node, {}).get("pbti", {})
        params.update(kwargs)
        model = BTIModel(**params)
        model.name = "PBTI"
        return model