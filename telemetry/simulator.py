"""
Digital Twin Telemetry - Synthetic Data Simulator
==================================================

Generates realistic synthetic telemetry for testing and validation.

Simulation Components:
----------------------
1. SyntheticAging - Physics-based degradation over time
2. NoiseModel - Sensor noise (Gaussian, quantization, outliers)
3. TelemetrySimulator - Complete simulator orchestration

Synthetic Aging Physics:
------------------------
Frequency degradation:
    f_RO(t) = f_0 × (1 - β × ΔV_th(t) - γ × Δμ(t))
    
    where:
    - β: frequency sensitivity to V_th (~0.1-0.15)
    - γ: frequency sensitivity to mobility (~0.2)
    - ΔV_th: accumulated threshold shift
    - Δμ: mobility degradation

Leakage current:
    I_leak(t) = I_0 × exp(ΔV_th(t) / nV_T)
    
    where:
    - n: subthreshold slope factor (~1.5)
    - V_T: thermal voltage (0.026V at 300K)
    - Exponential dependence on V_th shift

Delay increase:
    D_crit(t) = D_0 × (1 + α × ΔV_th(t) + ζ × EM_damage(t))
    
    where:
    - α: delay sensitivity to V_th
    - ζ: delay sensitivity to EM

Noise Model:
------------
Additive noise sources:
1. Gaussian: N(0, σ) - thermal noise
2. Quantization: ±0.5 LSB - ADC resolution
3. Outliers: Random spikes (rare events)
4. Drift: Slow sensor calibration drift

Temperature Effects:
--------------------
- Base temperature: 313K (40°C nominal)
- Daily cycle: 10K amplitude, 24-hour period
- Peak stress: afternoons
- Base stress: nights

Activity Patterns:
------------------
- Office hours: High activity (0.6-0.8)
- Off-hours: Low activity (0.2-0.3)
- Weekends: Medium activity (0.3-0.5)

Example:
--------
>>> from digital_twin.telemetry import TelemetrySimulator
>>> import numpy as np
>>> 
>>> sim = TelemetrySimulator(config)
>>> 
>>> # Generate 1 year of hourly samples
>>> for hour in range(365*24):
...     z = sim.step()
...     pipeline.step(z, T_k=sim.get_temperature(), activity=sim.get_activity())
>>> 
>>> # Get statistics
>>> stats = sim.get_statistics()
>>> print(f"Frequency degradation: {stats['f_ro_degradation']:.2f}%")

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SyntheticAging:
    """
    Synthetic aging model for realistic telemetry generation.
    
    Generates aging-induced observable changes based on physics.
    
    Physics Model:
    ---------------
    State-to-Observable mapping:
    - V_th shift → Frequency decrease
    - V_th shift → Leakage increase (exponential)
    - V_th shift + EM → Delay increase
    
    Example:
    --------
    >>> aging = SyntheticAging()
    >>> z = aging.compute_observables(state=[0.001, 0.0005, 0.002, 0.001, 0.005, 0.01])
    """
    
    def __init__(self,
                 f_ro_nominal: float = 2400.0,
                 i_leak_nominal: float = 1.2e-6,
                 d_crit_nominal: float = 1.5):
        """
        Initialize synthetic aging model.
        
        Args:
            f_ro_nominal: Nominal ring oscillator frequency [MHz]
            i_leak_nominal: Nominal leakage current [A]
            d_crit_nominal: Nominal critical path delay [ns]
        """
        self.f_ro_nominal = f_ro_nominal
        self.i_leak_nominal = i_leak_nominal
        self.d_crit_nominal = d_crit_nominal
        
        # Sensitivity parameters
        self.freq_v_th_sensitivity = 0.12  # Frequency ∝ V_th
        self.freq_mu_sensitivity = 0.20    # Frequency ∝ mobility
        self.delay_v_th_sensitivity = 0.08  # Delay ∝ V_th
        self.delay_em_sensitivity = 0.15    # Delay ∝ EM
        self.leakage_n_factor = 1.5        # Subthreshold factor
        
    def compute_observables(self,
                           state: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute frequency, leakage, delay from state.
        
        Args:
            state: State vector [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
            
        Returns:
            Tuple of (f_ro [MHz], i_leak [A], d_crit [ns])
        """
        # Extract state components
        v_th_shift = state[4]  # [V]
        mu_degrade = state[5]  # [fraction]
        em_damage = state[3]   # [fraction]
        
        # Frequency: Decreases with V_th and mobility loss
        f_decrease = (self.freq_v_th_sensitivity * v_th_shift / 0.45 +
                     self.freq_mu_sensitivity * mu_degrade)
        f_ro = self.f_ro_nominal * (1 - f_decrease)
        
        # Leakage: Exponential increase with V_th
        # I_leak = I_0 × exp(ΔV_th / (n × V_T))
        # V_T = 0.026V at 300K
        v_thermal = 0.026
        leakage_factor = np.exp(v_th_shift / (self.leakage_n_factor * v_thermal))
        i_leak = self.i_leak_nominal * leakage_factor
        
        # Delay: Increases with both V_th and EM
        d_increase = (self.delay_v_th_sensitivity * v_th_shift / 0.45 +
                     self.delay_em_sensitivity * em_damage)
        d_crit = self.d_crit_nominal * (1 + d_increase)
        
        return f_ro, i_leak, d_crit
    
    def compute_derivatives(self,
                           state: np.ndarray,
                           rates: Dict[str, float]) -> np.ndarray:
        """
        Compute observable derivatives from rates.
        
        Useful for checking consistency.
        
        Args:
            state: Current state
            rates: Degradation rates {mechanism: rate}
            
        Returns:
            Observable derivatives [df_ro/dt, dI_leak/dt, dD_crit/dt]
        """
        # Sum all V_th contributing rates
        d_vth_dt = (0.18e-3 * rates.get("NBTI", 0) +
                   0.12e-3 * rates.get("HCI", 0))  # mV/s → V/s
        
        # Sum all EM contributing rates
        d_em_dt = rates.get("EM", 0)
        
        v_th_shift = state[4]
        f_ro, i_leak, d_crit = self.compute_observables(state)
        
        # Frequency derivative
        df_ro_dt = -self.f_ro_nominal * self.freq_v_th_sensitivity * d_vth_dt / 0.45
        
        # Leakage derivative (exponential)
        v_thermal = 0.026
        di_leak_dt = (self.i_leak_nominal * 
                     np.exp(v_th_shift / (self.leakage_n_factor * v_thermal)) *
                     d_vth_dt / (self.leakage_n_factor * v_thermal))
        
        # Delay derivative
        dd_crit_dt = self.d_crit_nominal * (
            self.delay_v_th_sensitivity * d_vth_dt / 0.45 +
            self.delay_em_sensitivity * d_em_dt
        )
        
        return np.array([df_ro_dt, di_leak_dt, dd_crit_dt])


class NoiseModel:
    """
    Sensor noise model for realistic measurements.
    
    Noise Sources:
    ---------------
    1. Gaussian thermal noise
    2. Quantization (ADC discretization)
    3. Random outliers (rare sensor glitches)
    4. Slow drift (calibration drift)
    
    Example:
    --------
    >>> noise = NoiseModel(f_ro_noise_mhz=0.5, resolution_bits=14)
    >>> z_noisy = noise.add_noise(z_clean)
    """
    
    def __init__(self,
                 f_ro_noise_mhz: float = 0.5,
                 i_leak_noise_a: float = 5e-9,
                 d_crit_noise_ns: float = 0.02,
                 quantization_bits: int = 14,
                 outlier_probability: float = 0.001,
                 drift_rate: float = 1e-5):
        """
        Initialize noise model.
        
        Args:
            f_ro_noise_mhz: Gaussian noise std dev [MHz]
            i_leak_noise_a: Gaussian noise std dev [A]
            d_crit_noise_ns: Gaussian noise std dev [ns]
            quantization_bits: ADC resolution
            outlier_probability: Probability of outlier per sample
            drift_rate: Slow calibration drift rate
        """
        self.f_ro_sigma = f_ro_noise_mhz
        self.i_leak_sigma = i_leak_noise_a
        self.d_crit_sigma = d_crit_noise_ns
        self.bits = quantization_bits
        self.outlier_prob = outlier_probability
        self.drift_rate = drift_rate
        self.drift_state = 0.0
        self.step_count = 0
        
    def add_noise(self, z: np.ndarray) -> np.ndarray:
        """
        Add realistic noise to clean measurement.
        
        Args:
            z: Clean measurement [f_ro, i_leak, d_crit]
            
        Returns:
            Noisy measurement
        """
        z_noisy = z.copy()
        
        # Gaussian thermal noise
        z_noisy[0] += np.random.normal(0, self.f_ro_sigma)  # Frequency
        z_noisy[1] += np.random.normal(0, self.i_leak_sigma)  # Leakage
        z_noisy[2] += np.random.normal(0, self.d_crit_sigma)  # Delay
        
        # Quantization noise (simulate ADC)
        lsb_f = 2400.0 / (2 ** self.bits)  # LSB for frequency
        lsb_i = 10e-6 / (2 ** self.bits)   # LSB for leakage
        lsb_d = 10.0 / (2 ** self.bits)    # LSB for delay
        
        z_noisy[0] = np.round(z_noisy[0] / lsb_f) * lsb_f
        z_noisy[1] = np.round(z_noisy[1] / lsb_i) * lsb_i
        z_noisy[2] = np.round(z_noisy[2] / lsb_d) * lsb_d
        
        # Random outliers (rare spikes)
        if np.random.random() < self.outlier_prob:
            outlier_idx = np.random.randint(0, 3)
            z_noisy[outlier_idx] *= np.random.uniform(0.8, 1.2)
        
        # Slow calibration drift
        self.drift_state += np.random.normal(0, self.drift_rate)
        z_noisy[0] += self.drift_state * 0.01  # Small drift effect
        
        self.step_count += 1
        
        return z_noisy


class TelemetrySimulator:
    """
    Complete telemetry simulator for testing and validation.
    
    Combines synthetic aging + noise models to generate realistic
    measurement sequences.
    
    Features:
    ---------
    1. Physics-based aging progression
    2. Realistic sensor noise
    3. Temperature and activity patterns
    4. Multi-year simulation capability
    5. Statistics tracking
    
    Example:
    --------
    >>> from digital_twin.telemetry import TelemetrySimulator
    >>> 
    >>> sim = TelemetrySimulator(config)
    >>> 
    >>> # Run 1-year simulation
    >>> measurements = []
    >>> for hour in range(365*24):
    ...     z, T_k, activity = sim.step()
    ...     measurements.append(z)
    ...     pipeline.step(z, T_k, activity)
    >>> 
    >>> # Get degradation
    >>> stats = sim.get_statistics()
    """
    
    def __init__(self, config: Dict):
        """
        Initialize telemetry simulator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.aging = SyntheticAging()
        self.noise = NoiseModel()
        
        # Degradation state (mirrors actual state)
        self.state = np.zeros(6)
        
        # Time tracking
        self.hour = 0
        self.history = []
        
        # Temperature and activity patterns
        self.T_base_k = 323.15  # 50°C base
        self.T_amplitude = 10.0  # ±10K daily cycle
        
        logger.info("TelemetrySimulator initialized")
        
    def step(self) -> Tuple[np.ndarray, float, float]:
        """
        Execute one simulation step (1 hour).
        
        Returns:
            Tuple of (measurement [f_ro, i_leak, d_crit], T_k, activity)
        """
        # Get temperature (daily cycle + random variation)
        T_k = self._get_temperature()
        
        # Get activity level (office hours pattern + randomness)
        activity = self._get_activity()
        
        # Update degradation state (simplified: just accumulate)
        # In real system, this would come from physics engine
        self.state[4] += 1e-5 * (1 + 0.5 * np.sin(self.hour / 100))  # V_th drift
        self.state[5] += 1e-6 * activity  # Mobility drift (activity-dependent)
        
        # Compute observables from state
        z_clean = self.aging.compute_observables(self.state)
        
        # Add realistic noise
        z = self.noise.add_noise(np.array(z_clean))
        
        # Track history
        self.history.append({
            "hour": self.hour,
            "z": z,
            "z_clean": z_clean,
            "T_k": T_k,
            "activity": activity,
            "state": self.state.copy(),
        })
        
        self.hour += 1
        
        return z, T_k, activity
    
    def _get_temperature(self) -> float:
        """Compute temperature with daily cycle."""
        # Daily cycle (peak at hour 14)
        day_cycle = self.T_amplitude * np.sin(2 * np.pi * (self.hour % 24 - 14) / 24)
        
        # Weekly variation (weekends cooler)
        day_of_week = (self.hour // 24) % 7
        weekly_offset = 2 if day_of_week >= 5 else 0  # 2K cooler on weekends
        
        # Random variation
        random_noise = np.random.normal(0, 1)
        
        return self.T_base_k + day_cycle - weekly_offset + random_noise
    
    def _get_activity(self) -> float:
        """Compute switching activity with daily pattern."""
        hour_of_day = self.hour % 24
        
        # Office hours (8-18): High activity
        if 8 <= hour_of_day < 18:
            base_activity = 0.7
        # Off-hours (0-8, 18-24): Low activity
        else:
            base_activity = 0.25
        
        # Weekends: Reduce activity
        day_of_week = (self.hour // 24) % 7
        if day_of_week >= 5:
            base_activity *= 0.5
        
        # Add random variation
        activity = base_activity + np.random.normal(0, 0.05)
        
        return np.clip(activity, 0, 1)
    
    def get_temperature(self) -> float:
        """Get current temperature."""
        if self.history:
            return self.history[-1]["T_k"]
        return self.T_base_k
    
    def get_activity(self) -> float:
        """Get current activity."""
        if self.history:
            return self.history[-1]["activity"]
        return 0.4
    
    def get_statistics(self) -> Dict:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.history:
            return {}
        
        z_clean_array = np.array([h["z_clean"] for h in self.history])
        z_array = np.array([h["z"] for h in self.history])
        state_array = np.array([h["state"] for h in self.history])
        
        return {
            "n_samples": len(self.history),
            "hours_simulated": self.hour,
            "years_simulated": self.hour / (365.25 * 24),
            "f_ro_degradation": (z_clean_array[0, 0] - z_clean_array[-1, 0]) / z_clean_array[0, 0] * 100,
            "i_leak_increase": (z_clean_array[-1, 1] / z_clean_array[0, 1] - 1) * 100,
            "d_crit_increase": (z_clean_array[-1, 2] - z_clean_array[0, 2]) / z_clean_array[0, 2] * 100,
            "v_th_final_mv": state_array[-1, 4] * 1000,
            "noise_f_ro_mhz": np.std(z_array[:, 0] - z_clean_array[:, 0]),
            "noise_i_leak_a": np.std(z_array[:, 1] - z_clean_array[:, 1]),
            "noise_d_crit_ns": np.std(z_array[:, 2] - z_clean_array[:, 2]),
        }
    
    def reset(self):
        """Reset simulator state."""
        self.state = np.zeros(6)
        self.hour = 0
        self.history = []