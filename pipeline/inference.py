"""
Digital Twin Pipeline - Inference Engine
=========================================

Orchestrates end-to-end reliability inference:
Physics → State-Space → Filter → Constraints → Attribution

Inference Loop:
---------------
1. Physics: Compute degradation rates from stress conditions
2. State-Space: Map rates to state evolution
3. Filter: Predict and update state with telemetry
4. Constraints: Enforce physics validity
5. Diagnostics: Check for inconsistencies

State Vector (6D):
-----------------
x = [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]

Measurements (3D):
------------------
z = [f_RO (MHz), I_leak (A), D_crit (ns)]

Stress Inputs (3D):
-------------------
u = [T_k (K), V_dd (V), activity (0-1)]

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class StateTransition:
    """
    Models state transition during each time step.
    
    Maps degradation rates → state changes using physics.
    
    Dynamics:
    ---------
    x(k+1) = x(k) + Δx(k)
    
    where Δx(k) computed from:
    - Degradation rates (BTI, HCI, EM)
    - Time step dt
    - Temperature effects (Arrhenius)
    
    Example:
    --------
    >>> transition = StateTransition()
    >>> rates = {"NBTI": 1e-5, "HCI": 5e-6, "EM": 0}
    >>> delta_x = transition.compute_state_delta(rates, dt=3600)
    """
    
    def __init__(self):
        """Initialize state transition model."""
        self.state_dim = 6
        self.input_dim = 3
        
    def compute_state_delta(self,
                           rates: Dict[str, float],
                           dt: float) -> np.ndarray:
        """
        Compute state change from degradation rates.
        
        Args:
            rates: Dictionary {mechanism: rate [1/s]}
            dt: Time step [seconds]
            
        Returns:
            State delta vector [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
        """
        # Initialize state delta
        delta_x = np.zeros(self.state_dim)
        
        # Map rates to state components
        # NBTI damage (index 0)
        delta_x[0] = rates.get("NBTI", 0.0) * dt
        
        # PBTI damage (index 1)
        delta_x[1] = rates.get("PBTI", 0.0) * dt
        
        # HCI damage (index 2)
        delta_x[2] = rates.get("HCI", 0.0) * dt
        
        # EM damage (index 3)
        delta_x[3] = rates.get("EM", 0.0) * dt
        
        # V_th shift (index 4) - weighted combination
        # ΔV_th ≈ 0.18 mV per NBTI unit + 0.12 mV per HCI unit
        delta_x[4] = (0.18e-3 * delta_x[0] + 0.12e-3 * delta_x[2])
        
        # Mobility degradation (index 5)
        # μ ≈ 0.15 per NBTI unit + 0.10 per HCI unit
        delta_x[5] = (0.15 * delta_x[0] + 0.10 * delta_x[2])
        
        return delta_x
    
    def clamp_to_bounds(self,
                       x: np.ndarray) -> np.ndarray:
        """
        Clamp state to physically valid bounds.
        
        Args:
            x: State vector
            
        Returns:
            Clamped state
        """
        # Damage components: [0, 1]
        x[:4] = np.clip(x[:4], 0, 1)
        
        # V_th shift: [0, 0.5V] (maximum degradation)
        x[4] = np.clip(x[4], 0, 0.5)
        
        # Mobility degradation: [0, 0.5] (50% max)
        x[5] = np.clip(x[5], 0, 0.5)
        
        return x
    
    def enforce_monotonicity(self,
                            x_new: np.ndarray,
                            x_old: np.ndarray) -> np.ndarray:
        """
        Enforce monotonic increase (damage never decreases).
        
        Args:
            x_new: New state estimate
            x_old: Previous state estimate
            
        Returns:
            Monotonicity-enforced state
        """
        # First 4 components (damages) must be monotonic
        for i in range(4):
            x_new[i] = max(x_new[i], x_old[i])
        
        # V_th and mobility also monotonic (never recover)
        x_new[4] = max(x_new[4], x_old[4])
        x_new[5] = max(x_new[5], x_old[5])
        
        return x_new


class InferenceEngine:
    """
    Main inference engine for reliability prediction.
    
    Combines physics, state-space, filter, and constraints
    into unified inference loop.
    
    Pipeline:
    ---------
    Input: telemetry {f_RO, I_leak, D_crit} + stresses {T_k, V_dd, activity}
        ↓
    [Physics] Compute rates (BTI, HCI, EM)
        ↓
    [State-Space] Predict state delta
        ↓
    [Filter] EKF/UKF/PF update
        ↓
    [Constraints] Enforce monotonicity & bounds
        ↓
    Output: State estimate + diagnostics
    
    Example:
    --------
    >>> engine = InferenceEngine(config, filter_type="UKF")
    >>> engine.initialize()
    >>> 
    >>> for telemetry, T_k, activity in data_stream:
    ...     state_est = engine.step(telemetry, T_k, activity)
    ...     print(f"V_th shift: {state_est[4]*1000:.2f} mV")
    """
    
    def __init__(self,
                 config: Dict,
                 filter_type: str = "UKF"):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration dictionary (from YAML)
            filter_type: "EKF", "UKF", or "PF"
        """
        self.config = config
        self.filter_type = filter_type
        
        # Initialize components
        self.state_transition = StateTransition()
        self.state = np.zeros(6)
        self.state_covariance = np.eye(6) * 0.0001
        
        # History tracking
        self.history = []
        self.step_count = 0
        
        logger.info(f"InferenceEngine initialized with {filter_type} filter")
        
    def initialize(self, initial_state: Optional[np.ndarray] = None):
        """
        Initialize inference state.
        
        Args:
            initial_state: Initial state vector (default: zeros)
        """
        if initial_state is None:
            self.state = np.zeros(6)
        else:
            self.state = initial_state.copy()
        
        self.state_covariance = np.eye(6) * 0.0001
        self.step_count = 0
        self.history = []
        
        logger.info(f"Inference state initialized: {self.state}")
        
    def step(self,
             z: np.ndarray,
             T_k: float,
             activity: float = 0.5,
             V_dd: float = 0.75) -> np.ndarray:
        """
        Execute one inference step.
        
        Args:
            z: Measurement vector [f_RO (MHz), I_leak (A), D_crit (ns)]
            T_k: Temperature [K]
            activity: Switching activity [0, 1]
            V_dd: Supply voltage [V]
            
        Returns:
            Estimated state vector
        """
        # Get stress inputs
        u = np.array([T_k, V_dd, activity])
        
        # 1. Physics: Compute degradation rates
        rates = self.compute_degradation_rates(T_k, V_dd, activity)
        
        # 2. State-Space: Predict state delta
        dt = 3600  # Assume 1-hour steps (standard)
        delta_x = self.state_transition.compute_state_delta(rates, dt=dt)
        
        # 3. Filter: Predict
        x_pred = self.state + delta_x
        x_pred = self.state_transition.clamp_to_bounds(x_pred)
        
        # 4. Filter: Update with measurement
        x_upd = self.filter_update(x_pred, z)
        
        # 5. Constraints: Enforce monotonicity
        self.state = self.state_transition.enforce_monotonicity(x_upd, self.state)
        
        # Track history
        self.history.append({
            "step": self.step_count,
            "state": self.state.copy(),
            "rates": rates.copy(),
            "measurement": z.copy(),
            "temperature": T_k,
        })
        
        self.step_count += 1
        
        return self.state
    
    def compute_degradation_rates(self,
                                  T_k: float,
                                  V_dd: float,
                                  activity: float) -> Dict[str, float]:
        """
        Compute degradation rates from stress conditions.
        
        Args:
            T_k: Temperature [K]
            V_dd: Supply voltage [V]
            activity: Switching activity [0, 1]
            
        Returns:
            Dictionary {mechanism: rate [1/s]}
        """
        # Arrhenius acceleration factor
        E_a = 0.13  # Typical for BTI
        k_B = 8.617e-5
        arr_factor = np.exp((E_a / k_B) * (1/373.15 - 1/T_k))
        
        # Get prefactors from config
        nbti_prefactor = self.config.get("physics", {}).get("nbti", {}).get("prefactor", 1.1e-5)
        hci_prefactor = self.config.get("physics", {}).get("hci", {}).get("prefactor", 2.0e-7)
        em_prefactor = self.config.get("physics", {}).get("em", {}).get("prefactor", 2.5e-8)
        
        # NBTI: Temperature and voltage dependent
        nbti_rate = nbti_prefactor * arr_factor * (V_dd / 0.75) ** 0.25
        
        # PBTI: Similar but lower magnitude
        pbti_rate = nbti_rate * 0.1
        
        # HCI: Temperature, voltage, and activity dependent
        hci_rate = hci_prefactor * arr_factor * ((V_dd - 0.2) / 0.55) ** 1.8 * activity
        
        # EM: Very temperature and current sensitive
        J_a = 1.0e6 * activity  # Approximate current density
        em_rate = em_prefactor * arr_factor * ((J_a / 0.5e6) ** 2) if J_a > 0.5e6 else 0.0
        
        return {
            "NBTI": nbti_rate,
            "PBTI": pbti_rate,
            "HCI": hci_rate,
            "EM": em_rate,
        }
    
    def filter_update(self,
                     x_pred: np.ndarray,
                     z: np.ndarray) -> np.ndarray:
        """
        Update state estimate with measurement using filter.
        
        Args:
            x_pred: Predicted state
            z: Measurement [f_RO, I_leak, D_crit]
            
        Returns:
            Updated state estimate
        """
        # Simple linear update (Kalman-like)
        # In full implementation: use actual EKF/UKF/PF
        
        # Measurement mapping: h(x) = f(degradation)
        # Frequency decrease due to V_th shift
        f_deg = x_pred[4] / 0.45 * 0.5  # Normalized
        
        # Measurement residual
        z_pred = np.array([
            -f_deg * 100,  # Frequency decrease [MHz]
            np.exp(x_pred[4] / 0.026) - 1,  # Leakage increase
            x_pred[4] / 0.45 * 10,  # Delay increase [ns]
        ])
        
        # Simple Kalman gain (full impl. uses actual filter)
        K = 0.1  # Kalman gain (simplified)
        residual = z - z_pred
        
        # Update state
        # In full implementation, update specific state components
        x_upd = x_pred + K * residual[0] * 0.01
        
        return x_upd
    
    def get_diagnostics(self) -> Dict:
        """
        Get inference diagnostics.
        
        Returns:
            Dictionary of diagnostic metrics
        """
        if not self.history:
            return {}
        
        latest = self.history[-1]
        rates = latest["rates"]
        
        return {
            "step": self.step_count,
            "state": latest["state"],
            "rates": rates,
            "total_rate": sum(rates.values()),
            "dominant_mechanism": max(rates, key=rates.get),
            "temperature": latest["temperature"],
            "monotonic": np.all(np.diff(self.history[-10:] if len(self.history) >= 10 else self.history, axis=0) >= -1e-6),
        }


class ReliabilityPipeline:
    """
    Complete reliability prediction pipeline.
    
    Orchestrates inference engine with configuration, output, and analysis.
    
    Full Pipeline:
    ---------------
    1. Load config
    2. Create models
    3. Run inference
    4. Compute attribution
    5. Predict lifetime
    6. Export results
    
    Example:
    --------
    >>> from digital_twin.pipeline import ReliabilityPipeline
    >>> 
    >>> # Create pipeline
    >>> pipeline = ReliabilityPipeline(config, filter_type="UKF")
    >>> 
    >>> # Run inference
    >>> for i, (telemetry, T_k, activity) in enumerate(data_stream):
    ...     state = pipeline.step(telemetry, T_k, activity)
    ...     
    ...     if i % 100 == 0:
    ...         lifetime = pipeline.get_lifetime_remaining()
    ...         print(f"Hour {i}: V_th = {state[4]*1000:.2f} mV, "
    ...               f"Lifetime = {lifetime:.1f} years")
    >>> 
    >>> # Analyze results
    >>> attribution = pipeline.get_attribution()
    >>> print(f"BTI: {attribution['NBTI']:.1f}%")
    >>> print(f"HCI: {attribution['HCI']:.1f}%")
    >>> print(f"EM: {attribution['EM']:.1f}%")
    """
    
    def __init__(self,
                 config: Dict,
                 filter_type: str = "UKF"):
        """
        Initialize reliability pipeline.
        
        Args:
            config: Configuration dictionary
            filter_type: "EKF", "UKF", or "PF"
        """
        self.config = config
        self.engine = InferenceEngine(config, filter_type)
        self.engine.initialize()
        
        # Configuration parameters
        self.tech_node = config.get("technology", {}).get("node", "28nm")
        self.v_dd_nominal = config.get("telemetry", {}).get("nominal", {}).get("supply_voltage_v", 0.75)
        self.t_nominal = config.get("telemetry", {}).get("nominal", {}).get("temperature_k", 313.15)
        
        # Margin thresholds
        self.v_th_margin_limit = config.get("validation", {}).get("v_th_margin_mv", 100) / 1000
        self.frequency_margin_limit = config.get("validation", {}).get("frequency_margin_percent", 20) / 100
        
        logger.info(f"ReliabilityPipeline created for {self.tech_node} node")
        
    def step(self,
             z: np.ndarray,
             T_k: float,
             activity: float = 0.5) -> np.ndarray:
        """
        Execute one inference step.
        
        Args:
            z: Measurement [f_RO, I_leak, D_crit]
            T_k: Temperature [K]
            activity: Switching activity [0, 1]
            
        Returns:
            State estimate
        """
        return self.engine.step(z, T_k, activity, self.v_dd_nominal)
    
    def get_lifetime_remaining(self,
                              v_th_limit: Optional[float] = None) -> float:
        """
        Estimate remaining lifetime.
        
        Uses current degradation rate to extrapolate time to failure.
        
        Args:
            v_th_limit: V_th failure threshold [V]
            
        Returns:
            Remaining lifetime [years]
        """
        if not self.engine.history or len(self.engine.history) < 10:
            return np.inf
        
        # Get last state
        state = self.engine.state
        v_th_current = state[4]
        
        # Failure threshold
        if v_th_limit is None:
            v_th_limit = self.v_th_margin_limit
        
        # Remaining margin
        margin = max(0, v_th_limit - v_th_current)
        if margin == 0:
            return 0.0
        
        # Degradation rate (slope)
        recent_history = self.engine.history[-10:]
        v_th_history = [h["state"][4] for h in recent_history]
        rate = (v_th_history[-1] - v_th_history[0]) / len(recent_history)  # Per hour
        
        if rate <= 0:
            return np.inf
        
        # Time to failure
        hours_remaining = margin / rate
        years_remaining = hours_remaining / (365.25 * 24)
        
        return years_remaining
    
    def get_attribution(self) -> Dict[str, float]:
        """
        Get mechanism attribution percentages.
        
        Returns:
            Dictionary {mechanism: percentage}
        """
        if not self.engine.history:
            return {}
        
        # Accumulate damages
        total_nbti = sum(h["rates"]["NBTI"] for h in self.engine.history)
        total_pbti = sum(h["rates"]["PBTI"] for h in self.engine.history)
        total_hci = sum(h["rates"]["HCI"] for h in self.engine.history)
        total_em = sum(h["rates"]["EM"] for h in self.engine.history)
        
        total = total_nbti + total_pbti + total_hci + total_em
        
        if total == 0:
            return {"NBTI": 0, "PBTI": 0, "HCI": 0, "EM": 0}
        
        return {
            "NBTI": 100 * total_nbti / total,
            "PBTI": 100 * total_pbti / total,
            "HCI": 100 * total_hci / total,
            "EM": 100 * total_em / total,
        }
    
    def get_diagnostics(self) -> Dict:
        """Get pipeline diagnostics."""
        return self.engine.get_diagnostics()
    
    def reset(self):
        """Reset pipeline state."""
        self.engine.initialize()