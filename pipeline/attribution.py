"""
Digital Twin Pipeline - Attribution & Lifetime Analysis
========================================================

Analyzes mechanism breakdown and predicts lifetime.

Attribution Analysis:
---------------------
Decomposes total degradation into mechanism contributions:
- NBTI (Negative Bias Temperature Instability): Gate oxide
- PBTI (Positive Bias Temperature Instability): Lower magnitude
- HCI (Hot-Carrier Injection): Activity-dependent
- EM (Electromigration): Current-density dependent

Lifetime Prediction:
--------------------
Projects current degradation rate to time-of-failure:

MTF = Margin / (dX/dt)

where:
- Margin = V_th_limit - V_th_current
- dX/dt = degradation rate
- MTF = Mean Time To Failure

Margin Analysis:
----------------
Tracks remaining margin as degradation progresses:
- V_th margin: threshold voltage headroom
- Frequency margin: maximum clock frequency headroom
- Power margin: power budget remaining

Example:
--------
>>> analyzer = AttributionAnalyzer(state_history)
>>> attribution = analyzer.get_attribution()
>>> lifetime = LifetimePredictor(config).predict_mtf(state, rate)

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AttributionAnalyzer:
    """
    Analyzes mechanism attribution from state history.
    
    Maps accumulated state to mechanism percentages.
    
    Attribution Model:
    ------------------
    State components map to mechanisms:
    - D_NBTI → NBTI contribution
    - D_PBTI → PBTI contribution
    - D_HCI → HCI contribution
    - D_EM → EM contribution
    
    Example:
    --------
    >>> analyzer = AttributionAnalyzer(history)
    >>> attribution = analyzer.get_attribution()
    >>> breakdown = analyzer.get_mechanism_breakdown()
    """
    
    def __init__(self, history: List[Dict]):
        """
        Initialize attribution analyzer.
        
        Args:
            history: List of state snapshots with rates
        """
        self.history = history
        
    def get_attribution(self) -> Dict[str, float]:
        """
        Compute mechanism attribution as percentages.
        
        Returns:
            Dictionary {mechanism: percentage [0-100]}
        """
        if not self.history:
            return {}
        
        # Sum rates over history
        attribution = defaultdict(float)
        
        for snapshot in self.history:
            if "rates" in snapshot:
                for mechanism, rate in snapshot["rates"].items():
                    attribution[mechanism] += rate
        
        # Normalize to percentages
        total = sum(attribution.values())
        
        if total == 0:
            return {mech: 0.0 for mech in attribution}
        
        return {
            mech: 100 * value / total
            for mech, value in attribution.items()
        }
    
    def get_mechanism_breakdown(self) -> Dict[str, Dict]:
        """
        Get detailed breakdown by mechanism.
        
        Returns:
            Dictionary with statistics per mechanism
        """
        if not self.history:
            return {}
        
        breakdown = {}
        
        for mechanism in ["NBTI", "PBTI", "HCI", "EM"]:
            rates = [h.get("rates", {}).get(mechanism, 0) for h in self.history]
            
            breakdown[mechanism] = {
                "total_accumulated": sum(rates),
                "mean_rate": np.mean(rates),
                "max_rate": np.max(rates),
                "min_rate": np.min(rates),
                "std_dev": np.std(rates),
                "percentage": 100 * sum(rates) / sum(
                    h.get("rates", {}).get(m, 0)
                    for h in self.history
                    for m in ["NBTI", "PBTI", "HCI", "EM"]
                ) if sum(
                    h.get("rates", {}).get(m, 0)
                    for h in self.history
                    for m in ["NBTI", "PBTI", "HCI", "EM"]
                ) > 0 else 0,
            }
        
        return breakdown
    
    def get_dominant_mechanism(self, window: int = 10) -> str:
        """
        Get dominant mechanism in recent window.
        
        Args:
            window: Number of recent samples
            
        Returns:
            Dominant mechanism name
        """
        if not self.history:
            return "UNKNOWN"
        
        recent = self.history[-window:] if len(self.history) >= window else self.history
        
        mechanisms = ["NBTI", "PBTI", "HCI", "EM"]
        totals = {mech: sum(h.get("rates", {}).get(mech, 0) for h in recent) 
                  for mech in mechanisms}
        
        return max(totals, key=totals.get)
    
    def get_trends(self, window: int = 50) -> Dict[str, float]:
        """
        Get trend of degradation rates.
        
        Args:
            window: Window size for trend analysis
            
        Returns:
            Dictionary {mechanism: rate_trend}
        """
        if len(self.history) < window:
            return {}
        
        trends = {}
        mechanisms = ["NBTI", "PBTI", "HCI", "EM"]
        
        recent = self.history[-window:]
        
        for mechanism in mechanisms:
            rates = [h.get("rates", {}).get(mechanism, 0) for h in recent]
            
            # Linear fit: rate vs time
            if len(rates) > 1:
                t = np.arange(len(rates))
                coeffs = np.polyfit(t, rates, 1)
                trend_slope = coeffs[0]  # Positive: increasing, Negative: decreasing
            else:
                trend_slope = 0
            
            trends[mechanism] = trend_slope
        
        return trends


class LifetimePredictor:
    """
    Predicts remaining lifetime and MTTF.
    
    Physics-based lifetime projection using:
    - Current degradation state
    - Instantaneous degradation rate
    - Failure thresholds
    
    Lifetime Models:
    ----------------
    1. Linear extrapolation: MTF = Margin / Rate
    2. Lognormal (EM): Weibull distribution
    3. Time-dependent: Rate changes with temperature
    
    Example:
    --------
    >>> predictor = LifetimePredictor(config)
    >>> 
    >>> # Remaining lifetime
    >>> lifetime_years = predictor.predict_remaining_lifetime(
    ...     state=[...],
    ...     rate=1e-5,
    ...     margin=0.1
    ... )
    >>> 
    >>> # MTTF calculation
    >>> mttf = predictor.compute_mttf(J_a=1.2e6, T_k=373.15)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize lifetime predictor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Failure thresholds
        self.v_th_limit = config.get("validation", {}).get("v_th_limit_mv", 100) / 1000
        self.freq_degradation_limit = config.get("validation", {}).get("frequency_margin_percent", 20) / 100
        self.delay_limit = config.get("validation", {}).get("delay_limit_ns", 10)
        
        # Tech node specific
        self.tech_node = config.get("technology", {}).get("node", "28nm")
        
    def predict_remaining_lifetime(self,
                                   state: np.ndarray,
                                   rate: float,
                                   margin: Optional[float] = None) -> float:
        """
        Predict remaining lifetime.
        
        MTF = Margin / Rate
        
        Args:
            state: Current state vector [6D]
            rate: Current degradation rate [1/hour or equivalent]
            margin: Failure margin [V or fraction]
            
        Returns:
            Remaining lifetime [years]
        """
        if margin is None:
            margin = self.v_th_limit - state[4]
        
        # Clamp to [0, ∞)
        margin = max(0, margin)
        
        if rate <= 0:
            return np.inf
        
        # Time to failure
        hours_remaining = margin / rate
        years_remaining = hours_remaining / (365.25 * 24)
        
        return years_remaining
    
    def compute_mttf_em(self,
                        J_a: float,
                        T_k: float = 373.15,
                        J_crit: float = 1.0e6) -> float:
        """
        Compute MTTF for electromigration using Black's law.
        
        MTF = τ_0 × (J_0/J)^q × exp(E_a/kT)
        
        Args:
            J_a: Current density [A/cm²]
            T_k: Temperature [K]
            J_crit: Critical current density [A/cm²]
            
        Returns:
            Mean time to failure [hours]
        """
        # Black's law parameters
        E_a = 0.09  # Activation energy [eV]
        k_B = 8.617e-5
        q = 2.0  # Current exponent
        mtf_ref = 1000  # Reference MTF at 1e6 A/cm², 373K [hours]
        
        # Temperature factor
        arr_factor = np.exp((E_a / k_B) * (1/373.15 - 1/T_k))
        
        # Current factor
        if J_a < J_crit:
            return np.inf
        
        J_factor = (J_crit / J_a) ** q
        
        # MTTF
        mttf = mtf_ref * J_factor * arr_factor
        
        return mttf
    
    def get_margin_remaining(self, state: np.ndarray) -> Dict[str, float]:
        """
        Compute remaining margins.
        
        Args:
            state: Current state [6D]
            
        Returns:
            Dictionary {margin_type: remaining_margin}
        """
        v_th_current = state[4]
        freq_deg_current = state[5]  # Approximate from mobility
        
        return {
            "v_th_margin_mv": max(0, (self.v_th_limit - v_th_current) * 1000),
            "v_th_margin_percent": 100 * max(0, self.v_th_limit - v_th_current) / self.v_th_limit,
            "frequency_margin_percent": max(0, self.freq_degradation_limit - freq_deg_current) * 100,
            "delay_margin_ns": max(0, self.delay_limit - state[4] * 1000),  # Approximate
        }
    
    def is_within_margin(self, state: np.ndarray) -> bool:
        """
        Check if state is within acceptable margins.
        
        Args:
            state: Current state [6D]
            
        Returns:
            True if within margins, False otherwise
        """
        margins = self.get_margin_remaining(state)
        
        return all(m >= 0 for m in margins.values())
    
    def predict_time_to_margin(self,
                              state: np.ndarray,
                              rate: float) -> float:
        """
        Predict time until margin is exceeded.
        
        Args:
            state: Current state [6D]
            rate: Current degradation rate [1/hour]
            
        Returns:
            Hours until margin exceeded
        """
        v_th_current = state[4]
        margin = self.v_th_limit - v_th_current
        
        if margin <= 0:
            return 0.0
        
        if rate <= 0:
            return np.inf
        
        return margin / rate


class MechanismBreakdown:
    """
    Detailed breakdown of degradation contributions.
    
    Tracks how each mechanism contributes to observable effects:
    - V_th shift
    - Frequency decrease
    - Delay increase
    - Leakage increase
    
    Example:
    --------
    >>> breakdown = MechanismBreakdown()
    >>> 
    >>> # Add mechanism contributions
    >>> breakdown.add_nbti_effect(v_th_shift=0.005, f_decrease=0.02)
    >>> breakdown.add_hci_effect(v_th_shift=0.003, f_decrease=0.03)
    >>> 
    >>> # Get totals
    >>> effects = breakdown.get_total_effects()
    """
    
    def __init__(self):
        """Initialize mechanism breakdown tracker."""
        self.nbti_effects = {
            "v_th_shift": 0.0,
            "frequency_decrease": 0.0,
            "delay_increase": 0.0,
            "leakage_increase": 0.0,
        }
        self.pbti_effects = self.nbti_effects.copy()
        self.hci_effects = self.nbti_effects.copy()
        self.em_effects = {
            "v_th_shift": 0.0,
            "frequency_decrease": 0.0,
            "delay_increase": 0.0,
            "leakage_increase": 0.0,
            "resistance_increase": 0.0,
        }
    
    def add_nbti_effect(self,
                       v_th_shift: float = 0.0,
                       frequency_decrease: float = 0.0,
                       delay_increase: float = 0.0,
                       leakage_increase: float = 0.0):
        """Add NBTI effect contributions."""
        self.nbti_effects["v_th_shift"] += v_th_shift
        self.nbti_effects["frequency_decrease"] += frequency_decrease
        self.nbti_effects["delay_increase"] += delay_increase
        self.nbti_effects["leakage_increase"] += leakage_increase
    
    def add_hci_effect(self,
                      v_th_shift: float = 0.0,
                      frequency_decrease: float = 0.0,
                      delay_increase: float = 0.0,
                      leakage_increase: float = 0.0):
        """Add HCI effect contributions."""
        self.hci_effects["v_th_shift"] += v_th_shift
        self.hci_effects["frequency_decrease"] += frequency_decrease
        self.hci_effects["delay_increase"] += delay_increase
        self.hci_effects["leakage_increase"] += leakage_increase
    
    def add_em_effect(self,
                     delay_increase: float = 0.0,
                     resistance_increase: float = 0.0):
        """Add EM effect contributions."""
        self.em_effects["delay_increase"] += delay_increase
        self.em_effects["resistance_increase"] += resistance_increase
    
    def get_total_effects(self) -> Dict[str, float]:
        """
        Get total effects from all mechanisms.
        
        Returns:
            Dictionary of cumulative effects
        """
        return {
            "v_th_shift": (self.nbti_effects["v_th_shift"] +
                          self.pbti_effects["v_th_shift"] +
                          self.hci_effects["v_th_shift"]),
            "frequency_decrease": (self.nbti_effects["frequency_decrease"] +
                                  self.pbti_effects["frequency_decrease"] +
                                  self.hci_effects["frequency_decrease"]),
            "delay_increase": (self.nbti_effects["delay_increase"] +
                              self.pbti_effects["delay_increase"] +
                              self.hci_effects["delay_increase"] +
                              self.em_effects["delay_increase"]),
            "leakage_increase": (self.nbti_effects["leakage_increase"] +
                                self.pbti_effects["leakage_increase"] +
                                self.hci_effects["leakage_increase"]),
            "resistance_increase": self.em_effects["resistance_increase"],
        }
    
    def get_mechanism_attribution(self) -> Dict[str, Dict[str, float]]:
        """
        Get attribution by mechanism.
        
        Returns:
            Dictionary {mechanism: effects}
        """
        return {
            "NBTI": self.nbti_effects,
            "PBTI": self.pbti_effects,
            "HCI": self.hci_effects,
            "EM": self.em_effects,
        }
    
    def print_summary(self):
        """Print breakdown summary."""
        print("\n=== DEGRADATION BREAKDOWN ===")
        print("\nNBTI Effects:")
        for effect, value in self.nbti_effects.items():
            print(f"  {effect}: {value:.6f}")
        
        print("\nHCI Effects:")
        for effect, value in self.hci_effects.items():
            print(f"  {effect}: {value:.6f}")
        
        print("\nEM Effects:")
        for effect, value in self.em_effects.items():
            print(f"  {effect}: {value:.6f}")
        
        print("\nTotal Effects:")
        totals = self.get_total_effects()
        for effect, value in totals.items():
            print(f"  {effect}: {value:.6f}")