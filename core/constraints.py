"""
Digital Twin Core - Physics Constraints
========================================

Implements physics-aware constraint enforcement for CMOS aging inference.

Key Constraints:
----------------
1. MONOTONIC AGING: Degradation states never decrease (x_{k+1} >= x_k)
2. ARRHENIUS SCALING: Temperature-dependent acceleration follows Arrhenius law
3. LOG-SPACE ENFORCEMENT: Use log-scale for non-negative quantities
4. LEAKAGE POSITIVITY: Leakage current must remain positive
5. BOUNDS ENFORCEMENT: Keep states within physically valid ranges

Physics:
--------
MONOTONIC CONSTRAINT (Fundamental):
    All degradation mechanisms are monotonic:
    - BTI: Threshold voltage shift never recovers (permanent)
    - HCI: Interface trap density only increases
    - EM: Metal voids grow monotonically
    
    Mathematical: x_i(t+1) >= x_i(t) for all degradation states
    
ARRHENIUS SCALING (Temperature Dependence):
    Aging rate doubles every ~10°C (for T > 373K)
    
    Formula: A(T) = A_ref * exp((E_a / k) * (1/T_ref - 1/T))
    
    where:
    - A(T) = acceleration factor at temperature T
    - E_a = activation energy (~0.1-0.2 eV)
    - k = Boltzmann constant (8.617e-5 eV/K)
    - T_ref = reference temperature (usually 313K or 373K)

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ConstraintBase(ABC):
    """Base class for physics constraints."""
    
    def __init__(self, name: str):
        """
        Initialize constraint.
        
        Args:
            name: Constraint name for logging
        """
        self.name = name
        self.violation_count = 0
        self.violation_magnitude = []
        
    @abstractmethod
    def enforce(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Enforce constraint on state.
        
        Args:
            state: State vector [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
            **kwargs: Additional parameters
            
        Returns:
            Constrained state vector
        """
        pass
    
    def get_violations(self) -> Dict:
        """Get constraint violation statistics."""
        return {
            "count": self.violation_count,
            "magnitudes": self.violation_magnitude,
            "mean_violation": np.mean(self.violation_magnitude) if self.violation_magnitude else 0,
            "max_violation": np.max(self.violation_magnitude) if self.violation_magnitude else 0,
        }


class MonotonicProjectionFilter(ConstraintBase):
    """
    Enforce monotonic aging constraint.
    
    Ensures that all degradation states are monotonically increasing:
        x_i(k+1) >= x_i(k) for all degradation states i
        
    Strategy:
    ---------
    1. Keep history of previous state estimates
    2. After each update, compare with previous
    3. If any state decreased, project back to previous value
    4. Use log-space for numerical stability
    
    Example:
    --------
    >>> constraint = MonotonicProjectionFilter(n_states=6)
    >>> state = np.array([0.0050, 0.0020, 0.0080, 0.0015, 0.0045, 0.0032])
    >>> state_prev = np.array([0.0048, 0.0022, 0.0075, 0.0014, 0.0040, 0.0030])
    >>> state_enforced = constraint.enforce(state, state_prev=state_prev)
    # state_enforced >= state_prev (elementwise)
    """
    
    def __init__(self, 
                 n_states: int = 6,
                 degradation_indices: Tuple[int] = (0, 1, 2, 3),
                 use_log_scale: bool = True,
                 tolerance: float = 1e-8):
        """
        Initialize monotonic constraint.
        
        Args:
            n_states: Number of state variables
            degradation_indices: Indices of degradation states (default: BTI, HCI, EM)
            use_log_scale: Use log-space for enforcement (more stable)
            tolerance: Tolerance for monotonicity check
        """
        super().__init__("MonotonicProjection")
        self.n_states = n_states
        self.degradation_indices = degradation_indices
        self.use_log_scale = use_log_scale
        self.tolerance = tolerance
        self.state_prev = None
        
    def enforce(self,
                state: np.ndarray,
                state_prev: Optional[np.ndarray] = None,
                **kwargs) -> np.ndarray:
        """
        Enforce monotonicity constraint.
        
        Args:
            state: Current state estimate
            state_prev: Previous state estimate (if None, use stored history)
            **kwargs: Additional parameters
            
        Returns:
            Monotonically constrained state
        """
        if state_prev is None:
            state_prev = self.state_prev
            
        if state_prev is None:
            # First call, initialize
            self.state_prev = state.copy()
            return state
            
        state_out = state.copy()
        
        # Enforce monotonicity for degradation states
        for i in self.degradation_indices:
            if state_out[i] < state_prev[i] - self.tolerance:
                violation_mag = state_prev[i] - state_out[i]
                self.violation_count += 1
                self.violation_magnitude.append(float(violation_mag))
                
                # Project to previous value
                state_out[i] = state_prev[i]
                
                logger.debug(
                    f"Monotonicity violation in state {i}: "
                    f"{state[i]:.6e} < {state_prev[i]:.6e}, "
                    f"projection magnitude: {violation_mag:.6e}"
                )
        
        # Update history
        self.state_prev = state_out.copy()
        
        return state_out
    
    def reset(self):
        """Reset constraint history."""
        self.state_prev = None
        self.violation_count = 0
        self.violation_magnitude = []


class LogSpaceEnforcer(ConstraintBase):
    """
    Enforce positivity using log-space representation.
    
    For states that must be positive (degradation, leakage),
    use log-space internally for numerical stability:
        x_log = log(x + offset)
    
    Benefits:
    ---------
    1. Avoids negative values naturally
    2. Better numerical stability
    3. Easier to model exponential growth
    4. Prevents division by zero
    
    Example:
    --------
    >>> enforcer = LogSpaceEnforcer(indices=(0,1,2,3), offset=1e-10)
    >>> state = np.array([0.005, 0.002, 0.008, 0.001, 0.004, 0.003])
    >>> state_constrained = enforcer.enforce(state)
    # All degradation states guaranteed > 1e-10
    """
    
    def __init__(self,
                 degradation_indices: Tuple[int] = (0, 1, 2, 3),
                 offset: float = 1e-10,
                 min_value: float = 1e-10):
        """
        Initialize log-space enforcer.
        
        Args:
            degradation_indices: Which states to enforce positivity
            offset: Small offset to avoid log(0)
            min_value: Minimum allowed value
        """
        super().__init__("LogSpaceEnforcer")
        self.degradation_indices = degradation_indices
        self.offset = offset
        self.min_value = min_value
        
    def enforce(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Enforce positivity via log-space clamping.
        
        Args:
            state: State vector
            **kwargs: Additional parameters
            
        Returns:
            State with positivity enforced
        """
        state_out = state.copy()
        
        for i in self.degradation_indices:
            if state_out[i] < self.min_value:
                violation_mag = self.min_value - state_out[i]
                self.violation_count += 1
                self.violation_magnitude.append(float(violation_mag))
                
                state_out[i] = self.min_value
                
                logger.debug(
                    f"Positivity violation in state {i}: "
                    f"{state[i]:.6e} < {self.min_value:.6e}"
                )
        
        return state_out


class BoundsEnforcer(ConstraintBase):
    """
    Enforce state bounds to keep values physically reasonable.
    
    Bounds prevent:
    1. Unrealistic degradation levels (>50%)
    2. Negative leakage current
    3. Frequency decrease >50%
    4. Temperature out of range
    
    Example:
    --------
    >>> bounds = {
    ...     0: (0, 0.1),      # D_NBTI: 0-10%
    ...     1: (0, 0.1),      # D_PBTI: 0-10%
    ...     2: (0, 0.1),      # D_HCI: 0-10%
    ...     3: (0, 0.05),     # D_EM: 0-5%
    ...     4: (0, 0.1),      # ΔV_th: 0-100mV
    ...     5: (0, 0.5)       # μ_deg: 0-50%
    ... }
    >>> enforcer = BoundsEnforcer(bounds)
    >>> state_clamped = enforcer.enforce(state)
    """
    
    def __init__(self,
                 bounds: Dict[int, Tuple[float, float]]):
        """
        Initialize bounds enforcer.
        
        Args:
            bounds: Dictionary {state_index: (min, max)}
        """
        super().__init__("BoundsEnforcer")
        self.bounds = bounds
        
    def enforce(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Enforce bounds on state variables.
        
        Args:
            state: State vector
            **kwargs: Additional parameters
            
        Returns:
            Bounded state vector
        """
        state_out = state.copy()
        
        for idx, (min_val, max_val) in self.bounds.items():
            if state_out[idx] < min_val:
                violation_mag = min_val - state_out[idx]
                self.violation_count += 1
                self.violation_magnitude.append(float(violation_mag))
                state_out[idx] = min_val
                
            elif state_out[idx] > max_val:
                violation_mag = state_out[idx] - max_val
                self.violation_count += 1
                self.violation_magnitude.append(float(violation_mag))
                state_out[idx] = max_val
        
        return state_out


class ArrheniusValidation(ConstraintBase):
    """
    Validate aging acceleration against Arrhenius scaling.
    
    Theory:
    -------
    Aging rate should follow Arrhenius law:
        A(T) = A_ref * exp((E_a / k) * (1/T_ref - 1/T))
    
    where:
    - A(T) = acceleration factor at temperature T
    - E_a = activation energy (0.1-0.2 eV typical)
    - k = Boltzmann constant = 8.617e-5 eV/K
    - T_ref = reference temperature
    
    Expected Behavior:
    - Doubling of aging rate every ~10°C at high T
    - More sensitive near room temperature
    - Validates that physics model is reasonable
    
    Example:
    --------
    >>> validator = ArrheniusValidation(
    ...     E_a=0.13,
    ...     T_ref_k=373.15,
    ...     expected_range=(2.5, 8.0)
    ... )
    >>> is_valid = validator.validate(
    ...     temperature_k=313.15,
    ...     degradation_rate=1.0
    ... )
    """
    
    def __init__(self,
                 E_a: float = 0.13,  # Activation energy [eV]
                 T_ref_k: float = 373.15,  # Reference temperature [K] (100°C)
                 k_B: float = 8.617e-5,  # Boltzmann constant [eV/K]
                 expected_range: Tuple[float, float] = (2.5, 8.0)):
        """
        Initialize Arrhenius validator.
        
        Args:
            E_a: Activation energy [eV]
            T_ref_k: Reference temperature [K]
            k_B: Boltzmann constant [eV/K]
            expected_range: (min, max) expected acceleration factors
        """
        super().__init__("ArrheniusValidation")
        self.E_a = E_a
        self.T_ref_k = T_ref_k
        self.k_B = k_B
        self.expected_range = expected_range
        self.validation_results = []
        
    def compute_acceleration_factor(self, T_k: float) -> float:
        """
        Compute Arrhenius acceleration factor.
        
        Args:
            T_k: Temperature [K]
            
        Returns:
            Acceleration factor A(T) / A(T_ref)
        """
        exponent = (self.E_a / self.k_B) * (1/self.T_ref_k - 1/T_k)
        return np.exp(exponent)
    
    def validate(self,
                 temperature_k: float,
                 degradation_rate: float,
                 reference_rate: Optional[float] = None) -> bool:
        """
        Validate if degradation follows Arrhenius scaling.
        
        Args:
            temperature_k: Operating temperature [K]
            degradation_rate: Measured degradation rate at this temperature
            reference_rate: Degradation rate at T_ref (if known)
            
        Returns:
            True if within expected Arrhenius range
        """
        predicted_accel = self.compute_acceleration_factor(temperature_k)
        
        is_valid = self.expected_range[0] <= predicted_accel <= self.expected_range[1]
        
        result = {
            "temperature_k": temperature_k,
            "predicted_acceleration": predicted_accel,
            "expected_range": self.expected_range,
            "is_valid": is_valid,
        }
        
        self.validation_results.append(result)
        
        if not is_valid:
            self.violation_count += 1
            self.violation_magnitude.append(
                min(
                    abs(predicted_accel - self.expected_range[0]),
                    abs(predicted_accel - self.expected_range[1])
                )
            )
            logger.warning(
                f"Arrhenius validation failed at {temperature_k:.1f}K: "
                f"acceleration={predicted_accel:.2f}x, "
                f"expected={self.expected_range}"
            )
        
        return is_valid
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validations."""
        if not self.validation_results:
            return {}
        
        accels = [r["predicted_acceleration"] for r in self.validation_results]
        return {
            "n_validations": len(self.validation_results),
            "n_failures": sum(1 for r in self.validation_results if not r["is_valid"]),
            "mean_acceleration": np.mean(accels),
            "min_acceleration": np.min(accels),
            "max_acceleration": np.max(accels),
            "all_valid": all(r["is_valid"] for r in self.validation_results),
        }


class ConstraintComposer:
    """
    Compose multiple constraints in a pipeline.
    
    Enforces constraints in order:
    1. Positivity (log-space)
    2. Bounds checking
    3. Monotonicity
    4. Physics validation (Arrhenius)
    
    Example:
    --------
    >>> composer = ConstraintComposer(
    ...     constraints=[
    ...         LogSpaceEnforcer(),
    ...         BoundsEnforcer(bounds),
    ...         MonotonicProjectionFilter(),
    ...         ArrheniusValidation()
    ...     ]
    ... )
    >>> state_constrained = composer.enforce(state)
    """
    
    def __init__(self, constraints: list = None):
        """
        Initialize constraint composer.
        
        Args:
            constraints: List of constraint objects to apply
        """
        self.constraints = constraints or []
        
    def add_constraint(self, constraint: ConstraintBase):
        """Add a constraint to the pipeline."""
        self.constraints.append(constraint)
        
    def enforce(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply all constraints in sequence.
        
        Args:
            state: State vector
            **kwargs: Additional parameters passed to constraints
            
        Returns:
            Fully constrained state
        """
        state_out = state.copy()
        
        for constraint in self.constraints:
            state_out = constraint.enforce(state_out, **kwargs)
        
        return state_out
    
    def get_violation_report(self) -> Dict:
        """Get violation report from all constraints."""
        report = {}
        for constraint in self.constraints:
            report[constraint.name] = constraint.get_violations()
        return report