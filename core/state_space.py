"""
Digital Twin Core - State-Space Models
=======================================

Defines state-space models for CMOS aging inference.

State-Space Formulation:
------------------------
x(k+1) = f(x(k), u(k)) + w(k)     [Process model]
z(k)   = h(x(k)) + v(k)            [Measurement model]

where:
- x(k): Hidden state (degradation mechanisms)
- u(k): Control input (temperature, activity)
- z(k): Measurements (frequency, leakage, delay)
- w(k): Process noise N(0, Q)
- v(k): Measurement noise N(0, R)

State Vector (6-dimensional):
-----------------------------
x = [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]

where:
- D_NBTI: NBTI degradation fraction [0, 0.1]
- D_PBTI: PBTI degradation fraction [0, 0.1]
- D_HCI: HCI degradation fraction [0, 0.1]
- D_EM: EM degradation fraction [0, 0.05]
- ΔV_th: Threshold voltage shift [V]
- μ_deg: Mobility degradation fraction [0, 0.5]

Observable Vector (3-dimensional):
----------------------------------
z = [f_RO, I_leak, D_crit]

where:
- f_RO: Ring oscillator frequency [MHz]
- I_leak: Leakage current [A]
- D_crit: Critical path delay [ns]

Physics Models:
---------------
1. BTI (Bias Temperature Instability)
   - NBTI: High V_GS causes threshold voltage increase in PMOS
   - PBTI: High V_GS causes threshold voltage increase in NMOS
   
2. HCI (Hot-Carrier Injection)
   - High V_DS and switching activity cause trap generation
   - Stronger in short-channel devices
   
3. EM (Electromigration)
   - High current density in narrow wires
   - Temperature accelerated
   - Creates voids and resistance increase

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Optional, Callable, Tuple
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class StateSpaceModel(ABC):
    """
    Abstract base class for state-space models.
    
    Defines interface for predict and measurement functions.
    """
    
    def __init__(self,
                 n_states: int = 6,
                 n_observables: int = 3,
                 dt: float = 0.05):
        """
        Initialize state-space model.
        
        Args:
            n_states: Number of hidden states (default: 6 for CMOS)
            n_observables: Number of measurements (default: 3)
            dt: Time step [seconds]
        """
        self.n_states = n_states
        self.n_observables = n_observables
        self.dt = dt
        
        # Noise covariances (to be set by user)
        self.Q = np.eye(n_states) * 1e-6  # Process noise
        self.R = np.eye(n_observables) * 1e-4  # Measurement noise
        
    @abstractmethod
    def f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process model: x(k+1) = f(x(k), u(k))
        
        Args:
            x: State vector
            u: Control input (temperature, activity, voltage)
            
        Returns:
            Next state
        """
        pass
    
    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement model: z(k) = h(x(k))
        
        Maps hidden state to observables:
        - Ring oscillator frequency
        - Leakage current
        - Critical path delay
        
        Args:
            x: State vector
            
        Returns:
            Measurement vector
        """
        pass
    
    def jacobian_f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Jacobian of f: ∇f(x) = ∂f_i/∂x_j
        
        Used by EKF for covariance propagation.
        Default: numerical differentiation
        
        Args:
            x: State vector
            u: Control input
            
        Returns:
            Jacobian matrix (n_states x n_states)
        """
        eps = 1e-6
        F = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            F[:, i] = (self.f(x_plus, u) - self.f(x_minus, u)) / (2 * eps)
        
        return F
    
    def jacobian_h(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h: ∇h(x) = ∂h_i/∂x_j
        
        Used by EKF/UKF for measurement update.
        Default: numerical differentiation
        
        Args:
            x: State vector
            
        Returns:
            Jacobian matrix (n_observables x n_states)
        """
        eps = 1e-6
        H = np.zeros((self.n_observables, self.n_states))
        
        for i in range(self.n_states):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            H[:, i] = (self.h(x_plus) - self.h(x_minus)) / (2 * eps)
        
        return H


class LinearSSM(StateSpaceModel):
    """
    Linear State-Space Model.
    
    x(k+1) = A x(k) + B u(k) + w(k)
    z(k)   = C x(k) + v(k)
    
    Suitable for weakly nonlinear systems or linearization around
    operating point.
    
    Example:
    --------
    >>> A = np.eye(6) + 0.01 * np.random.randn(6, 6)
    >>> B = 0.01 * np.random.randn(6, 1)
    >>> C = np.random.randn(3, 6)
    >>> ssm = LinearSSM(A=A, B=B, C=C)
    >>> x_next = ssm.f(x, u)
    >>> z = ssm.h(x)
    """
    
    def __init__(self,
                 A: np.ndarray,
                 C: np.ndarray,
                 B: Optional[np.ndarray] = None,
                 n_states: int = 6,
                 n_observables: int = 3,
                 dt: float = 0.05):
        """
        Initialize linear state-space model.
        
        Args:
            A: State transition matrix (n_states x n_states)
            C: Output matrix (n_observables x n_states)
            B: Input matrix (n_states x n_inputs), optional
            n_states: Number of states
            n_observables: Number of observables
            dt: Time step
        """
        super().__init__(n_states, n_observables, dt)
        
        self.A = A
        self.C = C
        self.B = B if B is not None else np.zeros((n_states, 1))
        
    def f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear state transition."""
        if u is None:
            u = np.zeros((self.B.shape[1],))
        
        return self.A @ x + self.B @ u
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """Linear measurement."""
        return self.C @ x
    
    def jacobian_f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Jacobian of linear system is just A."""
        return self.A
    
    def jacobian_h(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of linear measurement is just C."""
        return self.C


class NonlinearSSM(StateSpaceModel):
    """
    Nonlinear State-Space Model for CMOS aging.
    
    Physics-based model implementing degradation mechanisms:
    - BTI (Bias Temperature Instability)
    - HCI (Hot-Carrier Injection)
    - EM (Electromigration)
    
    State: x = [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
    
    Measurement: z = [f_RO, I_leak, D_crit]
    
    Physics:
    --------
    BTI Degradation Rate:
        dD_BTI/dt = A * exp(-E_a/kT) * (V_GS)^n * t^m
    
    HCI Degradation Rate:
        dD_HCI/dt = A * exp(-E_a/kT) * (V_DS)^p * α * t^m
    
    EM Degradation Rate:
        dD_EM/dt = A * exp(-E_a/kT) * (J/J_crit)^q * t^m
    
    Frequency Impact:
        f_RO = f_0 * (1 - ΔV_th / V_t0)^(-1) * sqrt(μ_0 / μ)
    
    Leakage Impact:
        I_leak = I_0 * exp(ΔV_th / nV_T) * sqrt(μ_0 / μ)
    
    Example:
    --------
    >>> ssm = NonlinearSSM(config)
    >>> ssm.set_physics_parameters(nbti_coeff=1.1e-5, ...)
    >>> x_next = ssm.f(x, u)
    >>> z = ssm.h(x)
    """
    
    def __init__(self,
                 n_states: int = 6,
                 n_observables: int = 3,
                 dt: float = 0.05):
        """
        Initialize nonlinear CMOS aging model.
        
        Args:
            n_states: Number of states (6 for CMOS)
            n_observables: Number of measurements (3)
            dt: Time step [seconds]
        """
        super().__init__(n_states, n_observables, dt)
        
        # Physics parameters (set to nominal, override with config)
        self.T_k = 313.15  # Temperature [K]
        self.V_dd = 0.9    # Supply voltage [V]
        self.activity = 0.4  # Switching activity
        self.V_th_nominal = 0.45  # Nominal threshold voltage [V]
        self.f_nominal = 2400.0  # Nominal frequency [MHz]
        
        # Degradation coefficients
        self.nbti_coeff = 1.1e-5
        self.pbti_coeff = 1.0e-6
        self.hci_coeff = 0.8e-7
        self.em_coeff = 1.2e-8
        
        # Boltzmann constant and other constants
        self.k_B = 8.617e-5  # eV/K
        self.q_e = 1.602e-19  # Elementary charge [C]
        
    def set_physics_parameters(self, **kwargs):
        """
        Set physics model parameters.
        
        Args:
            nbti_coeff: BTI prefactor
            pbti_coeff: PBTI prefactor
            hci_coeff: HCI prefactor
            em_coeff: EM prefactor
            T_k: Temperature [K]
            V_dd: Supply voltage [V]
            activity: Switching activity
            etc.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Nonlinear state transition (degradation dynamics).
        
        Args:
            x: State [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
            u: Control input [T_k, V_dd, activity] (optional)
            
        Returns:
            Next state x(k+1)
        """
        if u is not None:
            self.T_k = u[0] if len(u) > 0 else self.T_k
            self.V_dd = u[1] if len(u) > 1 else self.V_dd
            self.activity = u[2] if len(u) > 2 else self.activity
        
        x_next = x.copy()
        
        # Temperature scaling
        exp_factor = np.exp((self.nbti_coeff / self.k_B) * 
                            (1/373.15 - 1/self.T_k))
        
        # Voltage scaling
        V_stress = self.V_dd
        
        # BTI (NBTI) - exponential time dependence
        nbti_rate = self.nbti_coeff * exp_factor * (V_stress ** 0.25) * (self.dt ** 0.5)
        x_next[0] = min(x[0] + nbti_rate, 0.1)  # Cap at 10%
        
        # BTI (PBTI) - lower rate
        pbti_rate = self.pbti_coeff * exp_factor * (V_stress ** 0.25) * (self.dt ** 0.5)
        x_next[1] = min(x[1] + pbti_rate, 0.1)
        
        # HCI - depends on activity and drain voltage
        hci_rate = self.hci_coeff * exp_factor * (V_stress ** 1.5) * self.activity * (self.dt ** 0.5)
        x_next[2] = min(x[2] + hci_rate, 0.1)
        
        # EM - depends on current density and temperature
        em_rate = self.em_coeff * exp_factor * (V_stress ** 2.0) * self.activity * (self.dt ** 0.5)
        x_next[3] = min(x[3] + em_rate, 0.05)  # Cap at 5%
        
        # ΔV_th: Accumulation from BTI and HCI
        # BTI contribution dominates
        dVth_bti = 0.18 * (x_next[0] - x[0])  # 0.18 mV per unit BTI (28nm)
        dVth_hci = 0.12 * (x_next[2] - x[2])  # 0.12 mV per unit HCI
        x_next[4] = x[4] + dVth_bti + dVth_hci
        
        # μ_deg: Mobility degradation from BTI and HCI
        mu_deg_bti = 0.15 * (x_next[0] - x[0])
        mu_deg_hci = 0.10 * (x_next[2] - x[2])
        x_next[5] = x[5] + mu_deg_bti + mu_deg_hci
        
        return x_next
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement model: map state to observables.
        
        Observable: z = [f_RO, I_leak, D_crit]
        
        Physics:
        --------
        Ring Oscillator Frequency:
            f_RO = f_0 * [1 - ΔV_th/(2*V_t0)]^(-0.5) * sqrt(μ_0/μ)
        
        Leakage Current:
            I_leak = I_0 * exp(ΔV_th / (nV_T)) * sqrt(μ_0/μ)
        
        Critical Path Delay:
            D_crit = D_0 * [1 - ΔV_th/(V_t0)]^(-1) / sqrt(μ_0/μ)
        
        Args:
            x: State [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
            
        Returns:
            Measurement z = [f_RO, I_leak, D_crit]
        """
        z = np.zeros(self.n_observables)
        
        # Extract state components
        delta_vth = x[4]  # V_th shift [V]
        mu_deg = x[5]     # Mobility degradation [fraction]
        
        # Constants
        V_t0 = self.V_th_nominal
        nV_T = 0.026  # Thermal voltage at 300K
        
        # 1. Ring Oscillator Frequency (affected by V_th and mobility)
        # Decrease from higher V_th, decrease from lower mobility
        f_degrade_vth = np.sqrt(1 - delta_vth / (2 * V_t0))
        f_degrade_mu = np.sqrt(1 - mu_deg)
        z[0] = self.f_nominal * f_degrade_vth * f_degrade_mu
        z[0] = max(z[0], self.f_nominal * 0.5)  # Floor at 50%
        
        # 2. Leakage Current (increases exponentially with V_th decrease)
        # Also increases with mobility degradation
        # I_leak = I_0 * exp(ΔV_th / (2*nV_T)) * (1 + mu_deg)
        I_0 = 1.5e-6  # Nominal leakage [A]
        leakage_exp = np.exp(delta_vth / (2 * nV_T))
        leakage_mu = 1.0 + 2.0 * mu_deg  # Mobility effect on leakage
        z[1] = I_0 * leakage_exp * leakage_mu
        
        # 3. Critical Path Delay (increases from V_th and mobility effects)
        # D_crit = D_0 / [1 - ΔV_th/V_t0]^0.5 / sqrt(1 - mu_deg)
        D_0 = 2.5  # Nominal delay [ns]
        delay_vth = 1.0 / np.sqrt(1 - delta_vth / V_t0)
        delay_mu = 1.0 / np.sqrt(1 - mu_deg)
        z[2] = D_0 * delay_vth * delay_mu
        z[2] = min(z[2], D_0 * 2.0)  # Cap at 200%
        
        return z
    
    def jacobian_f(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Jacobian of f numerically (can be optimized with analytics).
        
        Args:
            x: State vector
            u: Control input
            
        Returns:
            Jacobian matrix
        """
        if u is not None:
            self.T_k = u[0] if len(u) > 0 else self.T_k
            self.V_dd = u[1] if len(u) > 1 else self.V_dd
            self.activity = u[2] if len(u) > 2 else self.activity
        
        eps = 1e-6
        F = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            F[:, i] = (self.f(x_plus, u) - self.f(x_minus, u)) / (2 * eps)
        
        return F
    
    def jacobian_h(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of h numerically.
        
        Args:
            x: State vector
            
        Returns:
            Jacobian matrix
        """
        eps = 1e-8
        H = np.zeros((self.n_observables, self.n_states))
        
        for i in range(self.n_states):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            H[:, i] = (self.h(x_plus) - self.h(x_minus)) / (2 * eps)
        
        return H