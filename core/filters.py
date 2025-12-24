"""
Digital Twin Core - Bayesian Filters
=====================================

Implements three Bayesian filters for state estimation:

1. EXTENDED KALMAN FILTER (EKF)
   - First-order Taylor approximation
   - Fast, suitable for weakly nonlinear systems
   - Good for 28nm technology nodes
   - O(n³) complexity, O(n²) memory

2. UNSCENTED KALMAN FILTER (UKF)
   - Deterministic sampling (sigma points)
   - Better handling of nonlinearities
   - Required for 7nm technology nodes
   - O(n³) complexity, but better constants

3. PARTICLE FILTER (PF)
   - Sequential Monte Carlo
   - No Gaussian assumption
   - Gold standard for highly nonlinear systems
   - O(n*m) complexity where m = n_particles

Theory:
-------
All filters implement the predict-update cycle:

PREDICT STEP:
    x̂⁻(k) = f(x̂(k-1), u(k))
    P⁻(k) = ∇f P(k-1) ∇fᵀ + Q  [EKF]
    
UPDATE STEP:
    y(k) = h(x̂⁻(k)) + measurement noise
    Kalman gain K(k) = P⁻(k) ∇hᵀ [∇h P⁻(k) ∇hᵀ + R]⁻¹
    x̂(k) = x̂⁻(k) + K(k) [z(k) - y(k)]
    P(k) = [I - K(k) ∇h] P⁻(k)

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict
from abc import ABC, abstractmethod
import logging
from scipy.linalg import cholesky, solve
from scipy.special import erfc

logger = logging.getLogger(__name__)


class KalmanFilterBase(ABC):
    """Base class for Kalman-type filters."""
    
    def __init__(self, 
                 state_space_model,
                 n_states: int,
                 n_observables: int):
        """
        Initialize Kalman filter.
        
        Args:
            state_space_model: State-space model with f() and h() methods
            n_states: Number of state variables
            n_observables: Number of measurements
        """
        self.ssm = state_space_model
        self.n_states = n_states
        self.n_observables = n_observables
        
        # State estimate
        self.x = np.zeros(n_states)
        self.P = np.eye(n_states)
        
        # Measurement history for diagnostics
        self.residuals = []
        self.likelihoods = []
        self.step_count = 0
        
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict step - propagate state forward in time.
        
        Args:
            u: Control input (if any)
            
        Returns:
            Predicted state estimate
        """
        self.x = self.ssm.f(self.x, u)
        self.P = self.ssm.Q  # Will be updated by subclass
        return self.x
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step - incorporate measurement.
        
        Args:
            z: Measurement vector [f_RO, I_leak, delay]
            
        Returns:
            Updated state estimate
        """
        # Will be implemented by subclass
        pass
    
    def get_diagnostics(self) -> Dict:
        """Get filter diagnostics (residuals, likelihood, etc)."""
        return {
            "step_count": self.step_count,
            "residuals": self.residuals,
            "mean_residual": np.mean(self.residuals) if self.residuals else 0,
            "std_residual": np.std(self.residuals) if len(self.residuals) > 1 else 0,
            "likelihoods": self.likelihoods,
            "mean_likelihood": np.mean(self.likelihoods) if self.likelihoods else 0,
        }


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for mildly nonlinear systems.
    
    Uses first-order Taylor expansion (Jacobian) to linearize:
    f(x) ≈ f(x̂) + ∇f(x̂) (x - x̂)
    h(x) ≈ h(x̂) + ∇h(x̂) (x - x̂)
    
    Pros:
    - Fast computation (O(n³))
    - Suitable for 28nm (weakly nonlinear)
    - Well-understood, mature method
    
    Cons:
    - Can diverge if nonlinearity strong
    - Requires Jacobian computation
    - Underestimates uncertainty
    
    Algorithm:
    ----------
    PREDICT:
    x̂⁻ = f(x̂, u)
    P⁻ = F P F^T + Q where F = ∇f|_{x̂}
    
    UPDATE:
    y = h(x̂⁻)
    H = ∇h|_{x̂⁻}
    S = H P⁻ H^T + R
    K = P⁻ H^T S⁻¹
    x̂ = x̂⁻ + K(z - y)
    P = (I - K H) P⁻
    
    Example:
    --------
    >>> ekf = ExtendedKalmanFilter(
    ...     state_dim=6,
    ...     measurement_dim=3,
    ...     A=np.eye(6),
    ...     H=np.eye(3, 6),
    ...     Q=0.001*np.eye(6),
    ...     R=0.01*np.eye(3)
    ... )
    >>> x_pred = ekf.predict()
    >>> x_updated = ekf.update(measurement)
    """
    
    def __init__(self, state_dim, measurement_dim, A, H, Q, R, x0=None, P0=None):
        """
        Initialize Extended Kalman Filter.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state vector
        measurement_dim : int
            Dimension of measurement vector
        A : ndarray (state_dim, state_dim)
            State transition matrix
        H : ndarray (measurement_dim, state_dim)
            Measurement matrix
        Q : ndarray (state_dim, state_dim)
            Process noise covariance
        R : ndarray (measurement_dim, measurement_dim)
            Measurement noise covariance
        x0 : ndarray, optional
            Initial state estimate
        P0 : ndarray, optional
            Initial error covariance
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        
        # Initialize state and covariance
        self.x = x0 if x0 is not None else np.zeros(state_dim)
        self.P = P0 if P0 is not None else np.eye(state_dim)
        
        # History tracking
        self.residuals = []
        self.likelihoods = []
        self.step_count = 0
        
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        EKF predict step - propagate state forward in time.
        
        Args:
        -----
        u : ndarray, optional
            Control input
            
        Returns:
        --------
        x_pred : ndarray
            Predicted state estimate
        """
        # State prediction: x⁻ = A x
        self.x = self.A @ self.x
        
        # Covariance prediction: P⁻ = A P A^T + Q
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x
        
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        EKF update step - incorporate measurement.
        
        Args:
        -----
        z : ndarray
            Measurement vector
            
        Returns:
        --------
        x_updated : ndarray
            Updated state estimate
        """
        # Measurement prediction
        y = self.H @ self.x
        
        # Innovation (residual)
        innovation = z - y
        self.residuals.append(innovation)
        
        # Innovation covariance: S = H P H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P H^T S⁻¹
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance in EKF update, using pseudo-inverse")
            K = self.P @ self.H.T @ np.linalg.pinv(S)
            
        # State update: x = x⁻ + K(z - y)
        self.x = self.x + K @ innovation
        
        # Covariance update: P = (I - K H) P
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P
        
        # Likelihood for diagnostics
        try:
            likelihood = np.exp(-0.5 * innovation.T @ np.linalg.inv(S) @ innovation)
            self.likelihoods.append(float(likelihood))
        except:
            self.likelihoods.append(0.0)
            
        self.step_count += 1
        return self.x
        
    def get_diagnostics(self) -> Dict:
        """Get filter diagnostics."""
        return {
            "step_count": self.step_count,
            "residuals": self.residuals,
            "mean_residual": np.mean(self.residuals) if self.residuals else 0,
            "std_residual": np.std(self.residuals) if len(self.residuals) > 1 else 0,
            "likelihoods": self.likelihoods,
            "mean_likelihood": np.mean(self.likelihoods) if self.likelihoods else 0,
        }


# ============================================================================
# UNSCENTED KALMAN FILTER - CORRECTED
# ============================================================================

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear systems.
    
    Uses deterministic sampling (sigma points) instead of linearization.
    Better accuracy than EKF for moderately nonlinear systems.
    
    Pros:
    - Handles nonlinearities better than EKF
    - No need for Jacobian computation
    - Same computational complexity as EKF
    
    Cons:
    - More complex implementation
    - Requires tuning of alpha, beta, kappa
    - Still assumes Gaussian distributions
    
    Example:
    --------
    >>> ukf = UnscentedKalmanFilter(
    ...     state_dim=6,
    ...     measurement_dim=3,
    ...     alpha=1e-3,
    ...     beta=2.0,
    ...     kappa=0.0
    ... )
    """
    
    def __init__(self, state_dim, measurement_dim, alpha, beta, kappa, f=None, h=None, Q=None, R=None):
        """
        Initialize Unscented Kalman Filter.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state vector
        measurement_dim : int
            Dimension of measurement vector
        alpha : float
            Spread of sigma points (typically 1e-3)
        beta : float
            Distribution information (typically 2.0)
        kappa : float
            Secondary scaling parameter (typically 0.0)
        f : callable, optional
            State transition function
        h : callable, optional
            Measurement function
        Q : ndarray, optional
            Process noise covariance
        R : ndarray, optional
            Measurement noise covariance
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.f = f
        self.h = h
        self.Q = Q if Q is not None else np.eye(state_dim) * 0.01
        self.R = R if R is not None else np.eye(measurement_dim) * 0.01
        
        # UT parameters
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim
        self.gamma = np.sqrt(state_dim + self.lambda_)
        
        # Weights
        self.w_m = np.zeros(2*state_dim + 1)
        self.w_c = np.zeros(2*state_dim + 1)
        self.w_m[0] = self.lambda_ / (state_dim + self.lambda_)
        self.w_c[0] = self.lambda_ / (state_dim + self.lambda_) + (1 - alpha**2 + beta)
        
        for i in range(1, 2*state_dim + 1):
            self.w_m[i] = 1 / (2 * (state_dim + self.lambda_))
            self.w_c[i] = 1 / (2 * (state_dim + self.lambda_))
            
        # State and covariance
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
        # History
        self.residuals = []
        self.likelihoods = []
        self.step_count = 0
        
    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points using unscented transform.
        
        Args:
        -----
        x : ndarray
            Mean state
        P : ndarray
            Covariance matrix
            
        Returns:
        --------
        sigma_points : ndarray
            Array of sigma points (2n+1 x n)
        """
        try:
            sqrt_P = cholesky(P, lower=True)
        except np.linalg.LinAlgError:
            # Use eigendecomposition if Cholesky fails
            eigvals, eigvecs = np.linalg.eigh(P)
            sqrt_P = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10)))
            
        sigma_points = np.zeros((2*self.state_dim + 1, self.state_dim))
        sigma_points[0] = x
        
        for i in range(self.state_dim):
            sigma_points[i + 1] = x + self.gamma * sqrt_P[:, i]
            sigma_points[self.state_dim + i + 1] = x - self.gamma * sqrt_P[:, i]
            
        return sigma_points
        
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        UKF predict step.
        
        Args:
        -----
        u : ndarray, optional
            Control input
            
        Returns:
        --------
        x_pred : ndarray
            Predicted state
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)
        
        # Propagate through dynamics (use identity if f not provided)
        if self.f is not None:
            sigma_points_pred = np.array([self.f(sp, u) for sp in sigma_points])
        else:
            sigma_points_pred = sigma_points
            
        # Weighted mean
        self.x = np.average(sigma_points_pred, axis=0, weights=self.w_m)
        
        # Weighted covariance
        diffs = sigma_points_pred - self.x
        self.P = np.average(
            np.array([np.outer(d, d) for d in diffs]),
            axis=0,
            weights=self.w_c
        ) + self.Q
        
        return self.x
        
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        UKF update step.
        
        Args:
        -----
        z : ndarray
            Measurement
            
        Returns:
        --------
        x_updated : ndarray
            Updated state
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.x, self.P)
        
        # Propagate through measurement model
        if self.h is not None:
            sigma_measurements = np.array([self.h(sp) for sp in sigma_points])
        else:
            sigma_measurements = sigma_points[:, :self.measurement_dim]
            
        # Weighted mean of measurements
        y_pred = np.average(sigma_measurements, axis=0, weights=self.w_m)
        
        # Innovation
        innovation = z - y_pred
        self.residuals.append(innovation)
        
        # Measurement covariance
        diffs_y = sigma_measurements - y_pred
        S = np.average(
            np.array([np.outer(d, d) for d in diffs_y]),
            axis=0,
            weights=self.w_c
        ) + self.R
        
        # Cross-covariance
        diffs_x = sigma_points - self.x
        P_xy = np.average(
            np.array([np.outer(diffs_x[i], diffs_y[i]) for i in range(len(diffs_x))]),
            axis=0,
            weights=self.w_c
        )
        
        # Kalman gain and update
        try:
            K = P_xy @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance in UKF update, using pseudo-inverse")
            K = P_xy @ np.linalg.pinv(S)
            
        self.x = self.x + K @ innovation
        self.P = self.P - K @ S @ K.T
        
        # Likelihood
        try:
            likelihood = np.exp(-0.5 * innovation.T @ np.linalg.inv(S) @ innovation)
            self.likelihoods.append(float(likelihood))
        except:
            self.likelihoods.append(0.0)
            
        self.step_count += 1
        return self.x
        
    def get_diagnostics(self) -> Dict:
        """Get filter diagnostics."""
        return {
            "step_count": self.step_count,
            "residuals": self.residuals,
            "mean_residual": np.mean(self.residuals) if self.residuals else 0,
            "std_residual": np.std(self.residuals) if len(self.residuals) > 1 else 0,
            "likelihoods": self.likelihoods,
            "mean_likelihood": np.mean(self.likelihoods) if self.likelihoods else 0,
        }


# ============================================================================
# PARTICLE FILTER - CORRECTED
# ============================================================================

class ParticleFilter:
    """
    Particle Filter for highly nonlinear systems.
    
    Sequential Monte Carlo approach with no Gaussian assumption.
    Gold standard for highly nonlinear and non-Gaussian systems.
    
    Pros:
    - Handles arbitrary nonlinearities
    - No Gaussian assumption required
    - Can represent multimodal distributions
    
    Cons:
    - Higher computational cost (O(n*m) with m particles)
    - Susceptible to particle degeneracy
    - Requires effective resampling
    
    Example:
    --------
    >>> pf = ParticleFilter(
    ...     state_dim=6,
    ...     measurement_dim=3,
    ...     n_particles=1000
    ... )
    """
    
    def __init__(self, state_dim, measurement_dim, n_particles, f=None, h=None, Q=None, R=None):
        """
        Initialize Particle Filter.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state vector
        measurement_dim : int
            Dimension of measurement vector
        n_particles : int
            Number of particles
        f : callable, optional
            State transition function
        h : callable, optional
            Measurement function
        Q : ndarray, optional
            Process noise covariance
        R : ndarray, optional
            Measurement noise covariance
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.n_particles = n_particles
        self.f = f
        self.h = h
        self.Q = Q if Q is not None else np.eye(state_dim) * 0.01
        self.R = R if R is not None else np.eye(measurement_dim) * 0.01
        
        # Initialize particles
        self.particles = np.random.randn(n_particles, state_dim)
        self.weights = np.ones(n_particles) / n_particles
        self.x = np.average(self.particles, axis=0, weights=self.weights)
        self.P = np.cov(self.particles.T)
        
        # Parameters
        self.resample_threshold = 0.5
        
        # History
        self.residuals = []
        self.likelihoods = []
        self.step_count = 0
        
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Particle filter predict step.
        
        Args:
        -----
        u : ndarray, optional
            Control input
            
        Returns:
        --------
        x_pred : ndarray
            Mean of particle ensemble
        """
        # Propagate each particle
        if self.f is not None:
            for i in range(self.n_particles):
                self.particles[i] = self.f(self.particles[i], u)
        
        # Add process noise
        noise = np.random.multivariate_normal(np.zeros(self.state_dim), self.Q, self.n_particles)
        self.particles += noise
        
        # Mean of ensemble
        self.x = np.average(self.particles, axis=0, weights=self.weights)
        
        return self.x
        
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Particle filter update step.
        
        Args:
        -----
        z : ndarray
            Measurement
            
        Returns:
        --------
        x_updated : ndarray
            Updated state estimate
        """
        # Compute likelihood for each particle
        likelihoods = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            if self.h is not None:
                y_pred = self.h(self.particles[i])
            else:
                y_pred = self.particles[i, :self.measurement_dim]
                
            innovation = z - y_pred
            
            # Gaussian likelihood
            try:
                R_inv = np.linalg.inv(self.R)
                likelihood = np.exp(-0.5 * innovation.T @ R_inv @ innovation)
            except:
                likelihood = 0.0
                
            likelihoods[i] = likelihood
            
        # Update weights
        self.weights *= likelihoods
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles
            
        # Check for degeneracy
        N_eff = 1.0 / np.sum(self.weights**2)
        
        if N_eff < self.resample_threshold * self.n_particles:
            # Resample
            indices = np.random.choice(
                self.n_particles,
                size=self.n_particles,
                p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
            
        # Mean of ensemble
        self.x = np.average(self.particles, axis=0, weights=self.weights)
        
        # Residual and likelihood tracking
        if self.h is not None:
            y_mean = self.h(self.x)
        else:
            y_mean = self.x[:self.measurement_dim]
            
        innovation = z - y_mean
        self.residuals.append(innovation)
        self.likelihoods.append(float(np.mean(likelihoods)))
        
        self.step_count += 1
        return self.x
        
    def get_diagnostics(self) -> Dict:
        """Get filter diagnostics."""
        return {
            "step_count": self.step_count,
            "residuals": self.residuals,
            "mean_residual": np.mean(self.residuals) if self.residuals else 0,
            "std_residual": np.std(self.residuals) if len(self.residuals) > 1 else 0,
            "likelihoods": self.likelihoods,
            "mean_likelihood": np.mean(self.likelihoods) if self.likelihoods else 0,
            "n_particles": self.n_particles,
        }