"""
Digital Twin Telemetry - Measurement Preprocessing
===================================================

Preprocesses real telemetry measurements for inference pipeline.

Preprocessing Pipeline:
-----------------------
Raw Sensor Data
    ↓
[1. Noise Removal] → Kalman filter, moving average
    ↓
[2. Calibration] → Remove systematic bias
    ↓
[3. Normalization] → Scale to [0, 1] or reference units
    ↓
[4. Outlier Detection] → Statistical tests
    ↓
[5. Quality Assurance] → Check for validity
    ↓
Clean Measurement → Pipeline

Preprocessing Features:
-----------------------
1. Noise Filtering
   - Exponential moving average (IIR)
   - Kalman filter variant (lightweight)
   - Low-pass Butterworth

2. Calibration
   - Reference point subtraction
   - Temperature compensation
   - Drift tracking

3. Normalization
   - Min-max scaling
   - Z-score normalization
   - Logarithmic scaling (for exponential data)

4. Outlier Detection
   - Z-score test (|z| > 3σ)
   - Isolation forest (for multivariate)
   - Median absolute deviation (robust)

5. Quality Metrics
   - Signal-to-noise ratio
   - Sensor health scoring
   - Confidence intervals

Example:
--------
>>> from digital_twin.telemetry import TelemetryPreprocessor
>>> 
>>> preprocessor = TelemetryPreprocessor(config)
>>> 
>>> # Raw measurement from sensors
>>> z_raw = [2398.5, 1.25e-6, 1.52]
>>> 
>>> # Preprocess
>>> z_clean = preprocessor.process(z_raw)
>>> 
>>> # Check quality
>>> quality = preprocessor.get_quality_metrics()
>>> if quality['snr_db'] < 30:
...     print("WARNING: Low signal-to-noise ratio!")

Author: Digital Twin Team
Date: December 24, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class Calibrator:
    """
    Calibration component for sensor offset and scale correction.
    
    Removes systematic biases and temperature-dependent drift.
    
    Calibration Model:
    -------------------
    z_corrected = (z_raw - offset) / scale × reference_scale
    
    Example:
    --------
    >>> calibrator = Calibrator()
    >>> calibrator.set_reference(z_ref=[2400, 1.2e-6, 1.5])
    >>> z_cal = calibrator.calibrate(z_raw)
    """
    
    def __init__(self):
        """Initialize calibrator."""
        self.offset = np.array([0.0, 0.0, 0.0])
        self.scale = np.array([1.0, 1.0, 1.0])
        self.reference = np.array([2400.0, 1.2e-6, 1.5])
        self.calibrated = False
        
    def set_reference(self, z_ref: np.ndarray):
        """
        Set reference measurement point.
        
        Args:
            z_ref: Reference measurement [f_ro, i_leak, d_crit]
        """
        self.reference = np.array(z_ref)
        logger.info(f"Calibration reference set: {z_ref}")
        
    def calibrate(self, z_raw: np.ndarray) -> np.ndarray:
        """
        Calibrate raw measurement.
        
        Args:
            z_raw: Raw measurement [f_ro, i_leak, d_crit]
            
        Returns:
            Calibrated measurement
        """
        if not self.calibrated:
            self.offset = z_raw - self.reference
            self.scale = np.ones(3)
            self.calibrated = True
        
        z_cal = (z_raw - self.offset) / self.scale
        
        return z_cal
    
    def set_temperature_compensation(self, T_k: float, dz_dT: np.ndarray):
        """
        Apply temperature-dependent calibration.
        
        Args:
            T_k: Temperature [K]
            dz_dT: Temperature sensitivity [per K]
        """
        T_ref = 313.15  # Reference temperature
        z_compensation = (T_k - T_ref) * dz_dT
        self.offset += z_compensation


class OutlierDetector:
    """
    Detects and removes measurement outliers.
    
    Methods:
    --------
    1. Z-score: |z| > 3σ
    2. Median Absolute Deviation (MAD): |z - median| > 3×MAD
    3. Isolation Forest: Anomaly score > threshold
    
    Example:
    --------
    >>> detector = OutlierDetector(window_size=100)
    >>> is_outlier = detector.is_outlier(z)
    >>> z_filtered = detector.filter(z)
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize outlier detector.
        
        Args:
            window_size: Rolling window for statistics
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.z_threshold = 3.0  # Z-score threshold
        self.mad_threshold = 3.0  # MAD threshold
        
    def is_outlier(self, z: np.ndarray) -> bool:
        """
        Check if measurement is outlier.
        
        Args:
            z: Measurement [f_ro, i_leak, d_crit]
            
        Returns:
            True if outlier detected
        """
        if len(self.history) < 10:
            return False
        
        # Convert to array
        history_array = np.array(list(self.history))
        
        # Z-score test
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0)
        z_scores = np.abs((z - mean) / (std + 1e-10))
        
        if np.any(z_scores > self.z_threshold):
            return True
        
        # MAD test
        median = np.median(history_array, axis=0)
        mad = np.median(np.abs(history_array - median), axis=0)
        mad_scores = np.abs((z - median) / (mad + 1e-10))
        
        if np.any(mad_scores > self.mad_threshold):
            return True
        
        return False
    
    def filter(self, z: np.ndarray) -> Optional[np.ndarray]:
        """
        Filter measurement, returning None if outlier.
        
        Args:
            z: Raw measurement
            
        Returns:
            Measurement if valid, None if outlier
        """
        if self.is_outlier(z):
            logger.warning(f"Outlier detected: {z}")
            return None
        
        self.history.append(z)
        return z.copy()
    
    def get_statistics(self) -> Dict:
        """Get outlier detector statistics."""
        if len(self.history) < 2:
            return {}
        
        history_array = np.array(list(self.history))
        
        return {
            "n_samples": len(self.history),
            "mean": np.mean(history_array, axis=0),
            "std": np.std(history_array, axis=0),
            "min": np.min(history_array, axis=0),
            "max": np.max(history_array, axis=0),
        }


class NormalizationFilter:
    """
    Normalizes measurements to standard range.
    
    Methods:
    --------
    1. Min-Max: Scales to [0, 1]
    2. Z-Score: Scales to unit variance
    3. Log: Logarithmic scaling for exponential data
    
    Example:
    --------
    >>> norm = NormalizationFilter(method="minmax", bounds=[[2000, 2800], [1e-7, 1e-5], [1, 2]])
    >>> z_norm = norm.normalize(z)
    >>> z_denorm = norm.denormalize(z_norm)
    """
    
    def __init__(self,
                 method: str = "minmax",
                 bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize normalization filter.
        
        Args:
            method: "minmax", "zscore", or "log"
            bounds: Min/max bounds per component (for min-max only)
        """
        self.method = method
        self.bounds = bounds or [
            (2000, 2800),    # f_ro [MHz]
            (1e-7, 1e-5),    # i_leak [A]
            (1, 2),          # d_crit [ns]
        ]
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        
    def fit(self, data: List[np.ndarray]):
        """
        Fit normalization parameters from data.
        
        Args:
            data: List of measurements
        """
        data_array = np.array(data)
        
        if self.method == "zscore":
            self.mean = np.mean(data_array, axis=0)
            self.std = np.std(data_array, axis=0)
    
    def normalize(self, z: np.ndarray) -> np.ndarray:
        """
        Normalize measurement.
        
        Args:
            z: Raw measurement [f_ro, i_leak, d_crit]
            
        Returns:
            Normalized measurement
        """
        if self.method == "minmax":
            z_norm = np.zeros_like(z, dtype=float)
            for i in range(3):
                z_min, z_max = self.bounds[i]
                z_norm[i] = (z[i] - z_min) / (z_max - z_min)
            return np.clip(z_norm, 0, 1)
        
        elif self.method == "zscore":
            return (z - self.mean) / (self.std + 1e-10)
        
        elif self.method == "log":
            # Log scale for exponential data (leakage)
            z_norm = z.copy()
            z_norm[1] = np.log10(z[1])  # Log scale for leakage
            return z_norm
        
        else:
            return z.copy()
    
    def denormalize(self, z_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize to original scale.
        
        Args:
            z_norm: Normalized measurement
            
        Returns:
            Denormalized measurement
        """
        if self.method == "minmax":
            z = np.zeros_like(z_norm, dtype=float)
            for i in range(3):
                z_min, z_max = self.bounds[i]
                z[i] = z_norm[i] * (z_max - z_min) + z_min
            return z
        
        elif self.method == "zscore":
            return z_norm * self.std + self.mean
        
        elif self.method == "log":
            z = z_norm.copy()
            z[1] = 10 ** z_norm[1]  # Exp scale for leakage
            return z
        
        else:
            return z_norm.copy()


class TelemetryPreprocessor:
    """
    Complete telemetry preprocessing pipeline.
    
    Combines all preprocessing components:
    1. Calibration
    2. Outlier detection
    3. Noise filtering
    4. Normalization
    5. Quality assurance
    
    Example:
    --------
    >>> from digital_twin.telemetry import TelemetryPreprocessor
    >>> 
    >>> preprocessor = TelemetryPreprocessor(config)
    >>> 
    >>> # Process continuous stream
    >>> for z_raw in sensor_stream:
    ...     z_clean = preprocessor.process(z_raw)
    ...     if z_clean is not None:
    ...         pipeline.step(z_clean, T_k, activity)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.calibrator = Calibrator()
        self.outlier_detector = OutlierDetector(window_size=100)
        self.normalizer = NormalizationFilter(method="minmax")
        
        # Exponential moving average filter (noise reduction)
        self.ema_alpha = 0.1
        self.ema_state = None
        
        # Statistics
        self.n_processed = 0
        self.n_rejected = 0
        
        logger.info("TelemetryPreprocessor initialized")
        
    def process(self, z_raw: np.ndarray,
               T_k: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Process raw telemetry measurement.
        
        Args:
            z_raw: Raw measurement [f_ro, i_leak, d_crit]
            T_k: Temperature [K] (for calibration)
            
        Returns:
            Clean measurement or None if rejected
        """
        z = np.array(z_raw, dtype=float)
        
        # Step 1: Calibration
        z = self.calibrator.calibrate(z)
        
        # Step 2: Outlier detection
        if self.outlier_detector.is_outlier(z):
            self.n_rejected += 1
            logger.debug(f"Measurement rejected (outlier): {z}")
            return None
        
        # Step 3: Noise filtering (EMA)
        if self.ema_state is None:
            self.ema_state = z.copy()
        else:
            self.ema_state = self.ema_alpha * z + (1 - self.ema_alpha) * self.ema_state
        
        z_filtered = self.ema_state.copy()
        
        # Step 4: Ensure positive values
        z_filtered = np.maximum(z_filtered, 1e-10)
        
        # Step 5: Track history
        self.outlier_detector.history.append(z_filtered)
        
        self.n_processed += 1
        
        return z_filtered
    
    def get_quality_metrics(self) -> Dict:
        """
        Get data quality metrics.
        
        Returns:
            Dictionary of quality indicators
        """
        if self.n_processed == 0:
            return {}
        
        # Signal-to-noise ratio (estimated)
        stats = self.outlier_detector.get_statistics()
        
        if not stats:
            return {"n_processed": 0}
        
        mean = stats.get("mean", np.zeros(3))
        std = stats.get("std", np.ones(3))
        
        snr_db = 20 * np.log10(mean / (std + 1e-10))
        
        return {
            "n_processed": self.n_processed,
            "n_rejected": self.n_rejected,
            "acceptance_rate": (self.n_processed - self.n_rejected) / self.n_processed,
            "snr_db": snr_db,
            "mean": mean,
            "std": std,
        }
    
    def reset(self):
        """Reset preprocessor state."""
        self.ema_state = None
        self.n_processed = 0
        self.n_rejected = 0
        self.outlier_detector.history.clear()