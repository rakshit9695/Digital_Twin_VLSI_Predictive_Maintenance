"""
Digital Twin Utils - Logging & Diagnostics
===========================================

Comprehensive logging and diagnostic utilities.

Features:
---------
1. Logging Setup
   - Configure logging handlers
   - Set log levels
   - Support for file and console output
   - Structured logging format

2. Module Loggers
   - Get loggers for specific modules
   - Consistent naming conventions
   - Hierarchical logger organization

3. State Logging
   - Log system state snapshots
   - Log inference steps
   - Diagnostic summaries
   - Performance metrics

4. Formatting
   - Structured log format
   - Timestamps
   - Module names
   - Log levels

Log Format:
-----------
[2025-12-24 12:30:45.123] [INFO] [digital_twin.pipeline] Message here
[TIMESTAMP] [LEVEL] [MODULE] Message

Log Levels:
-----------
DEBUG:    Detailed information for debugging
INFO:     General informational messages
WARNING:  Warning messages for potential issues
ERROR:    Error messages for failures
CRITICAL: Critical failures requiring attention

Example:
--------
>>> from digital_twin.utils import setup_logging, get_logger
>>> 
>>> # Initialize logging
>>> setup_logging("logs/", level="INFO")
>>> 
>>> # Get logger
>>> logger = get_logger(__name__)
>>> 
>>> # Log messages
>>> logger.info("Pipeline initialized")
>>> logger.debug(f"State: {state}")
>>> logger.warning(f"Low SNR: {snr} dB")
>>> logger.error(f"Filter divergence at step {step}")

Author: Digital Twin Team
Date: December 24, 2025
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import numpy as np


# Global logger instance
_loggers = {}


class StructuredFormatter(logging.Formatter):
    """Structured logging formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structure.
        
        Args:
            record: Log record
            
        Returns:
            Formatted string
        """
        timestamp = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        
        return (
            f"[{timestamp}] [{record.levelname:8}] "
            f"[{record.name}] {record.getMessage()}"
        )


def setup_logging(log_dir: str = "logs",
                 level: str = "INFO",
                 console_output: bool = True,
                 file_output: bool = True) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console output
        file_output: Enable file output
        
    Example:
        >>> from digital_twin.utils import setup_logging
        >>> 
        >>> setup_logging("logs/", level="INFO")
        >>> logger = get_logger(__name__)
        >>> logger.info("System started")
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = StructuredFormatter()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"digital_twin_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup completion
    root_logger.info(f"Logging configured: level={level}, dir={log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for a module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    
    return _loggers[name]


def log_state(state: np.ndarray,
             step: int,
             diagnostics: Optional[Dict[str, Any]] = None) -> None:
    """
    Log system state snapshot.
    
    Args:
        state: State vector [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
        step: Step number
        diagnostics: Optional diagnostic information
        
    Example:
        >>> logger = get_logger(__name__)
        >>> log_state(state, step=100, diagnostics={"temp": 373.15})
    """
    logger = get_logger(__name__)
    
    logger.info(
        f"Step {step}: "
        f"V_th={state[4]*1000:.2f}mV, "
        f"μ={state[5]:.4f}, "
        f"NBTI={state[0]:.6f}, "
        f"HCI={state[2]:.6f}"
    )
    
    if diagnostics:
        for key, value in diagnostics.items():
            if isinstance(value, float):
                logger.debug(f"  {key}: {value:.4f}")
            else:
                logger.debug(f"  {key}: {value}")


def log_step(step: int,
            measurement: np.ndarray,
            state: np.ndarray,
            temperature: float,
            activity: float) -> None:
    """
    Log a single inference step.
    
    Args:
        step: Step number
        measurement: Measurement vector [f_ro, i_leak, d_crit]
        state: State estimate [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]
        temperature: Temperature [K]
        activity: Activity level [0, 1]
        
    Example:
        >>> log_step(
        ...     step=100,
        ...     measurement=np.array([2400, 1.2e-6, 1.5]),
        ...     state=state_estimate,
        ...     temperature=373.15,
        ...     activity=0.4,
        ... )
    """
    logger = get_logger(__name__)
    
    logger.debug(
        f"Step {step}: "
        f"f_RO={measurement[0]:.1f}MHz, "
        f"I_leak={measurement[1]:.2e}A, "
        f"D_crit={measurement[2]:.3f}ns, "
        f"T={temperature-273.15:.1f}°C, "
        f"activity={activity:.2f}"
    )


def log_statistics(stats: Dict[str, Any]) -> None:
    """
    Log statistics summary.
    
    Args:
        stats: Statistics dictionary
        
    Example:
        >>> stats = sim.get_statistics()
        >>> log_statistics(stats)
    """
    logger = get_logger(__name__)
    
    logger.info("=== Statistics Summary ===")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")


def log_error(error: Exception,
             context: str = "") -> None:
    """
    Log an error with context.
    
    Args:
        error: Exception that occurred
        context: Context information
        
    Example:
        >>> try:
        ...     pipeline.step(z, T_k, activity)
        ... except Exception as e:
        ...     log_error(e, context="Inference step failed")
    """
    logger = get_logger(__name__)
    
    if context:
        logger.error(f"{context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")
    
    logger.debug("", exc_info=True)


def log_warning(message: str,
               severity: str = "MEDIUM") -> None:
    """
    Log a warning message.
    
    Args:
        message: Warning message
        severity: Severity level (LOW, MEDIUM, HIGH)
        
    Example:
        >>> log_warning("Filter uncertainty increasing", severity="MEDIUM")
    """
    logger = get_logger(__name__)
    
    logger.warning(f"[{severity}] {message}")


def create_diagnostic_report(pipeline_history: list,
                            stats: Dict[str, Any]) -> str:
    """
    Create diagnostic report from simulation history.
    
    Args:
        pipeline_history: History from pipeline
        stats: Statistics dictionary
        
    Returns:
        Diagnostic report string
    """
    report = []
    report.append("=" * 60)
    report.append("DIAGNOSTIC REPORT")
    report.append("=" * 60)
    
    # Basic statistics
    report.append("\n[Simulation Statistics]")
    for key, value in stats.items():
        if isinstance(value, float):
            report.append(f"  {key}: {value:.4f}")
        else:
            report.append(f"  {key}: {value}")
    
    # History analysis
    if pipeline_history:
        report.append("\n[History Analysis]")
        report.append(f"  Total steps: {len(pipeline_history)}")
        
        # Temperature statistics
        temps = [h.get("temperature", 0) for h in pipeline_history]
        if temps:
            report.append(f"  Temperature range: {min(temps):.1f} - {max(temps):.1f} K")
        
        # Activity statistics
        activities = [h.get("activity", 0) for h in pipeline_history]
        if activities:
            report.append(f"  Activity range: {min(activities):.2f} - {max(activities):.2f}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


def save_diagnostic_report(report: str,
                          output_path: str) -> None:
    """
    Save diagnostic report to file.
    
    Args:
        report: Report content
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger = get_logger(__name__)
    logger.info(f"Saved diagnostic report to {output_path}")


class LogLevel:
    """Log level constants."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def set_log_level(name: str,
                 level: str) -> None:
    """
    Set log level for specific logger.
    
    Args:
        name: Logger name
        level: Log level string
        
    Example:
        >>> set_log_level("digital_twin.pipeline", "DEBUG")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))