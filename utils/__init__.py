"""
Digital Twin Utils Module - Initialization
==========================================

Utility functions and helpers for the Digital Twin system.

Submodules:
-----------
1. config.py   - Configuration loading and validation
2. logging.py  - Logging setup and diagnostics
3. plotting.py - Visualization and plotting (future)
4. io.py       - Data I/O and serialization (future)

Functions:
----------
1. Configuration Management
   - load_config()        - Load YAML config
   - validate_config()    - Validate config structure
   - merge_configs()      - Override defaults

2. Logging & Diagnostics
   - setup_logging()      - Configure logging
   - get_logger()         - Get module logger
   - log_state()          - Log system state

3. Data I/O
   - save_results()       - Save to CSV/JSON
   - load_results()       - Load from CSV/JSON
   - export_pdf()         - Generate PDF reports

4. Utilities
   - ensure_directory()   - Create directories
   - format_bytes()       - Format file sizes
   - timing_context()     - Measure execution time

Usage:
------
from digital_twin.utils import (
    load_config,
    setup_logging,
    get_logger,
    save_results,
)

# Load configuration
config = load_config("config/7nm.yaml")

# Setup logging
setup_logging("logs/", level="INFO")

# Get logger
logger = get_logger(__name__)

# Run system
pipeline = ReliabilityPipeline(config)
for z in measurements:
    state = pipeline.step(z, T_k, activity)

# Save results
save_results(pipeline, "results/simulation_7nm.csv")

Version: 1.0.0
Author: Digital Twin Team
Date: December 24, 2025
"""

import sys
from pathlib import Path

# Import utility modules
from .config import (
    load_config,
    validate_config,
    merge_configs,
    ConfigError,
)

from .logging import (
    setup_logging,
    get_logger,
    log_state,
    log_step,
)

__all__ = [
    # Config functions
    "load_config",
    "validate_config",
    "merge_configs",
    "ConfigError",
    # Logging functions
    "setup_logging",
    "get_logger",
    "log_state",
    "log_step",
]

__version__ = "1.0.0"
__author__ = "Digital Twin Team"
__date__ = "2025-12-24"

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [CONFIG_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)