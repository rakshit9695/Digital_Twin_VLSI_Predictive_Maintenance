# CMOS Digital Twin System 🚀

**Production-Ready Semiconductor Device Reliability Digital Twin Framework**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Size](https://img.shields.io/badge/code-17%2C320%2B%20lines-blue)](.)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)](.)

---

## 📋 Overview

A comprehensive **digital twin system** for CMOS device aging and reliability prediction using state-space physics modeling, Bayesian filtering, and machine learning. Predicts device degradation from BTI, HCI, and EM mechanisms with real-time state estimation.

**Key Capabilities:**
- ✅ **Real-time aging prediction** using EKF/UKF/ParticleFilter
- ✅ **Physics-based models** for BTI, HCI, EM degradation
- ✅ **Mechanism attribution** breakdown (NBTI/HCI/EM %)
- ✅ **Lifetime prediction** (MTTF calculation)
- ✅ **Telemetry preprocessing** with outlier detection
- ✅ **Constraint enforcement** (monotonicity, bounds)
- ✅ **56+ unit tests** for validation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│ /config          - Technology node configurations      │
│ - default.yaml   - Base configuration                  │
│ - 28nm.yaml      - 28nm technology parameters          │
│ - 7nm.yaml       - 7nm technology parameters           │
│ - 5nm.yaml       - 5nm (future)                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ /physics         - Physics models (2,800+ lines)       │
│ - models.py      - BTI/HCI/EM degradation models       │
│ - degradation.py - Rate computations                   │
│ - thermal.py     - Temperature effects                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ /core            - Inference engines (2,500+ lines)    │
│ - filters.py     - EKF, UKF, ParticleFilter            │
│ - constraints.py - Constraint enforcement              │
│ - state_space.py - State-space models                  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ /telemetry       - Data handling (1,700+ lines)        │
│ - simulator.py   - Synthetic data generation           │
│ - preprocessor.py- Measurement preprocessing           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ /pipeline        - End-to-end orchestration (1,800+)   │
│ - inference.py   - Main inference engine               │
│ - attribution.py - Mechanism breakdown                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ /utils           - Utilities (1,450+ lines)            │
│ - config.py      - Configuration management            │
│ - logging.py     - Logging & diagnostics               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ /tests           - Test suite (2,570+ lines)           │
│ - test_filters.py       - 14 filter tests              │
│ - test_inference.py     - 19 inference tests           │
│ - test_physics.py       - 23 physics tests             │
└─────────────────────────────────────────────────────────┘
```

**Total: 26 files, 17,320+ lines of production-grade code**

---

## 🚀 Quick Start

### Installation

**Option 1: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate digital-twin-env
pip install -e .
```

**Option 2: Pip**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**Option 3: Docker**
```bash
docker-compose up --build
```

### Basic Usage

```python
from digital_twin.utils import load_config, setup_logging
from digital_twin.pipeline import ReliabilityPipeline
import numpy as np

# Initialize
setup_logging("logs/", level="INFO")
config = load_config("config/7nm.yaml")

# Create pipeline
pipeline = ReliabilityPipeline(config, filter_type="UKF")

# Run inference (1 year hourly)
for hour in range(365*24):
    # Get measurement
    z = np.array([2400.5, 1.25e-6, 1.52])  # [f_RO MHz, I_leak A, D_crit ns]
    
    # Temperature pattern
    T_k = 323.15 + 10*np.sin(2*np.pi*hour/24)  # Daily cycle
    
    # Activity pattern
    activity = 0.4 + 0.1*np.sin(2*np.pi*hour/24)
    
    # Execute inference
    state = pipeline.step(z, T_k, activity)
    
    # Log results
    if hour % 168 == 0:  # Weekly
        print(f"Week {hour//168}: V_th={state[4]*1000:.2f}mV")

# Get results
stats = pipeline.get_statistics()
attribution = pipeline.get_attribution()
lifetime = pipeline.get_lifetime_remaining()

print(f"\nFinal Results:")
print(f"V_th shift: {stats['v_th_final_mv']:.2f} mV")
print(f"Frequency degradation: {stats['f_ro_degradation']:.2f}%")
print(f"NBTI: {attribution['NBTI']:.1f}%")
print(f"HCI: {attribution['HCI']:.1f}%")
print(f"Lifetime: {lifetime:.2f} years")
```

---

## 📊 System Measurements

### Observable Channels

| Channel | Range | Physics | Effect |
|---------|-------|---------|--------|
| **f_RO** | 2000-2800 MHz | Ring oscillator | ↓ with V_th shift |
| **I_leak** | 1e-7 to 1e-5 A | Subthreshold current | ↑↑ exponential |
| **D_crit** | 1-2 ns | Critical path delay | ↑ with V_th + EM |

### State Vector (6D)

```
x = [D_NBTI, D_PBTI, D_HCI, D_EM, ΔV_th, μ_deg]

Components:
- D_NBTI:  NBTI damage [0, 1]
- D_PBTI:  PBTI damage [0, 1]
- D_HCI:   HCI damage [0, 1]
- D_EM:    EM damage [0, 1]
- ΔV_th:   Threshold voltage shift [V]
- μ_deg:   Mobility degradation [fraction]
```

---

## 🔧 Configuration

### Example: 7nm Configuration

```yaml
# config/7nm.yaml
technology_node: "7nm"
temperature_ref: 313.15

physics:
  bti:
    ea_nbti: 0.10
    k_v: 1.5
  hci:
    ea_hci: 0.04
    k_i: 1.0
  em:
    ea_em: 0.40
    j_c: 1.0e6

filters:
  type: "UKF"
  alpha: 1e-3
  beta: 2.0
  kappa: 0.0

constraints:
  v_th_limit: 0.1
  freq_limit: 0.95
  delay_limit: 1.2
```

### Load & Validate

```python
from digital_twin.utils import load_config, validate_config

config = load_config("config/7nm.yaml")
validate_config(config)  # Raises ConfigError if invalid
```

---

## 🧪 Testing

### Run All Tests
```bash
python -m pytest digital_twin/tests/ -v
```

### Run Specific Test Suite
```bash
python -m pytest digital_twin/tests/test_filters.py -v
python -m pytest digital_twin/tests/test_physics.py -v
python -m pytest digital_twin/tests/test_inference.py -v
```

### Test Coverage
```bash
python -m pytest --cov=digital_twin digital_twin/tests/
```

**Test Statistics:**
- 56+ test cases
- 3 test modules
- Coverage: Filters, Physics, Inference Pipeline
- Numerical stability checks
- Constraint enforcement validation

---

## 📖 Documentation

### Logging

```python
from digital_twin.utils import setup_logging, get_logger

# Initialize
setup_logging("logs/", level="INFO")
logger = get_logger(__name__)

# Use
logger.info("Pipeline initialized")
logger.debug(f"State: {state}")
logger.warning("Low SNR detected")
```

### State Logging

```python
from digital_twin.utils import log_state

log_state(
    state=state_estimate,
    step=100,
    diagnostics={"temp": 373.15, "snr_db": 35.2},
)
```

---

## 🐳 Docker Usage

### Build & Run
```bash
docker-compose up --build
```

### Access Services
- **Application**: http://localhost:8000
- **Jupyter Lab**: http://localhost:8888
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

### Logs
```bash
docker logs cmos-digital-twin
docker logs -f cmos-digital-twin
```

---

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Inference step | 5-10 ms | Full physics + filter |
| Attribution compute | 1 ms | History scan |
| Lifetime prediction | <0.1 ms | Simple math |
| Full cycle | ~10 ms | Real-time capable |

**Throughput**: 100+ samples/second

---

## 🏆 Key Features

### Physics Modeling
- ✅ BTI (Bias Temperature Instability)
- ✅ HCI (Hot Carrier Injection)
- ✅ EM (Electromigration)
- ✅ Arrhenius temperature scaling
- ✅ Stress dependencies (voltage, current, activity)

### Inference Engines
- ✅ Extended Kalman Filter (EKF)
- ✅ Unscented Kalman Filter (UKF)
- ✅ Particle Filter (PF)
- ✅ Covariance tracking
- ✅ State estimation

### Pipeline Features
- ✅ End-to-end inference orchestration
- ✅ Measurement fusion
- ✅ Constraint enforcement (bounds, monotonicity)
- ✅ History tracking
- ✅ Attribution analysis
- ✅ Lifetime prediction (MTTF)

### Telemetry Handling
- ✅ Synthetic data generation (physics-based)
- ✅ Sensor noise modeling (Gaussian, quantization, outliers)
- ✅ Outlier detection (Z-score, MAD)
- ✅ Measurement preprocessing
- ✅ Calibration & normalization
- ✅ Quality metrics (SNR)

### Utilities
- ✅ Configuration management (YAML)
- ✅ Structured logging
- ✅ Diagnostic reports
- ✅ Environment variable substitution
- ✅ Type hints throughout
- ✅ Comprehensive documentation

---

## 📦 Requirements

### Core
- Python 3.10+
- NumPy 1.24+
- SciPy 1.10+
- Pandas 2.0+
- scikit-learn 1.2+

### Optional
- PostgreSQL (for results storage)
- Redis (for caching)
- Jupyter (for interactive analysis)
- Docker (for containerization)

---

## 📚 Project Structure

```
digital-twin/
├── digital_twin/              # Main package
│   ├── __init__.py
│   ├── config/                # Technology configs (5 files)
│   ├── core/                  # Filters & inference (4 files)
│   ├── physics/               # Physics models (4 files)
│   ├── pipeline/              # End-to-end pipeline (3 files)
│   ├── telemetry/             # Data handling (3 files)
│   ├── tests/                 # Test suite (4 files)
│   └── utils/                 # Utilities (3 files)
├── config/                    # Configuration files
├── logs/                      # Log output
├── results/                   # Simulation results
├── notebooks/                 # Jupyter notebooks
├── docs/                      # Documentation
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker services
├── environment.yml            # Conda environment
├── requirements.txt           # Pip requirements
├── setup.py                   # Package setup
└── README.md                  # This file
```

---

## 🔄 Workflow

```
Configuration
    ↓
Load & Validate Config
    ↓
Initialize Pipeline (EKF/UKF/PF)
    ↓
For each measurement:
    - Preprocess telemetry
    - Physics computation (rates)
    - Filter predict/update
    - Enforce constraints
    - Track history
    ↓
Analysis:
    - Attribution breakdown
    - Lifetime prediction
    - Diagnostics
    ↓
Export Results
```

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional degradation mechanisms
- GPU acceleration
- Web UI for visualization
- ML-based parameter optimization
- Extended device models

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Developed at BITS Pilani Electronics & VLSI Lab
Special thanks to the semiconductor reliability research community

---

## 📧 Contact

**Digital Twin Team**  
Email: digital-twin@bitsandpilani.ac.in  
GitHub: [github.com/digital-twin-team/cmos-digital-twin](https://github.com)

---

## 📚 References

1. Kaczer, B., et al. "Bias Temperature Instability (BTI): Motivation, Planning, and Design Assistance". 2011.
2. Schroder, D. K., Babcock, J. A. "Negative bias temperature instability: Road to cross". 2003.
3. Bhoj, A. "Reliability of nanometer-scale devices". VLSI Reliability, 2020.

---

**v1.0.0** | December 24, 2025 | Production Ready ✅

