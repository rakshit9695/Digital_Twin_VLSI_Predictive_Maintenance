"""
CMOS Digital Twin System - Setup Configuration
===============================================

Installation and package configuration for the digital twin system.

Usage:
------
pip install -e .              # Development install
pip install -e .[dev]         # With development dependencies
pip install -e .[test]        # With testing dependencies
pip install -e .[all]         # All optional dependencies

Author: Digital Twin Team
Date: December 24, 2025
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.startswith("#")
    ]

# Development requirements
dev_requirements = [
    "pytest>=7.2.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "ipdb>=0.13.0",
    "sphinx>=6.0.0",
]

# Test requirements
test_requirements = [
    "pytest>=7.2.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-asyncio>=0.20.0",
]

setup(
    # Project metadata
    name="digital-twin-cmos",
    version="1.0.0",
    author="Digital Twin Team",
    author_email="digital-twin@bitsandpilani.ac.in",
    description="CMOS Device Reliability Digital Twin System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/digital-twin-team/cmos-digital-twin",
    project_urls={
        "Documentation": "https://digital-twin.readthedocs.io",
        "Source": "https://github.com/digital-twin-team/cmos-digital-twin",
        "Bug Reports": "https://github.com/digital-twin-team/cmos-digital-twin/issues",
    },
    license="MIT",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "docs", "notebooks"]),
    include_package_data=True,
    
    # Python version
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": test_requirements,
        "all": dev_requirements + test_requirements,
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "digital-twin=digital_twin.cli:main",
        ],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
    keywords=[
        "CMOS",
        "digital twin",
        "reliability",
        "aging",
        "degradation",
        "BTI",
        "HCI",
        "electromigration",
        "semiconductor",
        "VLSI",
        "Kalman filter",
        "state estimation",
    ],
    
    # Additional options
    zip_safe=False,
    package_data={
        "digital_twin": [
            "config/*.yaml",
            "data/*",
        ],
    },
)