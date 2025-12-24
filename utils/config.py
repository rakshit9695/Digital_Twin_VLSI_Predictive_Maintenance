"""
Digital Twin Utils - Configuration Management
==============================================

Configuration loading, validation, and merging utilities.

Features:
---------
1. YAML Loading
   - Load configuration from YAML files
   - Support for includes/references
   - Environment variable substitution

2. Validation
   - Schema validation
   - Type checking
   - Required field checking
   - Bounds validation

3. Merging
   - Override defaults with custom configs
   - Deep merge capabilities
   - Priority handling

Configuration Structure:
-----------------------
config:
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
      j_c: 1e6
  
  filters:
    type: "UKF"
    alpha: 1e-3
    beta: 2.0
    kappa: 0.0
  
  constraints:
    v_th_limit: 0.1
    freq_limit: 0.95
    delay_limit: 1.2

Example:
--------
>>> from digital_twin.utils import load_config
>>> 
>>> config = load_config("config/7nm.yaml")
>>> validate_config(config)
>>> 
>>> # Override specific values
>>> custom_config = load_config("config/custom.yaml")
>>> merged = merge_configs(config, custom_config)

Author: Digital Twin Team
Date: December 24, 2025
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration error."""
    pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigError: If file not found or invalid YAML
        
    Example:
        >>> config = load_config("config/7nm.yaml")
        >>> print(config["technology_node"])
        '7nm'
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading config: {e}")
    
    if config is None:
        raise ConfigError(f"Empty config file: {config_path}")
    
    # Perform environment variable substitution
    config = _substitute_env_vars(config)
    
    logger.info(f"Loaded config from {config_path}")
    
    return config


def _substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute environment variables in config.
    
    Supports format: ${VAR_NAME:default_value}
    
    Args:
        obj: Config object (dict, list, str, etc.)
        
    Returns:
        Config with substituted variables
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Replace ${VAR:default} or ${VAR}
        import re
        pattern = r'\$\{(\w+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default = match.group(2) or ""
            return os.environ.get(var_name, default)
        
        return re.sub(pattern, replace_var, obj)
    else:
        return obj


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ConfigError: If validation fails
        
    Example:
        >>> config = load_config("config/7nm.yaml")
        >>> validate_config(config)
        True
    """
    # Required top-level keys
    required_keys = ["technology_node", "physics", "filters"]
    
    for key in required_keys:
        if key not in config:
            raise ConfigError(f"Missing required key: {key}")
    
    # Technology node validation
    valid_nodes = ["28nm", "16nm", "10nm", "7nm", "5nm"]
    if config["technology_node"] not in valid_nodes:
        raise ConfigError(
            f"Invalid technology node: {config['technology_node']}. "
            f"Must be one of {valid_nodes}"
        )
    
    # Physics parameters validation
    _validate_physics(config.get("physics", {}))
    
    # Filter configuration validation
    _validate_filters(config.get("filters", {}))
    
    # Constraints validation
    _validate_constraints(config.get("constraints", {}))
    
    logger.info("Configuration validation passed")
    return True


def _validate_physics(physics: Dict[str, Any]) -> None:
    """Validate physics parameters."""
    if not isinstance(physics, dict):
        raise ConfigError("Physics config must be a dictionary")
    
    # Validate BTI model
    if "bti" in physics:
        bti = physics["bti"]
        if not (0 < bti.get("ea_nbti", 0) < 1):
            raise ConfigError("BTI Ea must be between 0 and 1 eV")
        if not (0 < bti.get("k_v", 0)):
            raise ConfigError("BTI k_v must be positive")
    
    # Validate HCI model
    if "hci" in physics:
        hci = physics["hci"]
        if not (0 < hci.get("ea_hci", 0) < 1):
            raise ConfigError("HCI Ea must be between 0 and 1 eV")
    
    # Validate EM model
    if "em" in physics:
        em = physics["em"]
        if not (0 < em.get("ea_em", 0) < 1):
            raise ConfigError("EM Ea must be between 0 and 1 eV")


def _validate_filters(filters: Dict[str, Any]) -> None:
    """Validate filter configuration."""
    if not isinstance(filters, dict):
        raise ConfigError("Filters config must be a dictionary")
    
    valid_types = ["EKF", "UKF", "ParticleFilter"]
    filter_type = filters.get("type", "UKF")
    
    if filter_type not in valid_types:
        raise ConfigError(
            f"Invalid filter type: {filter_type}. "
            f"Must be one of {valid_types}"
        )
    
    # UKF parameters
    if filter_type == "UKF":
        alpha = filters.get("alpha", 1e-3)
        if not (0 < alpha <= 1):
            raise ConfigError("UKF alpha must be in (0, 1]")


def _validate_constraints(constraints: Dict[str, Any]) -> None:
    """Validate constraint parameters."""
    if not constraints:
        return
    
    if not isinstance(constraints, dict):
        raise ConfigError("Constraints config must be a dictionary")
    
    # Validate limit values
    for key, value in constraints.items():
        if not isinstance(value, (int, float)):
            raise ConfigError(f"Constraint {key} must be numeric")
        if value <= 0:
            logger.warning(f"Constraint {key} is non-positive: {value}")


def merge_configs(base: Dict[str, Any],
                 override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge override config into base config.
    
    Args:
        base: Base configuration
        override: Configuration to merge in (overrides base)
        
    Returns:
        Merged configuration
        
    Example:
        >>> config1 = {"a": 1, "b": {"c": 2}}
        >>> config2 = {"b": {"d": 3}}
        >>> merged = merge_configs(config1, config2)
        >>> merged
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Direct override
            result[key] = value
    
    logger.info(f"Merged {len(override)} config keys")
    return result


def get_config_value(config: Dict[str, Any],
                    key_path: str,
                    default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "physics.bti.ea_nbti")
        default: Default value if not found
        
    Returns:
        Config value or default
        
    Example:
        >>> config = {"physics": {"bti": {"ea_nbti": 0.10}}}
        >>> get_config_value(config, "physics.bti.ea_nbti")
        0.10
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_config_value(config: Dict[str, Any],
                    key_path: str,
                    value: Any) -> Dict[str, Any]:
    """
    Set nested config value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "physics.bti.ea_nbti")
        value: Value to set
        
    Returns:
        Modified config
        
    Example:
        >>> config = {"physics": {"bti": {"ea_nbti": 0.10}}}
        >>> set_config_value(config, "physics.bti.ea_nbti", 0.12)
        >>> config["physics"]["bti"]["ea_nbti"]
        0.12
    """
    keys = key_path.split(".")
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return config


def print_config(config: Dict[str, Any],
                indent: int = 0) -> None:
    """
    Pretty-print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{'  ' * indent}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{'  ' * indent}{key}: {value}")


def save_config(config: Dict[str, Any],
               output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved config to {output_path}")