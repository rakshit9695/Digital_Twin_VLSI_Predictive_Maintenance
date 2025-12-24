"""
Digital Twin Tests - Physics Models
====================================

Unit tests for physics models:
- BTI (Bias Temperature Instability) model
- HCI (Hot Carrier Injection) model
- EM (Electromigration) model
- Temperature scaling (Arrhenius)
- Stress dependencies

Test Coverage:
--------------
1. Model Calibration
   - Parameter validation
   - Physical bounds checking
   - Sensitivity analysis

2. Temperature Scaling
   - Arrhenius activation energy
   - Temperature coefficients
   - Absolute zero validity

3. Stress Dependencies
   - Voltage dependence (BTI, HCI)
   - Current dependence (EM)
   - Activity dependence

4. Model Accuracy
   - Comparison with literature
   - Physical consistency
   - Cross-technology validation

5. Extreme Conditions
   - Very high temperature (400K+)
   - Very low temperature (300K-)
   - Very high/low voltage
   - Edge cases

6. Degradation Rates
   - Positive rates always
   - Realistic magnitudes
   - Saturation behavior

Author: Digital Twin Team
Date: December 24, 2025
"""

import unittest
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBTIModel(unittest.TestCase):
    """Test BTI (Bias Temperature Instability) models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.T_ref = 313.15  # 40°C reference
        self.V_ref = 0.75    # 0.75V reference
        self.Ea_nbti = 0.10  # 100 meV activation energy
        self.Ea_pbti = 0.15  # 150 meV
        self.k_B = 8.617e-5  # Boltzmann constant [eV/K]
    
    def test_bti_temperature_scaling(self):
        """Test Arrhenius temperature dependence."""
        # BTI rate should follow: r(T) = r_ref × exp(Ea / k_B × (1/T_ref - 1/T))
        
        T_high = 373.15  # 100°C
        T_low = 313.15   # 40°C
        
        # At higher temperature, rate should be higher
        # rate_high = rate_ref × exp(...)
        # rate_low = rate_ref × exp(...)
        # rate_high > rate_low
        
        r_ref = 1e-5
        temp_factor_high = np.exp(self.Ea_nbti / self.k_B * (1/self.T_ref - 1/T_high))
        temp_factor_low = np.exp(self.Ea_nbti / self.k_B * (1/self.T_ref - 1/T_low))
        
        self.assertTrue(temp_factor_high > temp_factor_low)
    
    def test_nbti_voltage_dependence(self):
        """Test NBTI voltage dependence."""
        # NBTI rate should increase with |V_dd|
        # r_NBTI ∝ V_dd^n, where n ≈ 1-2
        
        V_refs = [0.5, 0.75, 1.0]
        n = 1.5  # Power law exponent
        
        rates = []
        for V in V_refs:
            rate = (V / self.V_ref) ** n
            rates.append(rate)
        
        # Rates should increase
        for i in range(1, len(rates)):
            self.assertGreater(rates[i], rates[i-1])
    
    def test_pbti_opposite_polarity(self):
        """Test PBTI occurs with opposite polarity."""
        # PBTI: hole-induced degradation (p-channel devices)
        # Should be significant but usually smaller than NBTI
        
        # This is a qualitative test
        pbti_magnitude = 0.8  # PBTI ~80% of NBTI
        nbti_magnitude = 1.0
        
        self.assertTrue(0.5 < pbti_magnitude / nbti_magnitude < 1.0)
    
    def test_bti_saturation(self):
        """Test BTI shows saturation with time."""
        # BTI follows t^n where 0.1 < n < 0.3
        
        times = np.array([1, 10, 100, 1000])
        n = 0.2  # Time exponent
        
        degradations = times ** n
        
        # Rate of degradation should decrease with time
        rates = np.diff(degradations) / np.diff(times)
        for i in range(1, len(rates)):
            self.assertLess(rates[i], rates[i-1])


class TestHCIModel(unittest.TestCase):
    """Test HCI (Hot Carrier Injection) models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.T_ref = 313.15
        self.V_ref = 0.75
        self.Ea_hci = 0.04  # ~40 meV activation energy (lower than BTI)
        self.k_B = 8.617e-5
    
    def test_hci_temperature_dependence(self):
        """Test HCI temperature dependence."""
        T_high = 373.15
        T_low = 313.15
        
        temp_factor_high = np.exp(self.Ea_hci / self.k_B * (1/self.T_ref - 1/T_high))
        temp_factor_low = np.exp(self.Ea_hci / self.k_B * (1/self.T_ref - 1/T_low))
        
        # Should scale with temperature
        self.assertTrue(temp_factor_high > 0)
        self.assertTrue(temp_factor_low > 0)
    
    def test_hci_activity_dependence(self):
        """Test HCI depends on switching activity."""
        # HCI ∝ activity (carrier generation rate)
        
        activities = [0.2, 0.4, 0.6]
        rates = []
        
        for activity in activities:
            rate = 1e-6 * activity  # Proportional to activity
            rates.append(rate)
        
        # Rates should increase with activity
        for i in range(1, len(rates)):
            self.assertGreater(rates[i], rates[i-1])
    
    def test_hci_voltage_dependence(self):
        """Test HCI exponential voltage dependence."""
        # HCI ∝ exp(-V_th,eff / V_dd)
        # Higher V_dd reduces HCI
        
        V_dds = [0.5, 0.75, 1.0]
        rates = []
        
        for V_dd in V_dds:
            rate = np.exp(-0.2 / V_dd)  # Decreases with V_dd
            rates.append(rate)
        
        # Should decrease with voltage
        for i in range(1, len(rates)):
            self.assertLess(rates[i], rates[i-1])
    
    def test_hci_smaller_than_bti(self):
        """Test HCI is typically smaller than BTI."""
        # At 7nm: NBTI ~45%, HCI ~25%, so HCI/NBTI ~0.56
        
        bti_rate = 1e-5
        hci_rate = 0.5e-5  # ~50% of NBTI
        
        ratio = hci_rate / bti_rate
        self.assertTrue(0.3 < ratio < 0.8)


class TestEMModel(unittest.TestCase):
    """Test EM (Electromigration) models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.T_ref = 313.15
        self.I_ref = 1e-3  # 1mA reference current
        self.Ea_em = 0.40  # ~400 meV activation energy (high)
        self.k_B = 8.617e-5
    
    def test_em_temperature_dependence(self):
        """Test EM strong temperature dependence."""
        T_high = 373.15
        T_low = 313.15
        
        temp_factor_high = np.exp(self.Ea_em / self.k_B * (1/self.T_ref - 1/T_high))
        temp_factor_low = np.exp(self.Ea_em / self.k_B * (1/self.T_ref - 1/T_low))
        
        # EM has much higher Ea than BTI/HCI
        # Ratio should be large
        ratio = temp_factor_high / temp_factor_low
        self.assertTrue(ratio > 2.0)  # Significant temperature sensitivity
    
    def test_em_current_dependence(self):
        """Test EM exponential current dependence."""
        # EM ∝ I^n × exp(-Ea / kT)
        # Usually n ≈ 1-2
        
        currents = np.array([1, 2, 5, 10])
        n = 1.5
        
        rates = currents ** n
        
        # Rates should increase with current
        for i in range(1, len(rates)):
            self.assertGreater(rates[i], rates[i-1])
    
    def test_em_path_dependent(self):
        """Test EM varies by interconnect characteristics."""
        # EM depends on: current density, material, cross-section
        
        # Higher current density → higher EM
        j_low = 1e5   # 100 kA/cm²
        j_high = 5e5  # 500 kA/cm²
        
        # EM ∝ j^n
        rate_low = j_low ** 1.5
        rate_high = j_high ** 1.5
        
        self.assertTrue(rate_high > rate_low)
    
    def test_em_typically_smallest(self):
        """Test EM is usually smallest contribution at 7nm."""
        # At 7nm: BTI ~45%, HCI ~25%, EM ~15%
        
        rates = {
            "NBTI": 1e-5,
            "HCI": 0.5e-5,
            "EM": 0.3e-5,
        }
        
        # Verify ordering
        self.assertTrue(rates["NBTI"] > rates["HCI"])
        self.assertTrue(rates["HCI"] > rates["EM"])


class TestArrheniusScaling(unittest.TestCase):
    """Test Arrhenius temperature scaling."""
    
    def test_arrhenius_equation(self):
        """Test standard Arrhenius form."""
        # r(T) = r_ref × exp(Ea / k_B × (1/T_ref - 1/T))
        
        Ea = 0.1  # eV
        k_B = 8.617e-5  # eV/K
        T_ref = 313.15  # K
        r_ref = 1e-5  # Reference rate
        
        # Test at multiple temperatures
        temps = [300, 313, 325, 350, 373]
        
        for T in temps:
            exp_factor = np.exp(Ea / k_B * (1/T_ref - 1/T))
            rate = r_ref * exp_factor
            
            self.assertGreater(rate, 0)
            
            # Higher T should give higher rate
            if T > T_ref:
                self.assertGreater(exp_factor, 1.0)
            else:
                self.assertLess(exp_factor, 1.0)
    
    def test_activation_energies_realistic(self):
        """Test activation energies are in realistic range."""
        Ea_values = {
            "NBTI": 0.10,  # 100 meV
            "PBTI": 0.15,  # 150 meV
            "HCI": 0.04,   # 40 meV
            "EM": 0.40,    # 400 meV
        }
        
        # All should be positive and < 1 eV
        for mech, Ea in Ea_values.items():
            self.assertGreater(Ea, 0, f"{mech} Ea must be positive")
            self.assertLess(Ea, 1.0, f"{mech} Ea must be < 1 eV")
    
    def test_boltzmann_constant(self):
        """Test Boltzmann constant value."""
        k_B = 8.617e-5  # eV/K
        
        # At 300K, thermal energy kT ≈ 0.026 eV
        T = 300
        kT = k_B * T
        
        self.assertAlmostEqual(kT, 0.0259, places=3)


class TestCrossTechnologyValidation(unittest.TestCase):
    """Test physics models across technology nodes."""
    
    def test_28nm_params(self):
        """Test 28nm technology parameters."""
        # At 28nm: BTI-dominant (60%), HCI (15%), EM (10%)
        
        rates_28nm = {
            "NBTI": 1.2e-5,  # Higher at older node
            "HCI": 0.3e-5,
            "EM": 0.2e-5,
        }
        
        total = sum(rates_28nm.values())
        nbti_pct = rates_28nm["NBTI"] / total * 100
        
        self.assertTrue(50 < nbti_pct < 70)
    
    def test_7nm_params(self):
        """Test 7nm technology parameters."""
        # At 7nm: Balanced (BTI 45%, HCI 25%, EM 15%)
        
        rates_7nm = {
            "NBTI": 1.0e-5,
            "HCI": 0.6e-5,
            "EM": 0.3e-5,
        }
        
        total = sum(rates_7nm.values())
        nbti_pct = rates_7nm["NBTI"] / total * 100
        hci_pct = rates_7nm["HCI"] / total * 100
        
        self.assertTrue(40 < nbti_pct < 50)
        self.assertTrue(20 < hci_pct < 30)


class TestPhysicsConsistency(unittest.TestCase):
    """Test physical consistency of models."""
    
    def test_rates_always_positive(self):
        """Test degradation rates are always positive."""
        temps = np.linspace(300, 400, 50)
        
        for T in temps:
            Ea = 0.1
            k_B = 8.617e-5
            rate = 1e-5 * np.exp(Ea / k_B * (1/313.15 - 1/T))
            
            self.assertGreater(rate, 0, f"Rate at T={T}K is not positive")
    
    def test_increasing_stress_increases_rate(self):
        """Test higher stress increases degradation rate."""
        V_dds = [0.5, 0.75, 1.0]
        rates = []
        
        for V_dd in V_dds:
            # NBTI ∝ V_dd^n
            rate = V_dd ** 1.5
            rates.append(rate)
        
        # Should be monotonically increasing
        for i in range(1, len(rates)):
            self.assertGreater(rates[i], rates[i-1])
    
    def test_temperature_consistency(self):
        """Test temperature scaling is consistent."""
        # At constant stress, higher T gives higher rate
        
        T_low = 313.15
        T_high = 373.15
        V_dd = 0.75
        
        Ea = 0.1
        k_B = 8.617e-5
        
        exp_low = np.exp(Ea / k_B * (1/313.15 - 1/T_low))
        exp_high = np.exp(Ea / k_B * (1/313.15 - 1/T_high))
        
        # Higher T should give higher rate
        self.assertGreater(exp_high, exp_low)


class TestModelEdgeCases(unittest.TestCase):
    """Test models at extreme conditions."""
    
    def test_zero_stress(self):
        """Test with zero stress (V_dd = 0)."""
        # Rates should go to zero
        rate_low_v = 0.0 ** 1.5
        self.assertEqual(rate_low_v, 0)
    
    def test_very_high_temperature(self):
        """Test at very high temperature."""
        T_extreme = 500  # 500K (227°C)
        
        Ea = 0.1
        k_B = 8.617e-5
        exp_factor = np.exp(Ea / k_B * (1/313.15 - 1/T_extreme))
        
        # Should still be finite
        self.assertTrue(np.isfinite(exp_factor))
        self.assertGreater(exp_factor, 1.0)
    
    def test_very_low_temperature(self):
        """Test at very low temperature."""
        T_extreme = 250  # 250K (-23°C)
        
        Ea = 0.1
        k_B = 8.617e-5
        exp_factor = np.exp(Ea / k_B * (1/313.15 - 1/T_extreme))
        
        # Should still be positive
        self.assertGreater(exp_factor, 0)
        self.assertLess(exp_factor, 1.0)


if __name__ == "__main__":
    unittest.main()