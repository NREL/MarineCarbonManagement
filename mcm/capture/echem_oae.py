"""Model of electrochemical ocean alkalinity enhancement system. Inputs based on Ferella (2025).

Citation:
    Ferella, Francesco, et al. "Ocean Alkalinity Enhancement Using Bipolar Membrane Electrodialysis: 
        Technical Analysis and Cost Breakdown of a Full-Scale Plant." 
        Industrial & Engineering Chemistry Research 64.13 (2025): 7085-7099.
        https://doi.org/10.1021/acs.iecr.4c04364
"""

__author__ = "James Niffenegger, Kaitlin Brunik"
__copyright__ = "Copyright 2024, National Renewable Energy Laboratory"
__maintainer__ = "Kaitlin Brunik"
__email__ = ("james.niffenegger", "kaitlin.brunik@nrel.gov")

import os
import math
import warnings
import numpy as np
import pandas as pd
import PyCO2SYS as pyco2
import matplotlib.pyplot as plt

from attrs import define, field
from typing import Tuple, Optional
from scipy.optimize import root_scalar

from mcm.utilities.validators import range_val, gt_zero, contains
from mcm.utilities.utilities import (
    sal_m_to_ppt,
    sal_ppt_to_m,
    m_to_umol_per_kg,
    umol_per_kg_to_m,
)

MM_CACO3 = 100.09 # g/mol, molar mass of CaCO3
MM_NACL = 58.44 # g/mol, molar mass of NaCl
R_H2O = 1030 # kg/m3, density of water

@define
class OAEInputs:
    """
    Configuration for an Ocean Alkalinity Enhancement (OAE) marine carbon capture system.

    Attributes:
        P_ed1 (float): Power per ED unit (W). Calculated from P_edMax and N_edMax if not provided.
        Q_ed1 (float): Flow rate per ED unit (m³/s). Calculated if not provided.
        N_edMin (int): Minimum number of ED units. Default is 1.
        P_edMax (float): Total ED system power (W). Default is 3.5 MW.
        N_edMax (int): Maximum number of ED units. Default is 10.
        E_HCl (float): Energy required per mole of HCl (kWh/mol). Computed if not given.
        E_NaOH (float): Energy required per mole of NaOH (kWh/mol). Computed if not given.
        assumed_CDR_rate (float): Assumed carbon dioxide removal rate (CDR) per mole of NAOH (mol/mol). Default is 0.8 moles.
        Q_edMax (float): Max ED system flow rate (m³/s). Default is based on empirical values ~0.0324.
        Q_OMax (float): Max overall intake flow (m³/s). Derived from ED flow if not provided.
        frac_EDflow (float): Fraction of total flow treated by ED. Computed if not given.
        frac_baseFlow (float): Fraction of ED-treated flow that becomes base. Default is 0.5.
        frac_acidFlow (float): Fraction of ED-treated flow producing acid. Computed as 1 - frac_baseFlow.
        c_a (float): Concentration of generated acid (mol/L). Default is 0.49.
        c_b (float): Concentration of generated base (mol/L). Default is 0.54.
        use_storage_tanks (bool): Whether storage tanks are used. Default is True.
        store_hours (float): Storage duration (h). Set to 0 if tanks are disabled.
        acid_disposal_method (str): Acid disposal strategy. Must be one of ["sell acid","sell rca","acid disposal"].
    """
    P_ed1: float = field(default=None)
    Q_ed1: float = field(default=None)
    N_edMin: int = field(default=1, validator=gt_zero)
    P_edMax: float = field(default=350*10**4, validator=gt_zero)
    N_edMax: int = field(default=10, validator=gt_zero)
    E_HCl: float = field(default=None)
    E_NaOH: float = field(default=None)
    assumed_CDR_rate: float = field(default=0.8)
    Q_edMax: float = field(default=(194.4)/1000/60*10, validator=gt_zero)
    Q_OMax: float = field(default=None)
    frac_EDflow: float = field(default=None)
    frac_baseFlow: float = field(default=1/2, validator=range_val(0, 1))
    frac_acidFlow: float = field(default=None)
    c_a: float = 0.49
    c_b: float = 0.54
    use_storage_tanks: bool = field(default=True)
    store_hours: float = field(default=12)
    acid_disposal_method: str = field(default="sell acid", validator=contains(["sell acid", "sell rca", "acid disposal"]))

    def __attrs_post_init__(self):
        # Calculate per-unit power and flow if not set
        ed_units = self.N_edMax - self.N_edMin + 1
        if self.P_ed1 is None:
            self.P_ed1 = self.P_edMax / ed_units
        if self.Q_ed1 is None:
            self.Q_ed1 = self.Q_edMax / ed_units

        # Adjust storage hours if tanks not used
        if not self.use_storage_tanks:
            self.store_hours = 0

        # Estimate flow fractions if not provided
        if self.frac_EDflow is None:
            self.Q_OMax = self.Q_edMax * 100
            self.frac_EDflow = self.Q_edMax / self.Q_OMax

        # Compute acid flow fraction
        self.frac_acidFlow = 1 - self.frac_baseFlow

        # Input validation
        if self.frac_acidFlow > 1:
            raise ValueError("frac_acidFlow exceeds 1; check inputs.")
        if self.frac_EDflow > 1:
            raise ValueError("frac_EDflow exceeds 1; check inputs.")

        # Seawater properties (used in energy calculation)
        seawater = SeaWaterInputs()

        # Estimate energy requirements for acid/base production if not set
        if self.E_HCl is None:
            acid_mol_per_m3 = self.frac_acidFlow * self.c_a * 1000 + seawater.h_i * 1000
            self.E_HCl = (self.P_edMax / (3600 * self.Q_edMax * acid_mol_per_m3)) / 1000  # kWh/mol

        if self.E_NaOH is None:
            base_mol_per_m3 = self.frac_baseFlow * self.c_b * 1000 + (seawater.kw / seawater.h_i) * 1000
            self.E_NaOH = (self.P_edMax / (3600 * self.Q_edMax * base_mol_per_m3)) / 1000  # kWh/mol

@define
class SeaWaterInputs:
    """
    Represents the initial inputs and derived parameters for seawater chemistry.

    Attributes:
        tempC (float): Average ambient seawater temperature in °C. Default is 10.0°C.
        sal (float): Average salinity in ppt. Default is equivalent to 1.3 mol/kg.
        desal_temp_delta (float): Anticipated temperature increase from desalination in °C. Default is 2°C.
        ed_temp_delta (float): Anticipated temperature increase from ED in °C. Default is 2°C.
        dic_i (float): Initial DIC concentration in mol/L. Default is 2.2e-3 mol/L.
        pH_i (float): Initial pH of seawater. Default is 8.1.

    Derived Attributes:
        kw (float): Water dissociation constant at the specified temperature and salinity.
        sal_i (float): Salinity converted to mol NaCl/m³.
        SAL_i_mol_m3 (float): Salinity in mol NaCl/m³.
        h_i (float): Initial hydrogen ion concentration in mol/L.
        dic_iu (float): Initial DIC in µmol/kg.
        ta_i (float): Total alkalinity in mol/L.
        ca_i (float): Total calcium concentration in mol/L.
    """
    tempC: float = 10.0
    desal_temp_delta: float = 2.0
    ed_temp_delta: float = 2.0 
    sal_ppt_i: float = sal_m_to_ppt(1.3)
    dic_i: float = 2*2.2e-3
    pH_i: float = 8.1

    kw: float = field(init=False)
    sal_i: float = field(init=False)
    SAL_i_mol_m3: float = field(init=False)
    h_i: float = field(init=False)
    dic_iu: float = field(init=False)
    ta_i: float = field(init=False)
    ca_i: float = field(init=False)
    tempC_i: float = field(init=False)

    def __attrs_post_init__(self):
        # (C) initial temperature of the brine is equivalent to the sum of the 
        # ambient temp and the change in temp from the desal system
        self.tempC_i = self.tempC + self.desal_temp_delta 
        tempK = self.tempC_i + 273.15  # Temperature in Kelvin

        # Calculate water dissociation constant (kw)
        self.kw = math.exp(
            -13847.26 / tempK
            + 148.9652
            - 23.6521 * math.log(tempK)
            + (
                (118.67 / tempK - 5.977 + 1.0495 * math.log(tempK)) * self.sal_ppt_i**0.5
                - 0.01615 * self.sal_ppt_i
            )
        )

        # Convert salinity to mol NaCl/m³
        self.sal_i = sal_ppt_to_m(self.sal_ppt_i)
        self.SAL_i_mol_m3 = self.sal_i * 1000

        # Calculate initial hydrogen ion concentration
        self.h_i = 10**-self.pH_i

        # Convert DIC to µmol/kg
        self.dic_iu = m_to_umol_per_kg(self.dic_i)

        # Use PyCO2SYS to calculate total alkalinity
        kwargs = dict(
            par1=self.dic_iu,
            par2=self.pH_i,
            par1_type=2,  # DIC
            par2_type=3,  # pH
            salinity=self.sal_ppt_i,  # Salinity of the sample (ppt)
            temperature=self.tempC_i,  # Temperature at input conditions (C)
        )
        results = pyco2.sys(**kwargs)
        self.ta_i = umol_per_kg_to_m(results["alkalinity"])
        self.ca_i = umol_per_kg_to_m(results["total_calcium"])

@define
class PumpInputs:
    """
    A class to define the input parameters for various pumps in the system.

    Attributes:
        y_pump (float): The constant efficiency of the pump. Default is 0.8.
        p_o_min_bar (float): The minimum pressure (in bar) for seawater intake with filtration. Default is 0.15.
        p_o_max_bar (float): The maximum pressure (in bar) for seawater intake with filtration. Default is 1.5.
        p_ed_min_bar (float): The minimum pressure (in bar) for ED (Electrodialysis) units. Default is 0.12.
        p_ed_max_bar (float): The maximum pressure (in bar) for ED (Electrodialysis) units. Default is 1.2.
        p_a_min_bar (float): The minimum pressure (in bar) for pumping acid. Default is 0.16.
        p_a_max_bar (float): The maximum pressure (in bar) for pumping acid. Default is 1.6.
        p_i_min_bar (float): The minimum pressure (in bar) for pumping seawater for acid addition. Default is 0.
        p_i_max_bar (float): The maximum pressure (in bar) for pumping seawater for acid addition. Default is 0.
        p_b_min_bar (float): The minimum pressure (in bar) for pumping base. Default is 0.69.
        p_b_max_bar (float): The maximum pressure (in bar) for pumping base. Default is 6.9.
        p_f_min_bar (float): The minimum pressure (in bar) for released seawater. Default is 0.2.
        p_f_max_bar (float): The maximum pressure (in bar) for released seawater. Default is 2.
    """

    y_pump: float = 0.8
    p_o_min_bar: float = 0.15
    p_o_max_bar: float = 1.5
    p_ed_min_bar: float = 0.12
    p_ed_max_bar: float = 1.2
    p_a_min_bar: float = 0.16
    p_a_max_bar: float = 1.6
    p_i_min_bar: float = 0
    p_i_max_bar: float = 0
    p_b_min_bar: float = 0.69
    p_b_max_bar: float = 6.9
    p_f_min_bar: float = 0.2
    p_f_max_bar: float = 2

@define
class Pump:
    """
    A class to represent a pump with specific flow rate and pressure characteristics.

    Attributes:
        Q_min (float): Minimum flow rate (m³/s).
        Q_max (float): Maximum flow rate (m³/s).
        p_min_bar (float): Minimum pressure (bar).
        p_max_bar (float): Maximum pressure (bar).
        eff (float): Efficiency of the pump.
        name (str): Name of the pump, used for identification in warnings.
        Q (float): Instantaneous flow rate (m³/s), initially set to zero.
    """

    Q_min: float
    Q_max: float
    p_min_bar: float
    p_max_bar: float
    eff: float
    name: str = field(default=None)
    Q: float = field(default=0)

    def pumpPower(self, Q):
        """
        Calculate the power required for the pump based on the flow rate.

        Args:
            Q (float): Flow rate (m³/s).

        Returns:
            float: Power required for the pump (W).

        Raises:
            ValueError: If the flow rate is out of the specified range or if the minimum pressure is greater than the maximum pressure.
        """
        if Q == 0:
            P_pump = 0
        elif Q < self.Q_min:
            warnings.warn(
                f"Pump {self.name:s}: Flow Rate is {(self.Q_min - Q) / self.Q_min * 100:.2f}% less than the range provided for pump power. Defaulting to minimum flow rate: {self.Q_min:.2f} (m³/s).",
                UserWarning,
            )
            Q = self.Q_min
            perc_range = (Q - self.Q_min) / (self.Q_max - self.Q_min)
            p_bar = (self.p_max_bar - self.p_min_bar) * perc_range + self.p_min_bar
            p = p_bar * 100000  # convert from bar to Pa
            P_pump = Q * p / self.eff
        elif Q > self.Q_max:
            warnings.warn(
                f"Flow Rate is {(Q - self.Q_max) / self.Q_max * 100:.2f}% larger than the range provided for pump power. Defaulting to maximum flow rate: {self.Q_max:.2f} (m³/s).",
                UserWarning,
            )
            Q = self.Q_max
            perc_range = (Q - self.Q_min) / (self.Q_max - self.Q_min)
            p_bar = (self.p_max_bar - self.p_min_bar) * perc_range + self.p_min_bar
            p = p_bar * 100000  # convert from bar to Pa
            P_pump = Q * p / self.eff
        elif self.p_min_bar > self.p_max_bar:
            raise ValueError(
                "Minimum Pressure Must Be Less Than or Equal to Maximum Pressure for Pump"
            )
        elif self.Q_min == self.Q_max:
            p_bar = self.p_max_bar  # max pressure used if the flow rate is constant
            p = p_bar * 100000  # convert from bar to Pa
            P_pump = Q * p / self.eff
        else:
            perc_range = (Q - self.Q_min) / (self.Q_max - self.Q_min)
            p_bar = (self.p_max_bar - self.p_min_bar) * perc_range + self.p_min_bar
            p = p_bar * 100000  # convert from bar to Pa
            P_pump = Q * p / self.eff
        return P_pump

    @property
    def P_min(self):
        """
        Calculate the minimum power required for the pump.

        Returns:
            float: Minimum power required for the pump (W).
        """
        return self.pumpPower(self.Q_min)

    @property
    def P_max(self):
        """
        Calculate the maximum power required for the pump.

        Returns:
            float: Maximum power required for the pump (W).
        """
        return self.pumpPower(self.Q_max)

    def power(self):
        """
        Calculate the power required for the pump based on the current flow rate.

        Returns:
            float: Power required for the pump (W).
        """
        return self.pumpPower(self.Q)


@define
class PumpOutputs:
    """
    A class to represent the outputs of initialized pumps.

    Attributes:
        pumpO (Pump): Seawater intake pump.
        pumpED (Pump): ED pump.
        pumpA (Pump): Acid pump.
        pumpI (Pump): Pump for seawater acidification.
        pumpB (Pump): Base pump.
        pumpF (Pump): Seawater output pump.
        pumpED4 (Pump): ED pump for S4.
    """

    pumpO: Pump
    pumpED: Pump
    pumpA: Pump
    pumpI: Pump
    pumpB: Pump
    pumpF: Pump
    pumpED4: Pump

def initialize_pumps(
    oae_config: OAEInputs, pump_config: PumpInputs
) -> PumpOutputs:
    """Initialize a list of Pump instances based on the provided configurations.

    Args:
        oae_config (OEAInputs): The ocean alkalinity enhancement inputs.
        pump_config (PumpInputs): The pump inputs.

    Returns:
        PumpOutputs: An instance of PumpOutputs containing all initialized pumps.
    """
    Q_ed1 = oae_config.Q_ed1
    N_edMin = oae_config.N_edMin
    N_edMax = oae_config.N_edMax
    p = pump_config
    pumpO = Pump(
        Q_ed1 * N_edMin * (1 / oae_config.frac_EDflow - 1),
        Q_ed1 * N_edMax * 1 / oae_config.frac_EDflow,
        p.p_o_min_bar,
        p.p_o_max_bar,
        p.y_pump,
        "pumpO",
    )  # features of seawater intake pump
    pumpED = Pump(
        Q_ed1 * N_edMin, 
        Q_ed1 * N_edMax, 
        p.p_ed_min_bar, 
        p.p_ed_max_bar, 
        p.y_pump,
        "pumpED",
    )  # features of ED pump
    pumpA = Pump(
        pumpED.Q_min * oae_config.frac_acidFlow, 
        pumpED.Q_max * oae_config.frac_acidFlow,
        p.p_a_min_bar, 
        p.p_a_max_bar,
        p.y_pump,
        "pumpA", 
    )  # features of acid pump
    pumpI = Pump(
        pumpED.Q_min * (1/oae_config.frac_EDflow - 1), 
        pumpED.Q_max * (1/oae_config.frac_EDflow - 1), 
        p.p_i_min_bar, 
        p.p_i_max_bar,
        p.y_pump,
        "pumpI",
    )  # features of pump for seawater acidification
    pumpB = Pump(
        pumpED.Q_min - pumpA.Q_min,
        pumpED.Q_max - pumpA.Q_max,
        p.p_b_min_bar,
        p.p_b_max_bar,
        p.y_pump,
        "pumpB",
    )  # features of base pump
    if oae_config.acid_disposal_method == "sell rca":
        # If acid disposal method is to sell rca, then the pumpF is not used
        pumpF = Pump(
            pumpI.Q_min + pumpB.Q_min + pumpA.Q_min, 
            pumpI.Q_max + pumpB.Q_max + pumpA.Q_max, 
            p.p_f_min_bar, 
            p.p_f_max_bar, 
            p.y_pump, 
            "pumpF") # features of seawater output pump (note min can be less if all base is used)
    else:
        pumpF = Pump(
        pumpI.Q_min + pumpB.Q_min,
        pumpI.Q_max + pumpB.Q_max,
        p.p_f_min_bar,
        p.p_f_max_bar,
        p.y_pump,
        "pumpF",
    )  # features of seawater output pump (note min can be less if all acid and base are used)
    pumpED4 = Pump(
        Q_ed1 * N_edMin, 
        Q_ed1 * N_edMax, 
        p.p_o_min_bar, 
        p.p_o_max_bar, 
        p.y_pump,
        "pumpED4",
    )  # features of ED pump for S4 (pressure of intake is used here)
    return PumpOutputs(
        pumpO=pumpO,
        pumpED=pumpED,
        pumpA=pumpA,
        pumpI=pumpI,
        pumpB=pumpB,
        pumpF=pumpF,
        pumpED4=pumpED4,
    )

@define
class RCALoadingCalculator:
    """
    Computes RCA loading required to neutralize acid to a target pH for OAE systems.

    Attributes:
        oae (OAEInputs): The full OAE system configuration.
        seawater (SeaWaterInputs): Seawater properties used in calculations.
        Wpg_rca (float): Power of motor for tumbling vs grams of RCA (W/g).
        frac_cao (float): Fraction of CaO in the RCA.
        frac_dissolved (float): Fraction of CaO that dissolves in seawater.
        frac_sellable_rca (float): Fraction of RCA that can be sold.
        tolerance (float): Acceptable error in pH target.
    """
    oae: OAEInputs
    seawater: SeaWaterInputs
    Wpg_rca =  1.543/1000 # (W/g) power of motor for tumbling vs grams of RCA
    frac_cao: float = 8.5 / 100
    frac_dissolved: float = 95 / 100
    frac_sellable_rca: float = 100 / 100
    tolerance: float = 0.01

    # RCA properties (constants)
    mm_cao: float = 56.1  # g/mol
    c_h2o = 4.18 # J/g-K, specific heat capacity of water
    c_rca = 0.75 # J/g-K, specific heat capacity of rock
    dH_cao = -186 * 10**3 # J/mol, enthalpy of neutralizing HCl with CaO

    # Private result attributes
    _w_final: float = field(default=None, init=False)
    _pH_final: float = field(default=None, init=False)
    _ca_final: float = field(default=None, init=False)
    _sal_final: float = field(default=None, init=False)
    _ta_final: float = field(default=None, init=False)
    _sal_a: float = field(default=None, init=False)
    _temp_final: float = field(default=None, init=False)
    _temp_change: float = field(default=None, init=False)

    def compute_acid_props(self):
        """Compute pH, salinity, TA, and Ca of the acid before RCA addition."""
        sal_a = self.seawater.sal_i - self.oae.c_a
        sal_a_ppt = sal_m_to_ppt(sal_a)
        ph_a = -math.log10(self.oae.c_a)
        tempC_a = self.seawater.tempC_i + self.seawater.ed_temp_delta #(C) temp of acid solution

        kwargs = dict(
            par1=self.seawater.dic_iu, 
            par2=ph_a,
            par1_type=2, 
            par2_type=3,
            salinity=sal_a_ppt,
            temperature=tempC_a
        )
        results = pyco2.sys(**kwargs)

        ta_a = umol_per_kg_to_m(results['alkalinity'])
        ca_a = umol_per_kg_to_m(results['total_calcium'])

        # Temperature considerations
        tempC_r = tempC_a # (C) assume rocks have the same temperature as the acid initially
        tempK_a = tempC_a + 273.15  # Temperature in Kelvin
        tempK_r = tempC_r + 273.15  # Temperature in Kelvin 

        return ph_a, sal_a, sal_a_ppt, ta_a, ca_a, tempK_a, tempK_r

    def get_target_ta(self, sal_ppt):
        """Get target total alkalinity at the final pH."""

        kwargs = dict(
            par1=self.seawater.dic_iu, 
            par2=self.seawater.pH_i,
            par1_type=2, 
            par2_type=3,
            salinity=sal_ppt,
            temperature=self.seawater.tempC_i
        )
        results = pyco2.sys(**kwargs)
        return umol_per_kg_to_m(results['alkalinity'])

    def estimate_initial_rca(self, ta_target, ta_a):
        """Estimate RCA loading ignoring added Ca2+."""
        return self.mm_cao * (ta_target - ta_a) / (2 * self.frac_cao * self.frac_dissolved)

    def simulate_rca_effect(self, w_rca, ta_a, ca_a, sal_a, temp_K_a, tempK_r):
        """Simulate resulting solution properties after RCA addition."""
        w_cao_d = self.frac_cao * self.frac_dissolved * w_rca
        mol_cao = w_cao_d / self.mm_cao
        oh_added = 2 * mol_cao

        ta_new = ta_a + oh_added
        ca_new = ca_a + mol_cao
        sal_new = sal_a + mol_cao

        ta_umolkg = m_to_umol_per_kg(ta_new)
        ca_umolkg = m_to_umol_per_kg(ca_new)
        sal_ppt = sal_m_to_ppt(sal_new)

        # Find temperature change
        w_rcaF = self.frac_sellable_rca*(w_rca - w_cao_d)  # (g/L) final loading of rca remaining after the reaction
        tempK_ad = (-1*self.dH_cao*self.frac_cao + R_H2O*self.c_h2o*temp_K_a +w_rcaF*self.c_rca*tempK_r) / (R_H2O*self.c_h2o + w_rcaF*self.c_rca)
        tempC_ad = tempK_ad - 273.15  # (C) final temperature

        kwargs = dict(
            par1=self.seawater.dic_iu,
            par2=ta_umolkg,
            par1_type=2,
            par2_type=1,
            salinity=sal_ppt,
            temperature=tempC_ad,
            total_calcium=ca_umolkg
        )

        results = pyco2.sys(**kwargs)

        return results["pH"], ca_new, sal_new, ta_new, tempC_ad

    def find_optimal_rca_loading(self, start, ta_a, ca_a, sal_a, tempK_a, tempK_r, step=0.01, max_range=2):
        """Iteratively search for RCA loading that hits target pH."""
        w_vals = np.arange(start, start + max_range, step)
        for w in w_vals:
            pH, ca, sal, ta, tempC = self.simulate_rca_effect(w, ta_a, ca_a, sal_a, tempK_a, tempK_r)
            if abs(pH - self.seawater.pH_i) <= self.tolerance:
                return w, pH, ca, sal, ta, tempC
        return None, None, None, None, None, None

    def run(self):
        """Main execution method for RCA loading calculation."""
        if self.oae.acid_disposal_method != "sell rca":
            return

        # Step 1: Acid system before RCA
        ph_a, sal_a, sal_a_ppt, ta_a, ca_a, tempK_a, tempK_r = self.compute_acid_props()

        # Step 2: Estimate target TA
        ta_target = self.get_target_ta(sal_a_ppt)

        # Step 3: Estimate initial RCA loading (ignoring Ca2+)
        w_rca_init = self.estimate_initial_rca(ta_target, ta_a)

        # Step 4: Estimate pH from this guess
        pH_test, _, _, _, _ = self.simulate_rca_effect(w_rca_init, ta_a, ca_a, sal_a, tempK_a, tempK_r)

        # Step 5: Search for RCA loading that achieves target pH
        w_final, pH_final, ca_final, sal_final, ta_final, temp_final = self.find_optimal_rca_loading(
            w_rca_init, ta_a, ca_a, sal_a, tempK_a, tempK_r, step=0.01
        )

        # Step 6: Refine if not found
        if w_final is None:
            w_final, pH_final, ca_final, sal_final, ta_final, temp_final = self.find_optimal_rca_loading(
                w_rca_init, ta_a, ca_a, sal_a, tempK_a, tempK_r, step=0.001
            )

        # Save results to instance
        self._w_final = w_final
        self._pH_final = pH_final
        self._ca_final = ca_final
        self._sal_final = sal_final
        self._ta_final = ta_final
        self._sal_a = sal_a
        self._temp_final = temp_final
        self._temp_change = temp_final - (tempK_a-273.15)  # Temperature change in C

    @property
    def results(self) -> dict:
        """
        Structured dictionary of RCA calculation results.
        Returns:
            dict: RCA output values if available, otherwise None.
                - "rca_loading_g_per_L": Final RCA loading in g/L.
                - "final_pH": Final pH after RCA addition.
                - "final_Ca_M": Final calcium concentration in M.
                - "final_salinity_M": Final salinity in M.
                - "final_TA_M": Final total alkalinity in M.
                - "final_temp_C": Final temperature in °C.
                - "temp_change_C": Temperature change due to RCA addition in °C.
                - "sal_a": Salinity of acid before RCA addition in M.
        """
        if self._w_final is None:
            return None  # or raise an error if you want to enforce `.run()` first

        return {
            "rca_loading_g_per_L": self._w_final,
            "final_pH": self._pH_final,
            "final_Ca_M": self._ca_final,
            "final_salinity_M": self._sal_final,
            "final_TA_M": self._ta_final,
            "final_temp_C": self._temp_final,
            "temp_change_C": self._temp_change,
            "sal_a": self._sal_a,
        }

@define
class OAERangeOutputs:
    """
    A class to represent the ocean alkalinity enhancement (OAE) device power and chemical ranges under each scenario.

    Attributes:
        S1 (dict): Chemical and power ranges for scenario 1 (e.g., tank filled).
            - "volAcid": Volume of acid (L).
            - "volBase": Volume of base (L).
            - "volExcessAcid": Volume of excess acid (L).
            - "mol_OH": mol OH added to seawater at each time (mol).
            - "mol_HCl": mol HCl excess acid generated (mol).
            - "pH_f": Final pH of the solution.
            - "dic_f": Final dissolved inorganic carbon concentration (mol/L).
            - "ta_f": Final total alkalinity concentrations (mol/L).
            - "sal_f": Final salinity of the solution (ppt).
            - "temp_f": Final temperature of the solution (C).
            - "rca_power": Power required for RCA addition (W).
            - "ca_f": Final calcium concentration (mol/L).
            - "c_a": Concentration of acid (mol/L).
            - "c_b": Concentration of base (mol/L).
            - "Qin": Flow rate into the system (m³/s).
            - "Qout": Flow rate out of the system (m³/s).
            - "alkaline_solid_added": Amount of alkaline solid added for acid disposal (g).
            - "alkaline_to_acid": Ratio of alkaline solid to acid needed for neutralization.
            - "pwrRanges": Power range for the scenario (W).

        S2 (dict): Chemical and power ranges for scenario 2 (e.g., variable power and chemical ranges).
            - Same keys as in S1.

        S3 (dict): Chemical and power ranges for scenario 3 (e.g., ED not active, tanks not zeros).
            - Same keys as in S1.

        S4 (dict): Chemical and power ranges for scenario 4 (e.g., ED active, no capture).
            - Same keys as in S1.

        S5 (dict): Chemical and power ranges for scenario 5 (reserved for special cases).
            - Same keys as in S1.

        P_minS1_tot (float): Minimum power for scenario 1.
        P_minS2_tot (float): Minimum power for scenario 2.
        P_minS3_tot (float): Minimum power for scenario 3.
        P_minS4_tot (float): Minimum power for scenario 4.
        V_aT_max (float): Maximum volume of acid in the tank.
        V_bT_max (float): Maximum volume of base in the tank.
        V_b3_min (float): Minimum volume of base required for S3.
        N_range (int): Number of ED units active in S1, S3, S4.
        S2_tot_range (int): Number of ED units active in S2.
        pump_power_min (float): Minimum pump power in MW.
        pump_power_max (float): Maximum pump power in MW.
    """

    S1: dict
    S2: dict
    S3: dict
    S4: dict
    S5: dict
    P_minS1_tot: float
    P_minS2_tot: float
    P_minS3_tot: float
    P_minS4_tot: float
    V_bT_max: float
    V_aT_max: float
    V_b3_min: float
    N_range: int
    S2_tot_range: int
    S2_ranges: np.ndarray
    pump_power_min: float
    pump_power_max: float
    pumps: PumpOutputs

def initialize_power_chemical_ranges(
    oae_config: OAEInputs,
    pump_config: PumpInputs,
    seawater_config: SeaWaterInputs,
    rca: RCALoadingCalculator,
) -> OAERangeOutputs:
    """
    Initialize the power and chemical ranges for an ocean alkalinity enhancement (OAE) system under various scenarios.

    This function calculates the power and chemical usage ranges for an OAE system across five distinct scenarios:

    1. Scenario 1: Tanks Full & ED unit Active

    2. Scenario 2: Capture CO2 & Fill Tank

    3. Scenario 3: ED not active, tanks not zeros*

    4. Scenario 4: ED active, no capture

    5. Scenario 5: All input power is excess

    Args:
        oae_config (OAEInputs): Configuration parameters for the OAE system.
        pump_config (PumpInputs): Configuration parameters for the pumping system.
        seawater_config (SeaWaterInputs): Configuration parameters for the seawater used in the process.

    Returns:
        OAERangeOutputs: An object containing the power and chemical ranges for each scenario.
    """

    N_edMin = oae_config.N_edMin
    N_edMax = oae_config.N_edMax
    P_ed1 = oae_config.P_ed1
    Q_ed1 = oae_config.Q_ed1
    E_HCl = oae_config.E_HCl
    E_NaOH = oae_config.E_NaOH
    dic_i = seawater_config.dic_i
    h_i = seawater_config.h_i
    ta_i = seawater_config.ta_i
    kw = seawater_config.kw
    pH_i = seawater_config.pH_i

    # Initialize RCA
    rca = RCALoadingCalculator(oae=oae_config, seawater=seawater_config)
    rca.run()

    # Define the range sizes
    N_range = N_edMax - N_edMin + 1
    S2_tot_range = (N_range * (N_range + 1)) // 2

    S2_ranges = np.zeros((S2_tot_range, 2))

    # Fill the ranges array
    k = 0
    for i in range(N_range):
        for j in range(N_range - i):
            S2_ranges[k, 0] = i + N_edMin
            S2_ranges[k, 1] = j
            k += 1

    # Define the array names
    keys = [
        "volAcid",
        "volBase",
        "volExcessAcid",
        "mol_OH",
        "mol_HCl",
        "mass_CO2_absorbed",
        "pH_f",
        "dic_f",
        "ta_f",
        "sal_f",
        "temp_f",
        "rca_power",
        "ca_f",
        "c_a",
        "c_b",
        "Qin",
        "Qout",
        "alkaline_solid_added",
        "alkaline_to_acid",
        "pwrRanges",
    ]

    # Initialize the dictionaries
    S1 = {key: np.zeros(N_range) for key in keys}
    S2 = {key: np.zeros(S2_tot_range) for key in keys}
    S3 = {key: np.zeros(N_range) for key in keys}
    S4 = {key: np.zeros(N_range) for key in keys}
    S5 = {key: np.zeros(1) for key in keys}

    p = initialize_pumps(oae_config=oae_config, pump_config=pump_config)

    ########################## Chemical & Power Ranges: S1, S3, S4 ###################################
    for i in range(N_range):
        ############################### S1: Chem Ranges: Tank Full #####################################
        P_EDi = (i + N_edMin) * P_ed1  # ED unit power requirements
        p.pumpED.Q = (i + N_edMin) * Q_ed1  # Flow rates for ED Units
        p.pumpO.Q = (
            1 / oae_config.frac_EDflow * p.pumpED.Q
        )  # Flow rate for seawater
        S1["Qin"][i] = p.pumpO.Q  # (m3/s) Intake

        # Acid and Base Concentrations
        p.pumpA.Q = p.pumpED.Q * oae_config.frac_acidFlow  # Acid flow rate
        C_a = (1 / p.pumpA.Q) * (
            P_EDi / (3600 * (E_HCl * 1000)) - (p.pumpED.Q * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        if C_a > seawater_config.SAL_i_mol_m3:
            C_a = seawater_config.SAL_i_mol_m3  # Limit acid concentration to seawater salinity
            warnings.warn(
                f"S1: Acid concentration exceeds seawater salinity. Limiting to {C_a:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_a = (seawater_config.SAL_i_mol_m3 - C_a) * p.pumpB.Q # (mol/s) NaCl needed to maintain salinity
        S1["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        S1["volExcessAcid"][i] = p.pumpA.Q * 3600 # (m3) all acid made is excess
        S1["mol_HCl"][i] = C_a * p.pumpA.Q * 3600 # (mol) Excess acid generated
        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q  # Base flow rate
        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units

        if C_b > seawater_config.SAL_i_mol_m3:
            C_b = seawater_config.SAL_i_mol_m3
            warnings.warn(
                f"S1: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i_mol_m3 - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
        S1["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units
        S1["mol_OH"][i] = C_b * p.pumpB.Q * 3600 # (mol) OH added to seawater
        S1["mass_CO2_absorbed"][i] = (oae_config.assumed_CDR_rate * S1["mol_OH"][i]*44/ 1000) # (kg) CO2 absorbed by the system

        # Base Addition
        p.pumpI.Q = p.pumpO.Q - p.pumpED.Q  # Intake remaining after diversion to ED
        
        # Find TA Before Base Addition
        TA_i = seawater_config.ta_i * 1000  # (mol/m3)

        # Find TA After Base Addition
        TA_f = (TA_i * p.pumpI.Q + C_b * p.pumpB.Q) / (
            p.pumpI.Q + p.pumpB.Q
        )  # (mol/m3)

        S1["ta_f"][i] = TA_f / 1000 # (mol/L) Total alkalinity after base addition

        # Find effluent chem after base addition
        ta_fu = m_to_umol_per_kg(S1["ta_f"][i]) 
        SAL_f = (SAL_b * p.pumpB.Q + seawater_config.SAL_i_mol_m3 * p.pumpI.Q) / (
            p.pumpB.Q + p.pumpI.Q
        )
        S1["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # (ppt) Salinity after base addition

        # Find temperature change
        tempC_mix = (p.pumpI.Q * seawater_config.tempC_i + p.pumpB.Q
                     * (seawater_config.tempC_i + seawater_config.ed_temp_delta)
                     ) / (p.pumpI.Q + p.pumpB.Q)

        # Define input conditions for mixing the base and the brine
        kwargs = dict(
            par1 = ta_fu, # Total alkalinity in umol/kg
            par2 = seawater_config.dic_iu, # DIC in umol/kg
            par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
            par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = S1["sal_f"][i],  # Salinity of the sample (ppt)
            temperature = tempC_mix,  # Temperature at input conditions (C)
        )
        results = pyco2.sys(**kwargs)
        S1["dic_f"][i] = dic_i # (mol/L) DIC after base addition
        ca_mix = umol_per_kg_to_m(results["total_calcium"]) # (mol/L) Calcium after base addition
        

        if oae_config.acid_disposal_method == "sell rca":
            ## The neutralized acid is mixed with the alkaline seawater
            # Find amount of RCA added
            S1["alkaline_to_acid"][i] = rca.results["rca_loading_g_per_L"] # (g/L) loading of RCA mass to volume of acid
            S1["alkaline_solid_added"][i] = rca.results["rca_loading_g_per_L"] * S1["volExcessAcid"][i] * 1000 # (g) total alkaline solid added to neutralize acid
            slurry_mass = S1["alkaline_solid_added"][i] + S1["volExcessAcid"][i] * R_H2O * 1000 # (g) total mass of acid slurry
            S1["rca_power"][i] = rca.Wpg_rca * slurry_mass # (W) power needed to tumble the RCA and acid

            # Find new TA
            TA_rf = rca.results["final_TA_M"] * 1000 # (mol/m3) The TA of the neutralized acid
            TA_f = (TA_i * p.pumpI.Q + C_b * p.pumpB.Q + TA_rf*p.pumpA.Q)/(p.pumpI.Q + p.pumpB.Q + p.pumpA.Q) # (mol/m3)
            S1["ta_f"][i] = TA_f/1000 # (mol/L)
            ta_fu = m_to_umol_per_kg(S1["ta_f"][i])
            
            # Find new salinity
            SAL_a = rca.results["sal_a"] * 1000 # (mol/m3) salinity of the acid
            SAL_f = (SAL_b*p.pumpB.Q + seawater_config.SAL_i_mol_m3*p.pumpI.Q + SAL_a*p.pumpA.Q)/(p.pumpB.Q + p.pumpI.Q + p.pumpA.Q)
            S1["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # ppt
            
            # Find new calcium concentration
            CA_mix = ca_mix*1000 # (mol/m3) calcium concentration of alkaline seawater
            CA_rf = rca.results["final_Ca_M"]*1000 # (mol/m3) calcium concentration of neutralized seawater
            CA_f =  (CA_mix*(p.pumpI.Q + p.pumpB.Q) + CA_rf*p.pumpA.Q)/(p.pumpI.Q+p.pumpB.Q+p.pumpA.Q) # (mol/m3) final calcium concentration
            S1["ca_f"][i] = CA_f/1000 # (mol/L) final calcium concentration
            ca_fu = m_to_umol_per_kg(S1["ca_f"][i]) # (umol/kg)

            # Find new temperature
            tempC_mix = (p.pumpI.Q * seawater_config.tempC_i + p.pumpB.Q
                        * (seawater_config.tempC_i + seawater_config.ed_temp_delta)
                        + p.pumpA.Q *rca.results["final_temp_C"]
                        ) / (p.pumpI.Q + p.pumpB.Q + p.pumpA.Q)

            # Define input conditions for mixing the alkaline seawater and neutralized acid
            kwargs = dict(
                par1 = ta_fu,  # Value of the first parameter (TA in umol/kg)
                par2 = seawater_config.dic_iu,  # Value of the second parameter DIC in umol/kg
                par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
                par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
                salinity = S1["sal_f"][i],  # Salinity of the sample (ppt)
                temperature = tempC_mix,  # Temperature at input conditions (C)
                total_calcium = ca_fu, # Calcium concentration in umol/kg
            )
            # Results from mixing
            results = pyco2.sys(**kwargs)
            S1["pH_f"][i] = results['pH']

            # Outtake
            p.pumpF.Q = p.pumpI.Q + p.pumpB.Q + p.pumpA.Q # (m3/s) Outtake flow rate
            S1["Qout"][i] = p.pumpF.Q # (m3/s) Outtake
            S1["temp_f"][i] = tempC_mix # (C) Outtake temperature

            # Power ranges for S1
            S1["pwrRanges"][i] = (
                P_EDi
                + p.pumpED.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
                + S1["rca_power"][i]
            )

        else:
            # If acid disposal method is not to sell RCA, then the acid is neutralized with base
            S1["pH_f"][i] = results["pH"] # (unitless) pH after base addition
            S1["ca_f"][i] = ca_mix # (mol/L) Calcium after base addition

            # Outtake
            p.pumpF.Q = p.pumpI.Q + p.pumpB.Q  # (m3/s) Outtake flow rate
            S1["Qout"][i] = p.pumpF.Q  # (m3/s) Outtake
            S1["temp_f"][i] = tempC_mix # (C) Temperature after base addition

            # Power ranges for S1
            S1["pwrRanges"][i] = (
                P_EDi
                + p.pumpED.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
            )

        P_minS1_tot = min(S1["pwrRanges"])

        ############################### S3: Chem Ranges: ED not active, tanks not zeros ##################
        P_EDi = 0  # ED Unit is off
        p.pumpED.Q = 0  # ED Unit is off
        p.pumpO.Q = (
            (1 / oae_config.frac_EDflow - 1) * (i + N_edMin) * Q_ed1
        )  # Flow rates for intake based on equivalent ED units that would be active
        S3["Qin"][i] = p.pumpO.Q
        p.pumpI.Q = p.pumpO.Q  # since no flow is going to the ED unit
        p.pumpB.Q = (
            (i + N_edMin) * Q_ed1 * oae_config.frac_baseFlow
        )  # Flow rate for base pump based on equivalent ED units that would be active
        p.pumpA.Q = 0 # No waste acid is pumped in this case

        # Change in volume due to acid and base use
        # S3["volAcid"][i] = -p.pumpA.Q * 3600 # (m3) volume of acid lost by the tank
        S3["volBase"][i] = -p.pumpB.Q * 3600  # (m3) volume of base lost by the tank

        # The concentration of acid and base produced does not vary with flow rate
        # Also does not vary with power since the power for the ED units scale directly with the flow rate
        C_b = (1 / p.pumpB.Q_min) * (
            P_ed1 * N_edMin / (3600 * (E_NaOH * 1000))
            - (p.pumpED.Q_min * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        if C_b > seawater_config.SAL_i_mol_m3:
            C_b = seawater_config.SAL_i_mol_m3
            warnings.warn(
                f"S3: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i_mol_m3 - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
        
        S3["c_b"][i] = C_b / 1000  # (mol/L) Base concentration used in S3
        S3["mol_OH"][i] = p.pumpB.Q * C_b * 3600 # (mol) OH added to seawater in S3
        S3["mass_CO2_absorbed"][i] = (oae_config.assumed_CDR_rate * S3["mol_OH"][i]*44/ 1000) # (kg) CO2 absorbed by the system

        # Find TA Before Base Addition
        TA_i = ta_i * 1000  # (mol/m3)
        # Find TA After Base Addition
        TA_f = (TA_i * p.pumpI.Q + C_b * p.pumpB.Q) / (
            p.pumpI.Q + p.pumpB.Q
        )  # (mol/m3)

        S3["ta_f"][i] = TA_f / 1000 # (mol/L) Total alkalinity after base addition

        # Find effluent chem after base addition
        ta_fu = m_to_umol_per_kg(S3["ta_f"][i]) 
        SAL_f = (SAL_b * p.pumpB.Q + seawater_config.SAL_i_mol_m3 * p.pumpI.Q) / (
            p.pumpB.Q + p.pumpI.Q
        )
        S3["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # (ppt) Salinity after base addition

        # Find temperature change
        tempC_mix = (p.pumpI.Q * seawater_config.tempC_i + p.pumpB.Q
                     * (seawater_config.tempC_i + seawater_config.ed_temp_delta)
                     ) / (p.pumpI.Q + p.pumpB.Q)

        # Define input conditions for mixing the base and the brine
        kwargs = dict(
            par1 = ta_fu, # Total alkalinity in umol/kg
            par2 = seawater_config.dic_iu, # DIC in umol/kg
            par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
            par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = S3["sal_f"][i],  # Salinity of the sample (ppt)
            temperature = tempC_mix,  # Temperature at input conditions (C)
        )
        results = pyco2.sys(**kwargs)
        S3["dic_f"][i] = dic_i # (mol/L) DIC after base addition
        ca_mix = umol_per_kg_to_m(results["total_calcium"]) # (mol/L) Calcium after base addition

        if oae_config.acid_disposal_method == "sell rca":
             # Determine amount of acid pumped from tank, based on amount of base pumped
            p.pumpA.Q = (i+N_edMin)*Q_ed1*oae_config.frac_acidFlow
            S3["volAcid"][i] = -p.pumpA.Q * 3600 # (m3) volume of acid lost by the tank

            ## The neutralized acid is mixed with the alkaline seawater
            # Find amount of RCA added
            S3["alkaline_to_acid"][i] = rca.results["rca_loading_g_per_L"] # (g/L) loading of RCA mass to volume of acid
            S3["alkaline_solid_added"][i] = rca.results["rca_loading_g_per_L"]*-S3["volAcid"][i]*1000 # (g) mass of RCA used to neutralize the acid
            slurry_mass = S3["alkaline_solid_added"][i] + -S3["volAcid"][i]*R_H2O*1000 # (g) mass of the RCA and acid slurry
            S3["rca_power"][i] = rca.Wpg_rca * slurry_mass  # (W) power needed to tumble the RCA and acid
            
            # Find new TA
            TA_rf = rca.results["final_TA_M"] * 1000 # (mol/m3) The TA of the neutralized acid
            TA_f = (TA_i * p.pumpI.Q + C_b * p.pumpB.Q + TA_rf*p.pumpA.Q)/(p.pumpI.Q + p.pumpB.Q + p.pumpA.Q) # (mol/m3)
            S3["ta_f"][i] = TA_f/1000 # (mol/L)
            ta_fu = m_to_umol_per_kg(S3["ta_f"][i])
            
            # Find new salinity
            SAL_a = rca.results["sal_a"] * 1000 # (mol/m3) salinity of the acid
            SAL_f = (SAL_b * p.pumpB.Q + seawater_config.SAL_i_mol_m3 * p.pumpI.Q + SAL_a* p.pumpA.Q)/(
                p.pumpB.Q + p.pumpI.Q + p.pumpA.Q)
            S3["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # ppt
            
            # Find new calcium concentration
            CA_mix = ca_mix*1000 # (mol/m3) calcium concentration of alkaline seawater
            CA_rf = rca.results["final_Ca_M"]*1000 # (mol/m3) calcium concentration of neutralized seawater
            CA_f =  (CA_mix*(p.pumpI.Q + p.pumpB.Q) + CA_rf*p.pumpA.Q)/(p.pumpI.Q+p.pumpB.Q+p.pumpA.Q) # (mol/m3) final calcium concentration
            S3["ca_f"][i] = CA_f/1000 # (mol/L) final calcium concentration
            ca_fu = m_to_umol_per_kg(S3["ca_f"][i]) # (umol/kg)

            # Find new temperature
            tempC_mix = (p.pumpI.Q*seawater_config.tempC_i 
                         + p.pumpB.Q
                         *(seawater_config.tempC_i+seawater_config.ed_temp_delta)
                         + p.pumpA.Q * rca.results["final_temp_C"]
                         )/(p.pumpI.Q + p.pumpB.Q + p.pumpA.Q)

            # Define input conditions for mixing the alkaline seawater and neutralized acid
            kwargs = dict(
                par1 = ta_fu,  # Value of the first parameter (TA in umol/kg)
                par2 = seawater_config.dic_iu,  # Value of the second parameter DIC in umol/kg
                par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
                par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
                salinity = S3["sal_f"][i],  # Salinity of the sample (ppt)
                temperature = tempC_mix,  # Temperature at input conditions (C)
                total_calcium = ca_fu, # Calcium concentration in umol/kg
            )
            # Results from mixing
            results = pyco2.sys(**kwargs)
            S3["pH_f"][i] = results['pH']

            # Outtake
            p.pumpF.Q = p.pumpI.Q + p.pumpB.Q + p.pumpA.Q # (m3/s) Outtake flow rate
            S3["Qout"][i] = p.pumpF.Q # (m3/s) Outtake
            S3["temp_f"][i] = tempC_mix # (C) Outtake temperature

            # Power ranges for S3
            S3["pwrRanges"][i] = (
                P_EDi
                + p.pumpED.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
                + S3["rca_power"][i]
            )

        else:
            S3["pH_f"][i] = results["pH"] # (unitless) pH after base addition
            S3["ca_f"][i] = ca_mix # (mol/L) Calcium after base addition

            # Outtake
            p.pumpF.Q = p.pumpI.Q + p.pumpB.Q  # (m3/s) Outtake flow rate
            S3["Qout"][i] = p.pumpF.Q
            S3["temp_f"][i] = tempC_mix # (C) Outtake temperature

            # Power ranges for S3
            S3["pwrRanges"][i] = (
                P_EDi
                + p.pumpED.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
            )
        P_minS3_tot = min(S3["pwrRanges"])

        ############################### S4: Chem Ranges: ED active, no capture ###########################
        P_EDi = (i + N_edMin) * P_ed1  # ED unit power requirements
        p.pumpED.Q = 0  # Regular ED pump is inactive here
        p.pumpED4.Q = (i + N_edMin) * Q_ed1  # ED pump with filtration pressure

        # Acid and base concentrations
        p.pumpA.Q = p.pumpED4.Q * oae_config.frac_acidFlow  # Acid flow rate
        p.pumpB.Q = p.pumpED4.Q - p.pumpA.Q  # Base flow rate
        C_a = (1 / p.pumpA.Q) * (
            P_EDi / (3600 * (E_HCl * 1000)) - (p.pumpED4.Q * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        if C_a > seawater_config.SAL_i_mol_m3:
            C_a = seawater_config.SAL_i_mol_m3  # Limit acid concentration to seawater salinity
            warnings.warn(
                f"S4: Acid concentration exceeds seawater salinity. Limiting to {C_a:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_a = (seawater_config.SAL_i_mol_m3 - C_a) * p.pumpB.Q # (mol/s) NaCl needed to maintain salinity
        S4["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED4.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        if C_b > seawater_config.SAL_i_mol_m3:
            C_b = seawater_config.SAL_i_mol_m3
            warnings.warn(
                f"S4: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i_mol_m3 - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
        S4["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units

        # Excess acid generated
        S4["volExcessAcid"][i] = p.pumpA.Q * 3600 # (m3) volume of excess acid after timestep
        S4["mol_HCl"][i] = C_a * p.pumpA.Q * 3600  # (mol) moles of excess acid generated

        # Base added to the tank
        # n_bT = C_b * p.pumpB.Q  # (mol/s) rate of base moles added to tank
        S4["volBase"][i] = p.pumpB.Q * 3600  # volume of base in tank after time step

        # Intake (ED4 pump not O pump is used)
        p.pumpO.Q = 0  # Need intake for ED & min CC
        S4["Qin"][i] = p.pumpED4.Q  # (m3/s) Intake

        # Other pumps not used
        p.pumpI.Q = 0  # Intake remaining after diversion to ED

        # Outtake
        p.pumpF.Q = 0  # Outtake flow rate
        S4["Qout"][i] = p.pumpF.Q  # (m3/s) Outtake

        # Since no OAE is conducted the final DIC and pH is the same as the initial
        S4["pH_f"][i] = pH_i
        S4["dic_f"][i] = dic_i  # (mol/L)
        S4["sal_f"][i] = seawater_config.sal_ppt_i # (ppt) 
        S4["ta_f"][i] = ta_i # (mol/L)
        S4["temp_f"][i] = seawater_config.tempC_i # (C)
        S4["ca_f"][i] = seawater_config.ca_i # (mol/L)

        if oae_config.acid_disposal_method == "sell rca":
            S4["volAcid"][i] = S4["volExcessAcid"][i]  # (m3) all acid is excess
            S4["rca_power"][i] = 0  # (W) no power needed for RCA since no acid is neutralized
            # Power ranges for S4
            S4["pwrRanges"][i] = (
                P_EDi
                + p.pumpED4.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
                + S4["rca_power"][i]
            )

        else:
            # Power ranges for S4
            S4["pwrRanges"][i] = (
                P_EDi
                + p.pumpED4.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
            )
        P_minS4_tot = min(S4["pwrRanges"])

    ################################ Chemical & Power Ranges: S2 #####################################
    for i in range(S2_tot_range):
        ##################### S2: Chem Ranges Avoiding Overflow: Capture CO2 & Fill Tank #################
        # ED Unit Characteristics
        N_edi = S2_ranges[i, 0] + S2_ranges[i, 1]
        P_EDi = (N_edi) * P_ed1  # ED unit power requirements
        p.pumpED.Q = (N_edi) * Q_ed1  # Flow rates for ED Units

        # Acid and Base Creation
        p.pumpA.Q = p.pumpED.Q * oae_config.frac_acidFlow  # Acid flow rate
        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q  # Base flow rate
        C_a = (1 / p.pumpA.Q) * (
            P_EDi / (3600 * (E_HCl * 1000)) - (p.pumpED.Q * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        if C_a > seawater_config.SAL_i_mol_m3:
            C_a = seawater_config.SAL_i_mol_m3
            warnings.warn(
                f"S2: Acid concentration exceeds seawater salinity. Limiting to {C_a:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_a = (seawater_config.SAL_i_mol_m3 - C_a) * p.pumpB.Q  # (mol/s) NaCl needed to maintain salinity
        S2["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        S2["mol_HCl"][i] = C_a * p.pumpA.Q * 3600  # (mol) Excess acid generated

        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        if C_b > seawater_config.SAL_i_mol_m3:
            C_b = seawater_config.SAL_i_mol_m3
            warnings.warn(
                f"S2: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i_mol_m3 - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
    
        S2["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units

        # Excess acid
        S2["volExcessAcid"][i] = p.pumpA.Q * 3600 # (m3) all acid is excess

        # Amount of base added for OAE
        Q_bOAE = S2_ranges[i, 0] * Q_ed1 * oae_config.frac_baseFlow  # flow rate used for OAE
        S2["mol_OH"][i] = C_b * Q_bOAE * 3600  # (mol) OH added to seawater
        S2["mass_CO2_absorbed"][i] = (oae_config.assumed_CDR_rate * S2["mol_OH"][i]*44/ 1000) # (kg) CO2 absorbed by the system

        # Base addition to tank
        Q_bT = p.pumpB.Q - Q_bOAE  # (m3/s) flow rate of base to tank
        #n_bT = C_b * Q_bT  # (mol/s) rate of base moles added to tank
        S2["volBase"][i] = Q_bT * 3600  # (m3) base added to tank

        # Base addition to seawater
        p.pumpI.Q = (1 / oae_config.frac_EDflow - 1) *  Q_bOAE / oae_config.frac_baseFlow

        # Seawater Intake
        p.pumpO.Q = p.pumpI.Q + p.pumpED.Q   # total seawater intake
        S2["Qin"][i] = p.pumpO.Q  # (m3/s) intake

        # Find TA Before Base Addition
        TA_i = ta_i * 1000  # (mol/m3)

        # Find TA After Base Addition
        TA_f = (TA_i * p.pumpI.Q + C_b * Q_bOAE)/(p.pumpI.Q + Q_bOAE) # (mol/m3)
        S2["ta_f"][i] = TA_f/1000 # (mol/L)

        # Find effluent chem after base addition
        ta_fu = m_to_umol_per_kg(S2["ta_f"][i]) 
        SAL_f = (SAL_b * Q_bOAE + seawater_config.SAL_i_mol_m3 * p.pumpI.Q) / (
            Q_bOAE + p.pumpI.Q
        )
        S2["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # (ppt) Salinity after base addition

        # Find temperature change
        tempC_mix = (p.pumpI.Q*seawater_config.tempC_i + Q_bOAE*(seawater_config.tempC_i+seawater_config.ed_temp_delta))/(p.pumpI.Q + Q_bOAE)

        # Define input conditions for mixing the base and the brine
        kwargs = dict(
            par1 = ta_fu, # Total alkalinity in umol/kg
            par2 = seawater_config.dic_iu, # DIC in umol/kg
            par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
            par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = S2["sal_f"][i],  # Salinity of the sample (ppt)
            temperature = tempC_mix,  # Temperature at input conditions (C)
        )
        results = pyco2.sys(**kwargs)
        
        S2["dic_f"][i] = dic_i # (mol/L) DIC after base addition
        ca_mix = umol_per_kg_to_m(results["total_calcium"]) # (mol/L) Calcium after base addition

        if oae_config.acid_disposal_method == "sell rca":
            # Find amount of acid that goes to tanks for temporary storage vs what is neutralized
            # For consistency a similar amount of acid will be stored as the base
            Q_aRCA = S2_ranges[i,0] * Q_ed1*oae_config.frac_acidFlow # flow rate used for RCA acid disposal
            Q_aT = p.pumpA.Q - Q_aRCA # flow rate of acid going to the tank
            S2["volAcid"][i] = Q_aT * 3600 # (m3) acid added to tank
            V_aRCA = S2["volExcessAcid"][i] - S2["volAcid"][i] # (m3) volume of acid used for RCA

            ## The neutralized acid is mixed with the alkaline seawater
            # Find amount of RCA added
            S2["alkaline_to_acid"][i] = rca.results["rca_loading_g_per_L"] # (g/L) loading of RCA mass to volume of acid
            S2["alkaline_solid_added"][i] = rca.results["rca_loading_g_per_L"]*V_aRCA*1000 # (g) mass of RCA used to neutralize the acid
            slurry_mass = S2["alkaline_solid_added"][i] + V_aRCA*R_H2O*1000 # (g) mass of the RCA and acid slurry
            S2["rca_power"][i] = rca.Wpg_rca*slurry_mass # (W) power needed to tumble the RCA

            # Find new TA
            TA_rf = rca.results["final_TA_M"] * 1000 # (mol/m3) The TA of the neutralized acid
            TA_f = (TA_i * p.pumpI.Q + C_b * Q_bOAE + TA_rf*Q_aRCA)/(p.pumpI.Q + Q_bOAE + Q_aRCA) # (mol/m3)
            S2["ta_f"][i] = TA_f/1000 # (mol/L)
            ta_fu = m_to_umol_per_kg(S2["ta_f"][i])

            # Find new salinity
            SAL_a = rca.results["sal_a"] * 1000 # (mol/m3) salinity of the acid
            SAL_f = (SAL_b*Q_bOAE + seawater_config.SAL_i_mol_m3 * p.pumpI.Q + SAL_a*Q_aRCA)/(
                Q_bOAE + p.pumpI.Q + Q_aRCA)
            S2["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # ppt
            
            # Find new calcium concentration
            CA_mix = ca_mix*1000 # (mol/m3) calcium concentration of alkaline seawater
            CA_rf = rca.results["final_Ca_M"]*1000  # (mol/m3) calcium concentration of neutralized seawater
            CA_f =  (CA_mix*(p.pumpI.Q + Q_bOAE) + CA_rf*Q_aRCA)/(p.pumpI.Q+Q_bOAE+Q_aRCA) # (mol/m3) final calcium concentration
            S2["ca_f"][i] = CA_f/1000 # (mol/L) final calcium concentration
            ca_fu = m_to_umol_per_kg(S2["ca_f"][i]) # (umol/kg)

            # Find new temperature
            tempC_mix = (p.pumpI.Q*seawater_config.tempC_i 
                        + Q_bOAE
                        * (seawater_config.tempC_i+seawater_config.ed_temp_delta)
                        + Q_aRCA*rca.results["final_temp_C"]
                        )/(p.pumpI.Q + Q_bOAE + Q_aRCA)

            # Define input conditions for mixing the base and brine
            kwargs = dict(
                par1 = ta_fu,  # Value of the first parameter (TA in umol/kg)
                par2 = seawater_config.dic_iu,  # Value of the second parameter DIC in umol/kg
                par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
                par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
                salinity = S2["sal_f"][i],  # Salinity of the sample (ppt)
                temperature = tempC_mix,  # Temperature at input conditions (C)
                total_calcium = ca_fu, # Calcium concentration in umol/kg
            )
            # Results from mixing
            results = pyco2.sys(**kwargs)
            
            # pH after base addition
            S2["pH_f"][i] = results['pH']

            # Outtake
            p.pumpF.Q = p.pumpI.Q + Q_bOAE + Q_aRCA # (m3/s) Outtake flow rate
            S2["Qout"][i] = p.pumpF.Q # (m3/s) Outtake
            S2["temp_f"][i] = tempC_mix # (C) Outtake temperature

            # Power ranges for S2
            S2["pwrRanges"][i] = (
                P_EDi
                + p.pumpED.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
                + S2["rca_power"][i]
            )

        else:
            S2["pH_f"][i] = results["pH"] # (unitless) pH after base addition
            S2["ca_f"][i] = ca_mix # (mol/L) Calcium after base addition

            # Outtake
            p.pumpF.Q = p.pumpI.Q + Q_bOAE  # (m3/s) Outtake flow rate
            S2["Qout"][i] = p.pumpF.Q  # (m3/s) Outtake
            S2["temp_f"][i] = tempC_mix # (C) Outtake temperature

            # Power ranges for S2
            S2["pwrRanges"][i] = (
                P_EDi
                + p.pumpED.power()
                + p.pumpO.power()
                + p.pumpA.power()
                + p.pumpI.power()
                + p.pumpB.power()
                + p.pumpF.power()
            )
        P_minS2_tot = min(S2["pwrRanges"])

    ##################### S5: Chem Ranges: When all input power is excess ############################
    S5["volExcessAcid"] = 0  # No acid generated
    S5["volBase"] = 0  # No base generated
    S5["mol_OH"] = 0  # No base addition
    S5["mass_CO2_absorbed"] = 0  # No CO2 absorbed
    S5["mol_HCl"] = 0  # No excess acid generated
    S5["pH_f"] = pH_i  # No changes in sea pH
    S5["dic_f"] = dic_i  # (mol/L) No changes in sea DIC
    S5["ta_f"] = ta_i  # (mol/L) No changes in sea TA
    S5["sal_f"] = seawater_config.sal_ppt_i  # (ppt) No changes in sea salinity
    S5["c_a"] = (
        h_i  # (mol/L) No acid generated so acid concentration is the same as that of seawater
    )
    S5["c_b"] = (
        kw / h_i
    )  # (mol/L) No base generated so base concentration is the same as that of seawater
    S5["Qin"] = 0  # (m3/s) No intake
    S5["Qout"] = 0  # (m3/s) No outtake
    S5["alkaline_solid_added"] = 0  # No alkaline solid added
    S5["alkaline_to_acid"] = 0  # No alkaline solid to acid ratio
    S5["temp_f"] = seawater_config.tempC_i  # (C) No changes in sea temperature
    S5["ca_f"] = seawater_config.ca_i  # (mol/L) No changes in sea calcium
    S5["rca_power"] = 0  # (W) No power needed for RCA since no acid is neutralized

    # Define Tank Max Volumes (note there are two but they have the same volume)
    V_bT_max = p.pumpED.Q_min * oae_config.frac_baseFlow * oae_config.store_hours * 3600  # (m3) enables enough storage for 1 day or the hours from storeTime

    V_aT_max = p.pumpED.Q_min * oae_config.frac_acidFlow * oae_config.store_hours * 3600

    # Volume needed for S3
    V_b3_min = p.pumpED.Q_min * oae_config.frac_baseFlow * 3600  # enables minimum OAE for 1 timestep

    # Pump Power Ranges
    pump_power_min = (
        p.pumpO.P_min
        + p.pumpI.P_min
        + p.pumpED.P_min
        + p.pumpA.P_min
        + p.pumpB.P_min
        + p.pumpF.P_min
    )
    pump_power_max = (
        p.pumpO.P_max
        + p.pumpI.P_max
        + p.pumpED.P_max
        + p.pumpA.P_max
        + p.pumpB.P_max
        + p.pumpF.P_max
    )

    return OAERangeOutputs(
        S1=S1,
        S2=S2,
        S3=S3,
        S4=S4,
        S5=S5,
        P_minS1_tot=P_minS1_tot,
        P_minS2_tot=P_minS2_tot,
        P_minS3_tot=P_minS3_tot,
        P_minS4_tot=P_minS4_tot,
        V_bT_max=V_bT_max,
        V_aT_max=V_aT_max,
        V_b3_min=V_b3_min,
        N_range=N_range,
        S2_tot_range=S2_tot_range,
        S2_ranges=S2_ranges,
        pump_power_min=pump_power_min / 1e6,
        pump_power_max=pump_power_max / 1e6,
        pumps=p,
    )

@define
class OAEOutputs:
    """Outputs from the ocean alkalinity enhancement (OAE) process.

    Attributes:
        OAE_outputs (dict): Dictionary containing various output arrays from the OAE process.
            Keys include:
                - N_ed (array): Number of OAE units in operation at each time step.
                - P_xs (array): Excess power available at each time step (W).
                - volExcessAcid (array): Volume of excess acid at each time step (m³).
                - volBase (array): Volume of base added or removed from tanks at each time step (m³).
                - tank_vol_b (array): Volume of base in the tank at each time step (m³).
                - mol_OH (array): Moles of OH added to seawater at each time step (mol).
                - mol_HCl (array): Moles of excess acid generated at each time step (mol).
                - mass_CO2_absorbed (array): Mass of CO2 absorbed at each time step (kg).
                - pH_f (array): Final pH at each time step.
                - dic_f (array): Final dissolved inorganic carbon concentration at each time step.
                - ta_f (array): Final total alkalinity at each time step (mol/L).
                - sal_f (array): Final salinity at each time step (ppt).
                - c_a (array): Acid concentration at each time step (mol/L).
                - c_b (array): Base concentration at each time step (mol/L).
                - Qin (array): Intake flow rate at each time step (m³/s).
                - Qout (array): Outtake flow rate at each time step (m³/s).
                - S_t (array): Scenario number active at each time step (1-5).
                - alkaline_solid_added (array): Amount of alkaline solid added at each time step (g).
                - alkaline_to_acid (array): Ratio of alkaline solid to acid at each time step.
        mol_OH_yr (float): Total moles of OH added to seawater over the year (mol).
        pH_avg (float): Average pH of seawater after OAE over the year.
        dic_avg (float): Average dissolved inorganic carbon concentration after OAE over the year (mol/L).
        ta_avg (float): Average total alkalinity after OAE over the year (mol/L).
        sal_avg (float): Average salinity of seawater after OAE over the year (ppt).
        tempC_avg (float): Average temperature of seawater after OAE over the year (C).
        ca_avg (float): Average calcium concentration in seawater after OAE over the year (mol/L).
        volOAEbase_yr (float): Average volume of effluent base added to seawater over the year (m³).
        mol_OH_yr_MaxPwr (float): Total moles of OH added to seawater over the year under maximum power conditions (mol).
        mol_HCl_yr (float): Total moles of excess acid generated over the year (mol).
        volXSacid_yr (float): Total volume of excess acid generated over the year (m³).
        pH_HCl_excess (float): pH of seawater after excess acid addition.
        m_adSolid_yr (float): Total mass of alkaline solid added over the year (g).
        slurry_mass_max (float): Maximum mass of slurry in the tank (g).
        M_rev_yr (float): Mass of Products Made (tonnes/yr).
        M_disposed_yr (float): Mass of Products Disposed (tonnes/yr).
        X_disp (float): Cost of disposing of hazardous waste ($/ton).
        X_rev_yr (float): Value of Products Made ($/yr).
        M_co2est (float): Estimated carbon dioxide removal (CDR) over the year (tonnes).
        M_co2cap (float): Estimated maximum carbon dioxide removal over the year (tonnes).
        max_tank_fill_percent (float): Maximum percentage of the tank that was filled with acid during simulation.
        max_tank_fill_m3 (float): Maximum volume of the tank that was filled with acid during simulation (m³).
        overall_capacity_factor (float): Overall capcity factor (times system is on).
        oae_capacity_factor (float): Capacity factor of OAE. Total OAE compared to maximum possible OAE.
        energy_capacity_factor (float): Capacity factor of energy.
    """

    OAE_outputs: dict
    mol_OH_yr: float
    pH_avg: float
    dic_avg: float
    ta_avg: float
    sal_avg: float
    tempC_avg: float
    ca_avg: float
    volOAEbase_yr: float
    mol_OH_yr_MaxPwr: float
    mol_HCl_yr: float
    volXSacid_yr: float
    pH_HCl_excess: float
    m_adSolid_yr: float
    slurry_mass_max: float
    M_rev_yr: float
    M_disposed_yr: float
    X_disp: float
    X_rev_yr: float
    M_co2est: float
    M_co2cap: float
    max_tank_fill_percent: float
    max_tank_fill_m3: float
    overall_capacity_factor: float
    oae_capacity_factor: float
    energy_capacity_factor: float

def simulate_ocean_alkalinity_enhancement(
    ranges: OAERangeOutputs,
    oae_config: OAEInputs,
    seawater_config: SeaWaterInputs,
    rca: RCALoadingCalculator,
    power_profile,
    power_capacity,
    initial_tank_volume_m3,
):
    """
    Simulates the operation of an ocean alkalinity enhancement (OAE) system over time, given power availability and initial tank volumes.
    The simulation considers various scenarios based on the power profile and tank volumes, updating the state of the system
    at each time step.

    Parameters:
        ranges (OAERangeOutputs): The power and chemical ranges for different scenarios of OAE operation.
        oae_config (OAEInputs): Configuration inputs for the OAE system.
        seawater_config (SeaWaterInputs): Configuration inputs for the seawater properties.
        rca (RCALoadingCalculator): Calculator for the loading of alkaline solid in the RCA process.
        power_profile (np.ndarray): Array representing the available power at each time step (W).
        power_capacity (float): Maximum power capacity of the power system (W).
        initial_tank_volume_m3 (float): The initial volume of acid and base in the tanks (m³).

    Returns:
        OAEOutputs: A data class containing the simulation results, including the total CO2 captured,
        capacity factor, and yearly CO2 capture under actual and maximum power conditions.

    Notes:
        - The function evaluates five scenarios based on the available power and tank volumes, prioritizing CO2 capture and
          tank filling in the most effective way possible.
        - Scenario 5 is considered when all input power is excess, meaning no ED units are used.
    """
    N_edMin = oae_config.N_edMin

    tank_vol_b = np.zeros(len(power_profile) + 1)
    tank_vol_b[0] = round(initial_tank_volume_m3,2)

    tank_vol_a = np.zeros(len(power_profile)+1)
    tank_vol_a[0] = round(initial_tank_volume_m3,2)

    # Define the array names
    keys = [
        "N_ed",  # Number of ED units active
        "P_xs",  # (W) Excess power at each time
        "volExcessAcid",  # (m³) Volume of excess acid generated at each time
        "volAcid",  # (m³) Volume of acid added/removed to/from tanks at each time
        "volBase",  # (m³) Volume of base added/removed to/from tanks at each time
        "tank_vol_b",  # (m³) Volume of base in the tank at each time
        "tank_vol_a",  # (m³) Volume of acid in the tank at each time
        "mol_OH",  # (mol) Moles of OH added to seawater at each time
        "mol_HCl",  # (mol) Moles of excess acid generated at each time
        "mass_CO2_absorbed",  # (kg) Mass of CO2 absorbed at each time
        "pH_f",  # Final pH at each time
        "dic_f",  # (mol/L) Final DIC at each time
        "ta_f",  # (mol/L) Final total alkalinity at each time
        "sal_f",  # (ppt) Final salinity at each time
        "temp_f",  # (C) Final temperature at each time
        "ca_f",  # (mol/L) Final calcium concentration at each time
        "c_a",  # (mol/L) Acid concentration at each time step
        "c_b",  # (mol/L) Base concentration at each time step
        "Qin",  # (m³/s) Intake flow rate at each time step
        "Qout",  # (m³/s) Outtake flow rate at each time step
        "alkaline_solid_added",  # (g) Amount of alkaline solid added at each time step
        "alkaline_to_acid",  # Ratio of alkaline solid to acid at each time step
        "S_t",  # The scenario activated at each time step
    ]

    # Initialize the dictionaries
    OAE_outputs = {key: np.zeros(len(power_profile)) for key in keys}

    nON = 0  # Timesteps when capture occurs (S1-3) used to determine capacity factor

    for i in range(len(power_profile)):
        # Scenario 1:  Tanks Full and ED unit Active
        if power_profile[i] >= ranges.P_minS1_tot and tank_vol_b[i] == ranges.V_bT_max:
            # Find number of active units based on power
            for j in range(ranges.N_range):
                if power_profile[i] >= ranges.S1["pwrRanges"][j]:
                    i_ed = j  # determine how many ED units can be used
            OAE_outputs["N_ed"][i] = N_edMin + i_ed  # number of ED units active

            # Update recorded values based on number of ED units active
            OAE_outputs["volExcessAcid"][i] = ranges.S1["volExcessAcid"][i_ed]
            OAE_outputs["volBase"][i] = ranges.S1["volBase"][i_ed]
            OAE_outputs["volAcid"][i] = ranges.S1["volAcid"][i_ed]
            OAE_outputs["mol_OH"][i] = ranges.S1["mol_OH"][i_ed]
            OAE_outputs["mass_CO2_absorbed"][i] = ranges.S1["mass_CO2_absorbed"][i_ed]
            OAE_outputs["mol_HCl"][i] = ranges.S1["mol_HCl"][i_ed]
            OAE_outputs["pH_f"][i] = ranges.S1["pH_f"][i_ed]
            OAE_outputs["dic_f"][i] = ranges.S1["dic_f"][i_ed]
            OAE_outputs["ta_f"][i] = ranges.S1["ta_f"][i_ed]
            OAE_outputs["sal_f"][i] = ranges.S1["sal_f"][i_ed]
            OAE_outputs["temp_f"][i] = ranges.S1["temp_f"][i_ed]
            OAE_outputs["ca_f"][i] = ranges.S1["ca_f"][i_ed]
            OAE_outputs["c_a"][i] = ranges.S1["c_a"][i_ed]
            OAE_outputs["c_b"][i] = ranges.S1["c_b"][i_ed]
            OAE_outputs["Qin"][i] = ranges.S1["Qin"][i_ed]
            OAE_outputs["Qout"][i] = ranges.S1["Qout"][i_ed]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S1["alkaline_solid_added"][i_ed]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S1["alkaline_to_acid"][i_ed]
            OAE_outputs["S_t"][i] = 1

            # Update Tank Volumes
            tank_vol_b[i + 1] = tank_vol_b[i] + OAE_outputs["volBase"][i]
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]
            tank_vol_a[i + 1] = tank_vol_a[i] + OAE_outputs["volAcid"][i]
            OAE_outputs["tank_vol_a"][i] = tank_vol_a[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max
            
            # Excess Power
            P_oae = ranges.S1["pwrRanges"][
                i_ed
            ]  # power needed for mCC given the available power
            OAE_outputs["P_xs"][i] = (
                power_profile[i] - P_oae
            )  # Remaining power available for batteries

            # Number of times system is on
            nON = nON + 1  # Used to determine Capacity Factor

        # Scenario 2: Capture CO2 and Fill Tanks
        elif (
            oae_config.use_storage_tanks
            and power_profile[i] >= ranges.P_minS2_tot
            and tank_vol_b[i] < ranges.V_bT_max
        ):
            # Find number of units that can be active based on power and volume
            # Determine number of scenarios that meet the qualifications
            v = 0
            for j in range(ranges.S2_tot_range):
                if (
                    power_profile[i] >= ranges.S2["pwrRanges"][j]
                    and ranges.V_bT_max >= tank_vol_b[i] + ranges.S2["volBase"][j]
                ):
                    v = v + 1  # determine size of matrix for qualifying scenarios
            S2_viableRanges = np.zeros((v, 2))
            i_v = 0
            for j in range(ranges.S2_tot_range):
                if (
                    power_profile[i] >= ranges.S2["pwrRanges"][j]
                    and ranges.V_bT_max >= tank_vol_b[i] + ranges.S2["volBase"][j]
                ):
                    S2_viableRanges[i_v, 0] = j  # index in the scenarios
                    S2_viableRanges[i_v, 1] = ranges.S2["volBase"][
                        j
                    ]  # adding volume to the tanks is prioritized
                    i_v = i_v + 1
            # Select the viable scenario that fills the tank the most
            for j in range(len(S2_viableRanges[:, 1])):
                if S2_viableRanges[j, 1] == max(S2_viableRanges[:, 1]):
                    i_s2 = int(S2_viableRanges[j, 0])

            # Number of ED Units Active
            OAE_outputs["N_ed"][i] = (
                ranges.S2_ranges[i_s2, 0] + ranges.S2_ranges[i_s2, 1]
            )  # number of ED units active

            # Update recorded values based on the case within S2
            OAE_outputs["volExcessAcid"][i] = ranges.S2["volExcessAcid"][i_s2]
            OAE_outputs["volBase"][i] = ranges.S2["volBase"][i_s2]
            OAE_outputs["volAcid"][i] = ranges.S2["volAcid"][i_s2]
            OAE_outputs["mol_OH"][i] = ranges.S2["mol_OH"][i_s2]
            OAE_outputs["mass_CO2_absorbed"][i] = ranges.S2["mass_CO2_absorbed"][i_s2]
            OAE_outputs["mol_HCl"][i] = ranges.S2["mol_HCl"][i_s2]
            OAE_outputs["pH_f"][i] = ranges.S2["pH_f"][i_s2]
            OAE_outputs["dic_f"][i] = ranges.S2["dic_f"][i_s2]
            OAE_outputs["ta_f"][i] = ranges.S2["ta_f"][i_s2]
            OAE_outputs["sal_f"][i] = ranges.S2["sal_f"][i_s2]
            OAE_outputs["temp_f"][i] = ranges.S2["temp_f"][i_s2]
            OAE_outputs["ca_f"][i] = ranges.S2["ca_f"][i_s2]
            OAE_outputs["c_a"][i] = ranges.S2["c_a"][i_s2]
            OAE_outputs["c_b"][i] = ranges.S2["c_b"][i_s2]
            OAE_outputs["Qin"][i] = ranges.S2["Qin"][i_s2]
            OAE_outputs["Qout"][i] = ranges.S2["Qout"][i_s2]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S2["alkaline_solid_added"][i_s2]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S2["alkaline_to_acid"][i_s2]
            OAE_outputs["S_t"][i] = 2

            # Update Tank Volume
            tank_vol_b[i + 1] = tank_vol_b[i] + OAE_outputs["volBase"][i]
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]
            tank_vol_a[i + 1] = tank_vol_a[i] + OAE_outputs["volAcid"][i]
            OAE_outputs["tank_vol_a"][i] = tank_vol_a[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max

            # Find excess power
            P_oae = ranges.S2["pwrRanges"][
                i_s2
            ]  # power needed for OAE given the available power
            OAE_outputs["P_xs"][i] = (
                power_profile[i] - P_oae
            )  # Remaining power available for batteries

            # Number of times system is on
            nON = nON + 1  # Used to determine Capacity Factor

        # Scenario 3: Tanks Used for CO2 Capture
        elif (
            oae_config.use_storage_tanks
            and power_profile[i] >= ranges.P_minS3_tot
            and tank_vol_b[i] >= ranges.V_b3_min
        ):
            # Find number of equivalent units active based on power
            for j in range(ranges.N_range):
                if (
                    power_profile[i] >= ranges.S3["pwrRanges"][j]
                    and -tank_vol_b[i] <= ranges.S3["volBase"][j]
                ):
                    i_ed = j  # determine how many ED units can be used
                elif ranges.V_bT_max == 0:
                    i_ed = 0
            OAE_outputs["N_ed"][i] = N_edMin + i_ed  # number of ED units active

            # Update recorded values based on number of ED units active
            OAE_outputs["volBase"][i] = ranges.S3["volBase"][i_ed]
            OAE_outputs["volAcid"][i] = ranges.S3["volAcid"][i_ed]
            OAE_outputs["mol_OH"][i] = ranges.S3["mol_OH"][i_ed]
            OAE_outputs["mass_CO2_absorbed"][i] = ranges.S3["mass_CO2_absorbed"][i_ed]
            OAE_outputs["mol_HCl"][i] = ranges.S3["mol_HCl"][i_ed]
            OAE_outputs["pH_f"][i] = ranges.S3["pH_f"][i_ed]
            OAE_outputs["dic_f"][i] = ranges.S3["dic_f"][i_ed]
            OAE_outputs["ta_f"][i] = ranges.S3["ta_f"][i_ed]
            OAE_outputs["sal_f"][i] = ranges.S3["sal_f"][i_ed]
            OAE_outputs["temp_f"][i] = ranges.S3["temp_f"][i_ed]
            OAE_outputs["ca_f"][i] = ranges.S3["ca_f"][i_ed]
            OAE_outputs["c_a"][i] = ranges.S3["c_a"][i_ed]
            OAE_outputs["c_b"][i] = ranges.S3["c_b"][i_ed]
            OAE_outputs["Qin"][i] = ranges.S3["Qin"][i_ed]
            OAE_outputs["Qout"][i] = ranges.S3["Qout"][i_ed]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S3["alkaline_solid_added"][i_ed]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S3["alkaline_to_acid"][i_ed]
            OAE_outputs["S_t"][i] = 3

            # Update Tank Volume
            tank_vol_b[i + 1] = tank_vol_b[i] + OAE_outputs["volBase"][i]
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]
            tank_vol_a[i + 1] = tank_vol_a[i] + OAE_outputs["volAcid"][i]
            OAE_outputs["tank_vol_a"][i] = tank_vol_a[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max

            # Find excess power
            P_oae = ranges.S3["pwrRanges"][i_ed]  # Power needed for oae
            OAE_outputs["P_xs"][i] = (
                power_profile[i] - P_oae
            )  # Excess Power to Batteries

            # Number of times system is on
            nON = nON + 1  # Used to determine Capacity Factor

        # Scenario 4: No Capture, Tanks Filled by ED Units
        elif (
            oae_config.use_storage_tanks
            and power_profile[i] >= ranges.P_minS4_tot
            and tank_vol_b[i] < ranges.V_b3_min
        ):
            # Determine number of ED units active
            for j in range(ranges.N_range):
                if (
                    power_profile[i] >= ranges.S4["pwrRanges"][j]
                    and ranges.V_bT_max >= tank_vol_b[i] + ranges.S4["volBase"][j]
                ):
                    i_ed = j  # determine how many ED units can be used
                elif ranges.V_bT_max == 0:
                    i_ed = 0
            OAE_outputs["N_ed"][i] = N_edMin + i_ed  # number of ED units active

            # Update recorded values based on number of ED units active
            OAE_outputs["volExcessAcid"][i] = ranges.S4["volExcessAcid"][i_ed]
            OAE_outputs["volBase"][i] = ranges.S4["volBase"][i_ed]
            OAE_outputs["volAcid"][i] = ranges.S4["volAcid"][i_ed]
            OAE_outputs["mol_OH"][i] = ranges.S4["mol_OH"][i_ed]
            OAE_outputs["mass_CO2_absorbed"][i] = ranges.S4["mass_CO2_absorbed"][i_ed]
            OAE_outputs["mol_HCl"][i] = ranges.S4["mol_HCl"][i_ed]
            OAE_outputs["pH_f"][i] = ranges.S4["pH_f"][i_ed]
            OAE_outputs["dic_f"][i] = ranges.S4["dic_f"][i_ed]
            OAE_outputs["ta_f"][i] = ranges.S4["ta_f"][i_ed]
            OAE_outputs["sal_f"][i] = ranges.S4["sal_f"][i_ed]
            OAE_outputs["temp_f"][i] = ranges.S4["temp_f"][i_ed]
            OAE_outputs["ca_f"][i] = ranges.S4["ca_f"][i_ed]
            OAE_outputs["c_a"][i] = ranges.S4["c_a"][i_ed]
            OAE_outputs["c_b"][i] = ranges.S4["c_b"][i_ed]
            OAE_outputs["Qin"][i] = ranges.S4["Qin"][i_ed]
            OAE_outputs["Qout"][i] = ranges.S4["Qout"][i_ed]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S4["alkaline_solid_added"][i_ed]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S4["alkaline_to_acid"][i_ed]
            OAE_outputs["S_t"][i] = 4

            # Update Tank Volume
            tank_vol_b[i + 1] = tank_vol_b[i] + OAE_outputs["volBase"][i]
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]
            tank_vol_a[i + 1] = tank_vol_a[i] + OAE_outputs["volAcid"][i]
            OAE_outputs["tank_vol_a"][i] = tank_vol_a[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max

            # Find excess power
            P_oae = ranges.S4["pwrRanges"][
                i_ed
            ]  # power needed for oae system given the available power
            OAE_outputs["P_xs"][i] = (
                power_profile[i] - P_oae
            )  # Remaining power available for batteries

            # No change to nON since no capture is done

        # Scenario 5: When all Input Power is Excess
        else:
            # Determine number of ED units active
            OAE_outputs["N_ed"][i] = 0  # None are used in this case

            # Update recorded values based on number of ED units active
            OAE_outputs["volExcessAcid"][i] = ranges.S5["volExcessAcid"]
            OAE_outputs["volBase"][i] = ranges.S5["volBase"]
            OAE_outputs["volAcid"][i] = ranges.S5["volAcid"].item()
            OAE_outputs["mol_OH"][i] = ranges.S5["mol_OH"]
            OAE_outputs["mass_CO2_absorbed"][i] = ranges.S5["mass_CO2_absorbed"]
            OAE_outputs["mol_HCl"][i] = ranges.S5["mol_HCl"]
            OAE_outputs["pH_f"][i] = ranges.S5["pH_f"]
            OAE_outputs["dic_f"][i] = ranges.S5["dic_f"]
            OAE_outputs["ta_f"][i] = ranges.S5["ta_f"]
            OAE_outputs["sal_f"][i] = ranges.S5["sal_f"]
            OAE_outputs["temp_f"][i] = ranges.S5["temp_f"]
            OAE_outputs["ca_f"][i] = ranges.S5["ca_f"]
            OAE_outputs["c_a"][i] = ranges.S5["c_a"]
            OAE_outputs["c_b"][i] = ranges.S5["c_b"]
            OAE_outputs["Qin"][i] = ranges.S5["Qin"]
            OAE_outputs["Qout"][i] = ranges.S5["Qout"]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S5["alkaline_solid_added"]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S5["alkaline_to_acid"]
            OAE_outputs["S_t"][i] = 5

            # Update Tank Volume
            tank_vol_b[i + 1] = tank_vol_b[i] + OAE_outputs["volBase"][i]
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]
            tank_vol_a[i + 1] = tank_vol_a[i] + OAE_outputs["volAcid"][i]
            OAE_outputs["tank_vol_a"][i] = tank_vol_a[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = ranges.V_bT_max
            if tank_vol_a[i + 1] > ranges.V_aT_max:
                tank_vol_a[i + 1] = ranges.V_aT_max

            # Find excess power
            OAE_outputs["P_xs"][i] = power_profile[
                i
            ]  # Otherwise the input power goes directly to the batteries

            # No change to nON since no capture is done

    # Overall tank fill
    maxTankFill_m3 = max(tank_vol_b)

    if ranges.V_bT_max == 0:
        maxTankFillP = 0
    else:
        maxTankFillP = (
            max(tank_vol_b) / ranges.V_bT_max * 100
        )  # max tank fill in percent

    # Totals
    # Total moles of alkalinity added
    mol_OH_total = sum(OAE_outputs["mol_OH"])

    # Total moles of excess acid generated
    mol_HCl_total = sum(OAE_outputs["mol_HCl"])

    # Range of Base Addition Rates (mol)
    OH_min_addition_mol = min(ranges.S1["mol_OH"])
    OH_max_addition_mol = max(ranges.S1["mol_OH"])

    # Average yearly alkalinity addition
    mol_OH_yr = mol_OH_total # mol/yr

    # Approximation for CDR Scale
    N_co2est = oae_config.assumed_CDR_rate * mol_OH_yr # (mol CO2/yr) Estimated moles of CO2 absorbed
    M_co2est = N_co2est * 44 /1000000 # (tCO2/yr) Estimated mass of CO2 absorbed

    # Approximation for MAX CDR capacity
    # Yearly alkalinity addition under constant max power conditions
    mol_OH_yr_MaxPwr = OH_max_addition_mol * 8760  # mol/yr
    N_co2cap = oae_config.assumed_CDR_rate * mol_OH_yr_MaxPwr # (mol CO2/yr) Estimated moles of maximum possible CO2 absorbed
    M_co2cap = N_co2cap * 44 /1000000 # (tCO2/yr) Estimated mass of maximum possible CO2 absorbed

    # Average pH, DIC, and sal of effluent when OAE is done
    pH_oae = [p for i, p in enumerate(OAE_outputs["pH_f"]) if p != seawater_config.pH_i]
    dic_oae = [OAE_outputs["dic_f"][i] for i, p in enumerate(OAE_outputs["pH_f"]) if p != seawater_config.pH_i]
    sal_oae = [OAE_outputs["sal_f"][i] for i, p in enumerate(OAE_outputs["pH_f"]) if p != seawater_config.pH_i]
    ta_oae = [OAE_outputs["ta_f"][i] for i, p in enumerate(OAE_outputs["pH_f"]) if p != seawater_config.pH_i]
    tempC_oae = [OAE_outputs["temp_f"][i] for i, p in enumerate(OAE_outputs["pH_f"]) if p != seawater_config.pH_i]
    ca_oae = [OAE_outputs["ca_f"][i] for i, p in enumerate(OAE_outputs["pH_f"]) if p != seawater_config.pH_i]

    pH_avg = sum(pH_oae) / len(pH_oae)
    dic_avg = sum(dic_oae) / len(dic_oae)
    sal_avg = sum(sal_oae) / len(sal_oae)
    ta_avg = sum(ta_oae) / len(ta_oae)
    tempC_avg = sum(tempC_oae) / len(tempC_oae)
    ca_avg = sum(ca_oae) / len(ca_oae)
    
    # pH of excess acid
    pH_HCl_excess = -math.log10(ranges.S1["c_a"][0])

    # Average yearly acid production
    mol_HCl_yr = sum(OAE_outputs["mol_HCl"][0:8760])

    # Average volume of acid to dispose of
    volXSacid_yr = sum(OAE_outputs["volExcessAcid"][0:8760])

    # Average mass of alkaline solids needed for neutralization
    m_adSolid_yr = sum(OAE_outputs["alkaline_solid_added"][0:8760]) 

    # Average mass of sold products and value per ton
    X_disp = 49.25 # $/ton of disposing hazardous waste in CA (Jin 2025)
    if oae_config.acid_disposal_method == "sell acid":
        M_rev_yr = volXSacid_yr * R_H2O/10**3 # mass of sold acid in tons/yr 
        X_rev = 9 # $/ton for dilute acid
        M_disposed_yr = 0 # No disposal of alkaline solids
        slurry_mass_max = 0

    elif oae_config.acid_disposal_method == "sell rca":
        W_rcaI_yr = m_adSolid_yr/10**6 # (t/yr) Mass of RCA used for neutralization
        W_caoDT_yr = rca.frac_cao*rca.frac_dissolved*W_rcaI_yr # (t/yr) Mass of CaO that dissolved to neutralize acid
        W_rcaF_yr = (W_rcaI_yr - W_caoDT_yr)*rca.frac_sellable_rca # (t/yr) mass of sellable RCAs
        M_rev_yr = W_rcaF_yr # (t/yr) this is what is sellable as a product
        X_rev = 40 # $/ton for RCAs
        M_disposed_yr = 0 # No need to dispose of hazardous waste
        # Maximum mass for RCA system to handle (mass of slurry)
        slurry_mass_max = max(ranges.S1["rca_power"]) / rca.Wpg_rca
    else:
        # Assume acid is sold
        M_rev_yr = 0 # mass of sold acid in tons/yr 
        X_rev = 9 # $/ton for dilute acid
        M_disposed_yr = volXSacid_yr * R_H2O/10**3 # mass of sold/disposed acid in tons/yr
        slurry_mass_max = 0

    # Average volume of alkaline seawater added to ocean
    volOAEbase_yr = sum(OAE_outputs["Qout"][0:8760]*3600) # (m3/yr) Volume of alkaline seawater added to ocean

    # Overall capacity factor (times system is on)
    OAE_timeFrac = nON/len(OAE_outputs["N_ed"])

    # OAE capacity factor (compare OAE with max if max power always available)
    OAEcapFact = mol_OH_yr/mol_OH_yr_MaxPwr

    # Print and determine the energy capacity factor (compare energy availability with max if max power always available)
    EcapFact = sum(power_profile[0:8760]) / (power_capacity*8760) 

    return OAEOutputs(
        OAE_outputs=OAE_outputs,
        mol_OH_yr=mol_OH_yr,
        pH_avg=pH_avg,
        dic_avg=dic_avg,
        ta_avg=ta_avg,
        sal_avg=sal_avg,
        tempC_avg=tempC_avg,
        ca_avg=ca_avg,
        volOAEbase_yr=volOAEbase_yr,
        mol_OH_yr_MaxPwr=mol_OH_yr_MaxPwr,
        mol_HCl_yr=mol_HCl_yr,
        volXSacid_yr=volXSacid_yr,
        pH_HCl_excess=pH_HCl_excess,
        m_adSolid_yr=m_adSolid_yr,
        slurry_mass_max=slurry_mass_max,
        M_rev_yr=M_rev_yr,
        M_disposed_yr=M_disposed_yr,
        X_disp=X_disp,
        X_rev_yr=X_rev,
        M_co2est=M_co2est,
        M_co2cap=M_co2cap,
        max_tank_fill_percent=maxTankFillP,
        max_tank_fill_m3=maxTankFill_m3,
        overall_capacity_factor=OAE_timeFrac,
        oae_capacity_factor=OAEcapFact,
        energy_capacity_factor=EcapFact,
    )

@define
class BioGeoChemOutputs:
    """
    This class contains the daily outputs of the biogeochemical processes
    involved in ocean alkalinity enhancement, including the volume of seawater treated,
    flow rates, and chemical properties of the treated seawater.
    Attributes:
        days (np.ndarray): Days of the year (1-365).
        volume_seawater_out (np.ndarray): Volume of seawater treated and released (m³).
        flow_rate_seawater_out (np.ndarray): Flow rate of seawater treated and released (m³/s).
        pH_avg_out (np.ndarray): Average pH of treated seawater.
        dic_out (np.ndarray): Dissolved inorganic carbon concentration in treated seawater (mol/L).
        sal_out (np.ndarray): Salinity of treated seawater (ppt).
        ta_out (np.ndarray): Total alkalinity of treated seawater (mol/L).
        temp_out (np.ndarray): Temperature of treated seawater (C).
        ca_out (np.ndarray): Calcium concentration in treated seawater (mol/L).
    """
    days: np.ndarray  # Days of the year (1-365)
    volume_seawater_out: np.ndarray  # Volume of seawater treated and released (m³)
    flow_rate_seawater_out: np.ndarray  # Flow rate of seawater treated and released (m³/s)
    pH_avg_out: np.ndarray  # Average pH of treated seawater
    dic_out: np.ndarray  # Dissolved inorganic carbon concentration in treated seawater (mol/L)
    sal_out: np.ndarray  # Salinity of treated seawater (ppt)
    ta_out: np.ndarray  # Total alkalinity of treated seawater (mol/L)
    temp_out: np.ndarray  # Temperature of treated seawater (C)
    ca_out: np.ndarray # Calcium concentration in treated seawater (mol/L)

def run_ocean_alkalinity_enhancement_physics_model(
    power_profile_w,
    power_capacity_w,
    initial_tank_volume_m3,
    oae_config: OAEInputs,
    pump_config: PumpInputs,
    seawater_config: SeaWaterInputs,
    rca: RCALoadingCalculator,
    save_plots=False,
    show_plots=False,
    plot_range=[0, 144],
    save_outputs=False,
    output_dir="./output/",
) -> Tuple[
    OAERangeOutputs, OAEOutputs
]:
    """
    Runs the OAE physics model to simulate CO2 capture and OAE operations based on the given configurations and power profile.

    Args:
        power_profile_w (np.ndarray): Power profile (in watts) for the simulation over the specified time period.
        initial_tank_volume_m3 (float): Initial volume of acid and base tanks in cubic meters.
        oae_config (OAEInputs): Configuration parameters for the OAE process, including power, flow rates, and efficiency.
        pump_config (PumpInputs): Configuration parameters for the pump system, including power, flow rates, and efficiencies.
        seawater_config (SeaWaterInputs): Seawater properties such as temperature and salinity to be used in the OAE process.
        save_plots (bool, optional): If True, plots of the results will be saved to the output directory. Defaults to False.
        show_plots (bool, optional): If True, plots will be displayed during the simulation. Defaults to False.
        plot_range (list, optional): Range of time steps (in hours) to plot results for. Defaults to [0, 144].
        save_outputs (bool, optional): If True, the simulation results will be saved as CSV files in the output directory. Defaults to False.
        output_dir (str, optional): Directory to save output files and plots. Defaults to "./output/".

    Returns:
        Tuple[OAERangeOutputs, OAERangeOutputs]:
            - `OAERangeOutputs`: Power and chemical ranges for the different scenarios simulated.
            - `OAEOutputs`: Simulation results including time-dependent CO2 capture, power usage, and acid/base production.
    """

    ranges = initialize_power_chemical_ranges(
        oae_config=oae_config,
        pump_config=pump_config,
        seawater_config=seawater_config,
        rca=rca
    )
    res = simulate_ocean_alkalinity_enhancement(
        ranges=ranges,
        oae_config=oae_config,
        seawater_config=seawater_config,
        rca=rca,
        power_profile=power_profile_w,
        power_capacity= power_capacity_w,
        initial_tank_volume_m3=initial_tank_volume_m3,
    )

    # Results for biogeochemistry (daily results)
    iDays = np.zeros(365)
    VswOut = np.zeros(365)
    QswOut = np.zeros(365)
    pHavgOut = np.zeros(365) # needs to only consider hours when q_out > 0
    dicOut = np.zeros(365) 
    salOut = np.zeros(365)
    taOut = np.zeros(365)
    tempOut = np.zeros(365)
    caOut = np.zeros(365)
    for i in range(365):
        iDays[i] = i+1
        VswOut[i] = 3600*sum(res.OAE_outputs["Qout"][24*i:24*(i+1)-1])
        QswOut[i] = VswOut[i]/(24*60*60) # average output flowrate in m3/s
        pHon = 0
        pHsum = 0
        dicSum = 0
        salSum = 0
        taSum = 0
        tempSum = 0
        caSum = 0
        for j in range(24*i, 24*(i+1)):
            if res.OAE_outputs["Qout"][j] > 0:
                pHon = pHon+1
                pHsum = pHsum + res.OAE_outputs["pH_f"][j]
                dicSum = dicSum + res.OAE_outputs["dic_f"][j]
                salSum = salSum + res.OAE_outputs["sal_f"][j]
                taSum = taSum + res.OAE_outputs["ta_f"][j]
                tempSum = tempSum + res.OAE_outputs["temp_f"][j]
                caSum = caSum + res.OAE_outputs["ca_f"][j]
        if pHon > 0:
            pHavgOut[i] = round(pHsum/pHon,2) # average effluent pH when plant is active at least once in a day
            dicOut[i] = dicSum/pHon # average DIC when plant can be active at least once in a day
            salOut[i] = round(salSum/pHon,2)
            taOut[i] = taSum/pHon # average TA when plant can be active at least once in a day
            tempOut[i] = round(tempSum/pHon,3)
            caOut[i] = caSum/pHon
        else:
            pHavgOut[i] = seawater_config.pH_i
            dicOut[i] = seawater_config.dic_i
            salOut[i] = seawater_config.sal_ppt_i
            taOut[i] = seawater_config.ta_i
            tempOut[i] = seawater_config.tempC
            caOut[i] = seawater_config.ca_i

    if save_plots or save_outputs:
        save_paths = [output_dir + "figures/", output_dir + "data/"]

        for savepath in save_paths:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

    if save_outputs:
            # Design Inputs
            design_inputs = {
                "Maximum Power Need for ED System (W)": round(oae_config.P_edMax, 2),
                "Maximum Flow Rate for ED System (m3/s)": oae_config.Q_edMax,
                "Percentage of ED Flow that Becomes Base (%)": round(oae_config.frac_baseFlow*100,2),
                "Concentration of Acid Made by ED (M)": oae_config.c_a,
                "Concentration of Base Made by ED (M)": oae_config.c_b,
                "Minimum Number of ED Units Used": oae_config.N_edMin,
                "Maximum Number of ED Units Used": oae_config.N_edMax,
                "Acid Production Efficiency (Wh/mol HCl)": round(oae_config.E_HCl*1000, 2),
                "Base Production Efficiency (Wh/mol NaOH)": round(oae_config.E_NaOH*1000, 2),
                "Method of Acid Disposal": oae_config.acid_disposal_method,
                "Average Seawater Temperature (C)": seawater_config.tempC_i,
                "Average Seawater Salinity (ppt)": round(seawater_config.sal_ppt_i,2),
                "Initial Seawater pH": seawater_config.pH_i,
                "Initial Seawater DIC (M)": seawater_config.dic_i,
            }
            diDF = pd.DataFrame(design_inputs, index=[0]).T
            diDF = diDF.reset_index()
            diDF.columns = ["Design Input", "Values"]
            diDF.to_csv(save_paths[1] + "OAE_resultTotals.csv", index=False)

            # Time Dependent Inputs and Results
            timeDepDict = {
                "Input Power (W)": power_profile_w,
                "Scenario": res.OAE_outputs["S_t"],
                "ED Units Active": res.OAE_outputs["N_ed"],
                "Excess Power (W)": res.OAE_outputs["P_xs"],
                "Concentration of Acid Made (mol/L)": res.OAE_outputs["c_a"],
                "Concentration of Base Made (mol/L)": res.OAE_outputs["c_b"],
                "Moles of Base Added to Seawater (mol)": res.OAE_outputs["mol_OH"],
                "Moles of Excess Acid Generated (mol)": res.OAE_outputs["mol_HCl"],
                "Mass of CO2 Absorbed (kg)": res.OAE_outputs["mass_CO2_absorbed"],
                "Volume of Excess Acid (m3)": res.OAE_outputs["volExcessAcid"],
                "Base Tank Volume (m3)": res.OAE_outputs["tank_vol_b"],
                "Base Added Volume (m3)": res.OAE_outputs["volBase"],
                "Acid Tank Volume (m3)": res.OAE_outputs["tank_vol_a"],
                "Acid Added Volume (m3)": res.OAE_outputs["volAcid"],
                "Seawater Flow Rate Into Plant (m3/s)": res.OAE_outputs["Qin"],
                "Seawater Flow Rate Out of Plant (m3/s)": res.OAE_outputs["Qout"],
                "pH of Effluent Seawater": res.OAE_outputs["pH_f"],
                "DIC of Effluent Seawater (mol/L)": res.OAE_outputs["dic_f"],
                "TA of Effluent Seawater (mol/L)": res.OAE_outputs["ta_f"],
                "Salinity of Effluent Seawater (ppt)": res.OAE_outputs["sal_f"],
                "Temperature of Effluent Seawater (C)": res.OAE_outputs["temp_f"],
                "Calcium of Effluent Seawater (mol/L)": res.OAE_outputs["ca_f"],
                "Alkaline Solid Added (g)": res.OAE_outputs["alkaline_solid_added"],

            }
            timeDepDF = pd.DataFrame(timeDepDict)
            timeDepDF.to_csv(
                save_paths[1] + "OAE_timeDependentResults.csv", mode="a", index=False
            )

            # Scenario Ranges for Simulations
            # Define scenarios and related ranges
            scenarios = [
                (
                    "S1: Base Added to Seawater, Tanks Not Filled, ED On",
                    ranges.S1["pwrRanges"],
                    oae_config.N_edMin,
                    0,
                ),
                (
                    "S2: Base Added to Seawater, Tanks Filled, ED On",
                    ranges.S2["pwrRanges"],
                    ranges.S2_ranges[:, 0],
                    ranges.S2_ranges[:, 1],
                ),
                (
                    "S3: Base Added to Seawater, Tanks Emptied, ED Off",
                    ranges.S3["pwrRanges"],
                    oae_config.N_edMin,
                    0,
                ),
                (
                    "S4: No Base Added to Seawater, Tanks Filled, ED On",
                    ranges.S4["pwrRanges"],
                    0,
                    oae_config.N_edMin,
                ),
            ]

            # Generate scenario names
            scenNames = [
                name for name, pwrRange, *_ in scenarios for _ in range(len(pwrRange))
            ]

            # Number of ED units (or equivalent) used for OAE
            scenEDoae = np.zeros(len(ranges.S1["pwrRanges"])+len(ranges.S2["pwrRanges"])+len(ranges.S3["pwrRanges"])+len(ranges.S4["pwrRanges"]))
            edo = 0 # ED units used for OAE counter
            for i in range(len(ranges.S1["pwrRanges"])):
                scenEDoae[edo] = oae_config.N_edMin + i
                edo = edo + 1
            for i in range(len(ranges.S2["pwrRanges"])):
                scenEDoae[edo] = ranges.S2_ranges[i,0]
                edo = edo + 1
            for i in range(len(ranges.S3["pwrRanges"])):
                scenEDoae[edo] = oae_config.N_edMin + i
                edo = edo + 1
            for i in range(len(ranges.S4["pwrRanges"])):
                scenEDoae[edo] = 0
                edo = edo +1
            # Number of ED units used to fill tanks
            scenEDtank = np.zeros(len(ranges.S1["pwrRanges"])+len(ranges.S2["pwrRanges"])+len(ranges.S3["pwrRanges"])+len(ranges.S4["pwrRanges"]))
            edt = 0 # ED units used for filling tanks counter
            for i in range(len(ranges.S1["pwrRanges"])):
                scenEDtank[edt] = 0
                edt = edt + 1
            for i in range(len(ranges.S2["pwrRanges"])):
                scenEDtank[edt] = ranges.S2_ranges[i,1]
                edt = edt + 1
            for i in range(len(ranges.S3["pwrRanges"])):
                scenEDtank[edt] = 0
                edt = edt + 1
            for i in range(len(ranges.S4["pwrRanges"])):
                scenEDtank[edt] = oae_config.N_edMin + i
                edt = edt + 1

            # Power, mCC, acid, and base values
            scenPwr = np.concatenate([pwrRange for _, pwrRange, *_ in scenarios])
            scenNB = np.concatenate(
                [getattr(ranges, key)["mol_OH"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenNA = np.concatenate(
                [getattr(ranges, key)["mol_HCl"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenCDR = np.concatenate(
                [getattr(ranges, key)["mass_CO2_absorbed"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenVolExcessAcid = np.concatenate(
                [getattr(ranges, key)["volExcessAcid"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenVolBase = np.concatenate(
                [getattr(ranges, key)["volBase"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenVolAcid = np.concatenate(
                [getattr(ranges, key)["volAcid"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenMadSolid = np.concatenate(
                [getattr(ranges, key)["alkaline_solid_added"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenPH = np.concatenate(
                [getattr(ranges, key)["pH_f"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenDIC = np.concatenate(
                [getattr(ranges, key)["dic_f"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenTA = np.concatenate(
                [getattr(ranges, key)["ta_f"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenSAL = np.concatenate(
                [getattr(ranges, key)["sal_f"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenQout = np.concatenate(
                [getattr(ranges, key)["Qout"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenAlk2Acid = np.concatenate(
                [getattr(ranges, key)["alkaline_to_acid"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenRCApower = np.concatenate(
                [getattr(ranges, key)["rca_power"] for key in ["S1", "S2", "S3", "S4"]]
            )

            # Create dictionary and save CSV
            scenDict = {
                "Scenario": scenNames,
                "ED Units Used for OAE (or Equivalent for S3)": scenEDoae,
                "ED Units Used to Fill Tanks": scenEDtank,
                "Power Needed (W)": scenPwr,
                "Rate of Base Added to Seawater (molOH/hr)":scenNB,
                "Rate of CO2 Absorbed (kgCO2/hr)": scenCDR,
                "Rate of Excess Acid Generated (molHCl/hr)":scenNA,
                "Rate of Excess Acid Produced (m3/hr)":scenVolExcessAcid,
                "Volume of Base Added to Tanks (m3)":scenVolBase,
                "Volume of Acid Added to Tanks (m3)":scenVolAcid, 
                "Mass of Alkaline Solid Used of Acid Disposal (g)":scenMadSolid,
                "Ratio of Alkaline Solid to Acid for Neutralization (g/L)":scenAlk2Acid, 
                "RCA Power (W)": scenRCApower,
                "Effluent pH":scenPH, 
                "Effluent DIC (M)":scenDIC, 
                "Effluent TA (M)":scenTA, 
                "Effluent Salinity (ppt)":scenSAL,
                "Effluent Volume (m3)": scenQout*3600
            }

            scenDF = pd.DataFrame(scenDict)
            scenDF.to_csv(
                save_paths[1] + "OAE_operationScenarios.csv", index=False
            )

            # Totals for Simulations
            total_results = {
                "Average Moles of Base Added to Seawater (molOH/yr)": res.mol_OH_yr,
                "Average pH of Effluent": round(res.pH_avg,2),
                "Average DIC of Effluent (M)": res.dic_avg, 
                "Average TA of Effluent (M)": res.ta_avg,
                "Average Temperature of Effluent (C)": round(res.tempC_avg,2),
                "Average Salinity of Effluent (ppt)": round(res.sal_avg,2),
                "Average Volume of Effluent (m3/yr)": round(res.volOAEbase_yr,2),
                "Min Total Power Need for OAE (W)": round(min(ranges.S3["pwrRanges"]),2),
                "Max Total Power Need for OAE (W)": round(max(ranges.S1["pwrRanges"]),2),
                "Min OAE Rate (molOH/hr)": round(min(ranges.S1["mol_OH"]),2),
                "Max OAE Rate (molOH/hr)": round(max(ranges.S1["mol_OH"]),2),
                "Base Added to Seawater Under 100% Max Power (molOH/yr)": res.mol_OH_yr_MaxPwr,
                "OAE Capacity Factor (%)": round(res.oae_capacity_factor,2),
                "Fraction of Time OAE is Performed (%)": round(res.overall_capacity_factor,2),
                "Max Tank Fill (m3)": round(max(res.OAE_outputs["tank_vol_b"]),2),
                "Max Tank Fill (%)": round(res.max_tank_fill_percent),
                "Min ED Power (W)": round(oae_config.P_ed1
                * oae_config.N_edMin,2),
                "Max ED Power (W)": round(oae_config.P_ed1
                * oae_config.N_edMax,2),
                "Min Pump Power (W)": round(ranges.pump_power_min*10e6,3),
                "Max Pump Power (W)": round(ranges.pump_power_max*10e6,3),
                "Min Intake Pump Flow Rate (m3/s)": round(ranges.pumps.pumpO.Q_min,2),
                "Max Intake Pump Flow Rate (m3/s)": round(ranges.pumps.pumpO.Q_max,2),
                "Average Moles of Excess Acid Generated (molHCl/yr)": round(res.mol_HCl_yr,2),
                "Average Volume of Excess Acid Produced (m3/yr)": round(res.volXSacid_yr,2),
                "pH of Excess Acid": round(res.pH_HCl_excess,2),
                "Average Mass of Alkaline Solid Used for Acid Disposal (g/yr)": round(res.m_adSolid_yr,2),
                "Ratio of Alkaline Solid to Acid for Neutralization (g/L)": round(ranges.S1["alkaline_to_acid"][0],5),
                "Mass of Products Made (tonnes/yr)": res.M_rev_yr,
                "Value of Products Made ($/yr)": res.X_rev_yr,
                "Mass of Acid Disposed of (tonnes/yr)": res.M_disposed_yr,
                "Disposal Cost of Acid ($/t)": res.X_disp,
                "Estimated CDR (0.8 mol CO2:NaOH) (tCO2/yr)": res.M_co2est,
                "Estimated Max CDR Scale (0.8 mol CO2:NaOH) (tCO2/yr)": res.M_co2cap,
                "Maximum Mass of RCA Tumbler Slurry (g)": res.slurry_mass_max,
            }
            totsDF = pd.DataFrame(total_results, index=[0]).T
            totsDF = totsDF.reset_index()
            totsDF.columns = ["Parameter", "Values"]
            totsDF.to_csv(save_paths[1] + "OAE_resultTotals.csv", mode="a", index=False)

            biogeochem_results = {
                "Day": iDays,
                "Average Flow Rate of Alaline Seawater (m3/s)": QswOut,
                "pH of Alkaline Seawater": pHavgOut,
                "DIC of Alkaline Seawater (mol/m3)": dicOut*10**6,
                "TA of Alkaline Seawater (mol/m3)": taOut*10**6,
                "Salinity of Alkaline Seawater (ppt)": salOut,
                "Temperature of Alkaline Seawater (C)": tempOut,
                "Calcium of Alkaline Seawater (mol/m3)": caOut*10**6,
            }
            biogeochemDF = pd.DataFrame(biogeochem_results)
            biogeochemDF.to_csv(
                save_paths[1] + "OAE_biogeochemResults.csv", index=False
            )

            biogeochem_hourly = {
                "Hour": np.arange(0, len(res.OAE_outputs["Qout"])),
                "Flow Rate of Alkaline Seawater (m3/s)": res.OAE_outputs["Qout"],
                "pH of Alkaline Seawater": res.OAE_outputs["pH_f"],
                "DIC of Alkaline Seawater (mol/m3)": res.OAE_outputs["dic_f"]*10**6,
                "TA of Alkaline Seawater (mol/m3)": res.OAE_outputs["ta_f"]*10**6,
                "Calcium of Alkaline Seawater (mol/m3)": res.OAE_outputs["ca_f"]*10**6,
                "Salinity of Alkaline Seawater (ppt)": res.OAE_outputs["sal_f"],
                "Temperature of Alkaline Seawater (C)": res.OAE_outputs["temp_f"],
            }

            biogeochem_hourlyDF = pd.DataFrame(biogeochem_hourly)
            biogeochem_hourlyDF.to_csv(
                save_paths[1] + "OAE_biogeochem_hourly_results.csv", index=False
            )

    if save_plots or show_plots:
        # Create time as a NumPy array for easy indexing
        time = np.arange(plot_range[0], plot_range[1])

        # Make Threshold Lines for S2, S3, & S4
        lowThresS2 = min(ranges.S2["pwrRanges"]) / 10**6 * np.ones(len(time))
        hiThresS2 = max(ranges.S2["pwrRanges"]) / 10**6 * np.ones(len(time))
        lowThresS3 = min(ranges.S3["pwrRanges"]) / 10**6 * np.ones(len(time))
        hiThresS3 = max(ranges.S3["pwrRanges"]) / 10**6 * np.ones(len(time))
        lowThresS4 = min(ranges.S4["pwrRanges"]) / 10**6 * np.ones(len(time))
        hiThresS4 = max(ranges.S4["pwrRanges"]) / 10**6 * np.ones(len(time))

        # Time Dependent Plot
        labelsize = 22

        # Create the first plot
        fig, ax1 = plt.subplots(figsize=(19, 10))

        # Plot on the primary y-axis
        ax1.plot(
            time, power_profile_w[time] / 10**6, label="Input Power", linewidth=2.5
        )
        ax1.plot(
            time,
            res.OAE_outputs["P_xs"][time] / 10**6,
            label="Excess Power",
            linewidth=2.5,
        )
        
        ax1.plot(time, res.OAE_outputs["Qout"][time], label="Rate of OAE", linewidth=2.5)
        ax1.plot(time, lowThresS2, linestyle="--", label="S1/S2 Min Pwr", linewidth=2)
        ax1.plot(time, hiThresS2, linestyle="--", label="S1/S2 Max Pwr", linewidth=2)
        ax1.plot(time, lowThresS3, linestyle="--", label="S3 Min Pwr", linewidth=2)
        ax1.plot(time, hiThresS3, linestyle="--", label="S3 Max Pwr", linewidth=2)
        ax1.plot(time, lowThresS4, linestyle="--", label="S4 Min Pwr", linewidth=2)
        ax1.plot(time, hiThresS4, linestyle="--", label="S4 Max Pwr", linewidth=2)
        if oae_config.acid_disposal_method != "sell acid" or "acid disposal":
            ax1.plot(time, res.OAE_outputs["alkaline_solid_added"][time]/10**6, "brown", linewidth=2.5)
            ax1.set_ylabel(f"Power (MW) & Effluent Alkaline \n Seawater (pH {res.pH_avg:.2f}) (m³/s) & Mass of \n {oae_config.acid_disposal_method} for AD (t/hr)", fontsize=labelsize)
        else:
            ax1.set_ylabel(f"Power (MW) & \n Effluent Alkaline Seawater (pH {res.pH_avg:.2f}) (m³/s)", fontsize=labelsize)
        ax1.tick_params(axis="x", labelsize=labelsize - 2)
        ax1.tick_params(axis="y", labelsize=labelsize - 2)
        ax1.set_title("OAE Plant Model Time-Dependent Results", fontsize=labelsize + 2)
        ax1.set_xlabel("Hours of Operation", fontsize=labelsize)
        
        ax1.grid(color="k", linestyle="--", linewidth=0.5)
        ax1.set_xlim(plot_range[0], plot_range[1])
        ax2 = ax1.twinx()
        ax2.tick_params(axis="x", labelsize=labelsize - 2)
        ax2.tick_params(axis="y", labelsize=labelsize - 2)
        ax2.plot(
            time,
            res.OAE_outputs["tank_vol_b"][time],
            color="black",
            label="Tank Volume",
            linewidth=2.5,
        )
        ax2.set_ylabel("Tank Volume (m³)", fontsize=labelsize)
        ax1.legend(
            fontsize=16,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
        )
        ax2.legend(
            fontsize=16,
            bbox_to_anchor=(1, -0.15),
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )  # Adjust the plot area so the legend fits

        ax1_ylims = ax1.axes.get_ylim()
        ax1_yratio = ax1_ylims[0] / ax1_ylims[1]

        ax2_ylims = ax2.axes.get_ylim()
        ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
        if ax1_yratio < ax2_yratio:
            ax2.set_ylim(bottom=ax2_ylims[1] * ax1_yratio)
        else:
            ax1.set_ylim(bottom=ax1_ylims[1] * ax2_yratio)
        # Show the plot
        if save_plots:
            plt.savefig(
                save_paths[0] + "OAE_Time-Dependent_Results.png", bbox_inches="tight"
            )
        if show_plots:
            plt.show()

        return (ranges, res)
    
@define
class OAECosts:
    """Computes the costs of an Ocean Alkalinity Enhancement (OAE) system.

    Attributes:
        mass_product (float): Mass of products made (tonnes/yr).
        value_product(float): Value of products ($/tonne).
        waste_mass (float): Mass of waste made (tonnes/yr).
        waste_disposal_cost (float): Cost of waste disposal ($/tonne).
        estimated_cdr (float): Estimated Carbon Dioxide Removal (CDR) in tonnes/yr.
        base_added_seawater_max_power (float): Base added to seawater under 100% max power (molOH/yr).
        acid_disposal_method (float): Method of acid disposal, with options "sell rca", "sell acid" or "acid disposal". Defaults to "sell rca".
        mass_rca (Optional float): Mass of RCA tumbler slurry (g).

        save_path

        ed_system_cost (float): Capital cost of the electrodialysis (ED) system ($).
        bop_cost (float): Balance of plant cost, including pumps, tanks, and filtration systems ($).
        import_tariff (float): Import tariff on foreign equipment ($).
        transportation_cost (float): Transportation cost for equipment and materials ($).
        yard_improvements_cost (float): Cost for civil works and yard improvements ($).
        lab_equipment_cost (float): Capital cost of laboratory equipment ($).
        piping_system_cost (float): Cost of the plant piping system ($).
        land_area_m2 (float): Total land area required for the plant (m²).
        land_cost_per_m2 (float): Land cost per square meter at the plant site ($/m²).

        annual_raw_materials_cost (float): Yearly cost of raw materials such as Na₂SO₄, NaOH, HCl, and demineralized water ($/year).
        annual_labor_cost (float): Yearly cost of plant labor ($/year).
        annual_lab_qc_rd_cost (float): Yearly cost for lab operations, quality control, and R&D ($/year).
        annual_consumables_cost (float): Yearly cost of consumables used in operations ($/year).
        annual_transport_cost (float): Yearly cost of transportation for logistics ($/year).
        annual_misc_cost (float): Yearly miscellaneous costs such as marketing and administrative expenses ($/year).
        annual_royalty_cost (float): Yearly royalty payments ($/year).

        startup_cost (float): One-time startup cost for plant commissioning and integration ($).
        upfront_rd_cost (float): One-time cost for R&D incurred prior to plant commissioning ($).
        upfront_royalty_cost (float): One-time royalty payment made before operations begin ($).

        num_membrane_replacements (int): Total number of ED membrane replacements over the plant lifetime.
        plant_lifetime_yrs (int): Operational lifetime of the plant (years).
        learning_rate (float): Learning rate for cost reduction as a decimal (e.g., 0.37 for 37%).
        inflation_rate (float): Annual inflation rate as a decimal (e.g., 0.015 for 1.5%).
        recovery_period_yrs (int): Financial recovery period for investment returns (years).
        interest_rate (float): Annual interest rate used in financial calculations (decimal).
        corporate_tax_rate (float): Corporate tax rate applied to net profits (decimal).
        salvage_value_percent (float): Percentage of capital costs recovered at the end of the plant lifetime (decimal).
        opportunity_cost_capital (float): Opportunity cost of capital as a decimal (e.g., 0.06 for 6%).
    """
    mass_product: float = field()
    value_product: float = field()
    waste_mass: float = field()
    waste_disposal_cost: float = field()
    estimated_cdr: float = field(validator=gt_zero)
    base_added_seawater_max_power: float = field(validator=gt_zero)
    acid_disposal_method: str = field(default="sell acid", validator=contains(["sell acid", "sell rca", "acid disposal"]))
    mass_rca: Optional[float] = field(default=None)

    # --- Direct Capital Costs ---
    ed_system_cost: float = field(default=1_819_238/1.8, validator=gt_zero)  # $
    bop_cost: float = field(default=1_243_308, validator=gt_zero)        # $
    import_tariff: float = field(default=26_839, validator=gt_zero)      # $
    transportation_cost: float = field(default=146_446, validator=gt_zero)  # $
    yard_improvements_cost: float = field(default=188_735, validator=gt_zero)  # $
    lab_equipment_cost: float = field(default=50_000, validator=gt_zero)     # $
    piping_system_cost: float = field(default=161_492, validator=gt_zero)    # $
    land_area_m2: float = field(default=70**2, validator=gt_zero)            # m²
    land_cost_per_m2: float = field(default=38.0, validator=gt_zero)         # $/m²

    # --- Technical Operating Costs (Annual) ---
    annual_energy_cost: float = field(default=5618.45*221.5, validator=gt_zero)       # $/yr
    annual_raw_materials_cost: float = field(default=139_708, validator=gt_zero)  # $/yr
    annual_labor_cost: float = field(default=228_000, validator=gt_zero)          # $/yr
    annual_lab_qc_rd_cost: float = field(default=10_000, validator=gt_zero)       # $/yr
    annual_consumables_cost: float = field(default=15_000, validator=gt_zero)     # $/yr
    annual_transport_cost: float = field(default=0.0)                             # $/yr
    annual_misc_cost: float = field(default=7_000, validator=gt_zero)             # $/yr
    annual_royalty_cost: float = field(default=0.0)                               # $/yr

    # --- Upfront Capital Cost Adjustments ---
    startup_cost: float = field(default=20_000, validator=gt_zero)               # $
    upfront_rd_cost: float = field(default=15_000, validator=gt_zero)            # $
    upfront_royalty_cost: float = field(default=0.0)                              # $

    # --- Fixed Inputs ---
    num_membrane_replacements: int = field(default=6, validator=gt_zero)
    plant_lifetime_yrs: int = field(default=20, validator=gt_zero)
    learning_rate: int = field(default=0.37, validator=range_val(0.0, 1.0))
    inflation_rate: float = field(default=0.015, validator=range_val(0.0, 1.0))
    recovery_period_yrs: int = field(default=10, validator=gt_zero)
    interest_rate: float = field(default=0.0425, validator=range_val(0.0, 1.0))
    corporate_tax_rate: float = field(default=0.2984, validator=range_val(0.0, 1.0))
    salvage_value_percent: float = field(default=0, validator=range_val(0.0, 1.0))
    opportunity_cost_capital: float = field(default=0.06, validator=range_val(0.0, 1.0))

    b: float = field(init=False)

    def __attrs_post_init__(self):
        # Constants from Ferella 2025
        cf_lit = 330 / 365
        N_naoh100 = self.base_added_seawater_max_power
        N_naohEq = cf_lit * N_naoh100
        N_naohLit = 24_948_000  # mol/year
        self.b = -1 * math.log2(1 - self.learning_rate)
        f_adj = (N_naohEq / N_naohLit) ** (1 - self.b)

        # Default values before scaling for comparison
        default_vals = {
            "ed_system_cost": 1_819_238/1.8,
            "bop_cost": 1_243_308,
            "import_tariff": 26_839,
            "transportation_cost": 146_446,
            "yard_improvements_cost": 188_735,
            "lab_equipment_cost": 50_000,
            "piping_system_cost": 161_492,
            "land_area_m2": 70**2,
            "annual_raw_materials_cost": 139_708,
            "annual_labor_cost": 228_000,
            "annual_lab_qc_rd_cost": 10_000,
            "annual_consumables_cost": 15_000,
            "annual_misc_cost": 7_000,
            "startup_cost": 20_000,
            "upfront_rd_cost": 15_000,
        }

        # Scale only fields that are unchanged from defaults
        for field_name, base_val in default_vals.items():
            if getattr(self, field_name) == base_val:
                setattr(self, field_name, base_val * f_adj)

        if self.acid_disposal_method == "sell rca" and not hasattr(self, "mass_rca"):
            if self.mass_rca is None:
                raise ValueError("`mass_rca` must be provided when `acid_disposal_method` is 'sell rca'")

    # Optimization using adaptation of Excel's Goal Seek Function
    def npv_objective(
            self, 
            carbon_credit_value,
            lifetime_annual_operating_cost,
            annual_depreciation,
            annual_interest_payment,
            lifetime_annual_loan_repayment
            ):

        years = self.plant_lifetime_yrs

        # --- Revenue ---
        base_revenue = (
            self.mass_product * self.value_product 
            + self.estimated_cdr * carbon_credit_value
        )
        annual_revenue = base_revenue * (1 + self.inflation_rate) ** np.arange(years)

        # --- EBITDA ---
        ebitda = annual_revenue - lifetime_annual_operating_cost

        # --- Taxable income ---
        taxable_income = ebitda - annual_depreciation - annual_interest_payment

        # --- Tax loss carry forward ---
        tax_loss = np.zeros(years + 1)
        for i in range(years):
            tax_loss[i + 1] = min(tax_loss[i] + taxable_income[i], 0)

        # --- Taxes ---
        taxes = np.maximum(
            self.corporate_tax_rate * (taxable_income + tax_loss[1:]),
            0
        )

        # --- Net cash flow ---
        net_cash_flow = ebitda - lifetime_annual_loan_repayment - taxes

        # --- Discounted cash flow ---
        discount_factors = (1 + self.opportunity_cost_capital) ** np.arange(1, years + 1)
        discounted_cash_flow = net_cash_flow / discount_factors

        # --- Net Present Value ---
        return np.sum(discounted_cash_flow)

    def run(self,
        save_outputs=False,
        output_dir="./output/",
    ):
        """Calculates the costs associated with the Ocean Alkalinity Enhancement (OAE) process.
        This function computes the initial and yearly costs for the OAE system, including capital and operational costs.
        """

        if self.acid_disposal_method == "sell rca":
            # Rotary drum specifications
            ROT_DRUM_CAPACITY = 14_515 * 1_000     # g - Weight capacity of 6DH rotary drum
            ROT_DRUM_COST = 120 * 1_000           # $ - Capital cost of one 6DH rotary drum
            ROT_DRUM_AREA = 30                    # m² - Footprint of the rotary drum
            ROT_DRUM_LABOR_COST = 30 * 1_000      # $/yr - Entry-level pay for a concrete worker

            # Adjustment factor using same learning rate as rest of model
            f_adj_rca = (self.mass_rca / ROT_DRUM_CAPACITY) ** (1 - self.b)

            # --- Estimated RCA Costs ---
            rca_drum_capital_cost = f_adj_rca * ROT_DRUM_COST       # $ - Capital cost
            rca_area_m2 = f_adj_rca * ROT_DRUM_AREA       # m² - Area requirement
            rca_labor_cost = f_adj_rca * ROT_DRUM_LABOR_COST  # $/yr - Labor cost

            # --- Update Overall Costs ---
            self.bop_cost += rca_drum_capital_cost      # Add drum capital cost to BOP capital cost
            self.land_area_m2 += rca_area_m2     # Increase total plant area
            self.annual_labor_cost += rca_labor_cost   # Increase total labor costs

        # Plant Direct Costs Intermediate Calculations
        main_equipment_cost = self.ed_system_cost+self.bop_cost # ($) Main equipment cost
        auxiliary_equipment_cost = 0.02*main_equipment_cost # ($) Auxiliary equipment cost
        equipment_cost = main_equipment_cost + auxiliary_equipment_cost # ($) Total equipment cost
        installation_cost = 0.15 * equipment_cost # ($) Cost for equipment installation
        instrumentation_cost = 0.09 * equipment_cost # ($) Cost for instrumentation and DCS
        insulation_cost = 0.005 * equipment_cost # ($) Cost of insulation
        electrical_cost = 0.03 * equipment_cost # ($) Cost of electrical infrasturcture for plant 
        building_cost = 0.06 * equipment_cost # ($) Cost of buildings 
        land_cost = self.land_area_m2 * self.land_cost_per_m2 # ($) Cost of land

        # Plant Direct Costs Final Calculation ($)
        direct_plant_costs = (
            equipment_cost
            +self.import_tariff
            +self.transportation_cost
            +self.yard_improvements_cost
            +self.lab_equipment_cost
            +installation_cost
            +self.piping_system_cost
            +instrumentation_cost
            +insulation_cost
            +electrical_cost
            +building_cost
            +land_cost
        )

        # Plant Indirect Costs Intermediate Calculations
        engineering_procurement_cost = 0.07 * direct_plant_costs # ($) Engineering, procurement, and construction costs
        start_up_cost = 0.04 * direct_plant_costs # ($) Supervision, start-up, and training costs
        contingencies_cost = 0.005 * direct_plant_costs # ($) Contingencies cost

        # Plant Indirect Costs Final Calculation ($)
        indirect_plant_cost = engineering_procurement_cost+start_up_cost+contingencies_cost

        # Technical Operating Cost Intermediate Calculations
        membrane_cost = self.ed_system_cost/1.5 # ($) Cost of membranes 
        annual_membrane_replacement_cost = 0 # ($/yr) Yearly cost of membrane replacement
        for n in range(self.num_membrane_replacements):
            annual_membrane_replacement_cost = annual_membrane_replacement_cost + (membrane_cost/self.plant_lifetime_yrs * 0.9**(n+1)) # Cost of membranes anticipated to decrease by 10% with each replacement
        maintainance_repair_cost = 0.008 * direct_plant_costs # ($/yr) Yearly cost of maintenance and repairs 
        insurance_taxes_cost = 0.004 * direct_plant_costs # ($/yr) Yearly cost of insurances and local taxes
        annual_facility_dependent_cost = annual_membrane_replacement_cost+maintainance_repair_cost+insurance_taxes_cost # ($/yr) Yearly facility dependent cost

        # Technical Operating Cost Final Calculation ($/yr)
        annual_operating_cost = (
            self.annual_raw_materials_cost
            +self.annual_labor_cost
            +annual_facility_dependent_cost
            +self.annual_lab_qc_rd_cost
            +self.annual_consumables_cost
            +(self.waste_mass * self.waste_disposal_cost)  # Waste disposal cost
            +self.annual_energy_cost
            +self.annual_transport_cost
            +self.annual_misc_cost
            +self.annual_royalty_cost
        )

        # Capital Cost Intermediate Calculations
        direct_fixed_capital_cost = direct_plant_costs+indirect_plant_cost # ($) Direct fixed capital cost
        working_capital_cost = (
            self.annual_raw_materials_cost
            +self.annual_labor_cost
            +(self.waste_mass * self.waste_disposal_cost)  # Waste disposal cost
            +self.annual_energy_cost
            +self.annual_misc_cost
        ) # ($) Working capital cost

        # Capital Cost (CAPEX) Final Calculation ($)
        capital_cost = (
            direct_fixed_capital_cost
            +working_capital_cost
            +self.startup_cost
            +self.upfront_rd_cost
            +self.upfront_royalty_cost
        )

        # Average OPEX or Annual Operating Cost Intermediate Calcualations

        # Yearly techincal operating cost over plant lifetime
        lifetime_annual_operating_cost = np.zeros(self.plant_lifetime_yrs) 
        lifetime_annual_operating_cost[0] = annual_operating_cost 
        for i in range(len(lifetime_annual_operating_cost)-1):
            lifetime_annual_operating_cost[i+1] = lifetime_annual_operating_cost[i] * (1+self.inflation_rate)

        # Yearly loan repayment (constant) over plant lifetime
        S_lon = 0 # Sum used in calculation
        for t in range(self.recovery_period_yrs):
            S_lon = S_lon + (1+self.interest_rate)**t
        lifetime_annual_loan_repayment = np.zeros(self.plant_lifetime_yrs)
        for i in range(self.recovery_period_yrs):
            lifetime_annual_loan_repayment[i] = capital_cost * (1+self.interest_rate*S_lon)/S_lon

        # Yearly OPEX or annual operating cost over plant lifetime
        lifetime_annual_opex = np.zeros(self.plant_lifetime_yrs)
        for i in range(self.plant_lifetime_yrs):
            lifetime_annual_opex[i] = lifetime_annual_operating_cost[i] + lifetime_annual_loan_repayment[i]

        # Average OPEX or Annual Operating Cost Final Calcualation
        average_opex = np.mean(lifetime_annual_opex)

        # Loan repayment schedule
        loan_balance = np.zeros(self.recovery_period_yrs + 1)
        interest_payments = np.zeros(self.plant_lifetime_yrs)
        principal_payments = np.zeros(self.recovery_period_yrs)

        loan_balance[0] = capital_cost
        interest_payments[0] = capital_cost * self.interest_rate
        principal_payments[0] = lifetime_annual_loan_repayment[0] - interest_payments[0]
        loan_balance[1] = loan_balance[0] - principal_payments[0]

        for year in range(1, self.recovery_period_yrs):
            interest_payments[year] = interest_payments[year - 1] - self.interest_rate * principal_payments[year - 1]
            principal_payments[year] = lifetime_annual_loan_repayment[year] - interest_payments[year]
            loan_balance[year + 1] = loan_balance[year] - principal_payments[year]

        # Annual depreciation (constant over recovery period)
        annual_depreciation = np.zeros(self.plant_lifetime_yrs)
        depreciable_value = (direct_fixed_capital_cost - land_cost) * (1 - self.salvage_value_percent)
        annual_depreciation[:self.recovery_period_yrs] = depreciable_value / self.recovery_period_yrs


        # Define the objective function for root finding
        def npv_objective_for_credit(carbon_credit_value, target_npv):
            return self.npv_objective(
                carbon_credit_value,
                lifetime_annual_operating_cost=lifetime_annual_operating_cost,
                annual_depreciation=annual_depreciation,
                annual_interest_payment=interest_payments[:self.plant_lifetime_yrs],
                lifetime_annual_loan_repayment=lifetime_annual_loan_repayment[:self.plant_lifetime_yrs]
            ) - target_npv

        # Solve for carbon credit value that achieves the target NPV
        carbon_credit_solution = root_scalar(
            npv_objective_for_credit,
            args=(0.0,),  # Target NPV value, set to 0 for this case
            x0=0 # Initial guess for carbon credit value
        )

        # Final value
        carbon_credit_value = carbon_credit_solution.root

        if carbon_credit_value >=0:
            base_revenue = (self.mass_product * self.value_product) + self.estimated_cdr * carbon_credit_value
        else:
            base_revenue = (self.mass_product * self.value_product)

        # Yearly revenue of plant ($/yr)
        lifetime_annual_revenue = np.zeros(self.plant_lifetime_yrs)
        lifetime_annual_revenue[0] = base_revenue
        for i in range(len(lifetime_annual_revenue)-1):
            lifetime_annual_revenue[i+1] = lifetime_annual_revenue[i] * (1+self.inflation_rate)
        average_revenue = np.mean(lifetime_annual_revenue)

        # Yearly EBITDA ($/yr)
        lifetime_annual_ebitda = lifetime_annual_revenue - lifetime_annual_operating_cost

        # Yearly Taxable Income ($/yr)
        lifetime_taxable_income = lifetime_annual_ebitda - annual_depreciation - interest_payments[:self.plant_lifetime_yrs]

        # Yearly Tax Loss Carry Forward ($/yr)
        lifetime_tax_loss = np.zeros(self.plant_lifetime_yrs + 1)
        for i in range(self.plant_lifetime_yrs):
            lifetime_tax_loss[i + 1] = min(lifetime_tax_loss[i] + lifetime_taxable_income[i], 0)

        # Yearly Taxes ($/yr)
        lifetime_taxes = np.maximum(
            self.corporate_tax_rate * (lifetime_taxable_income + lifetime_tax_loss[1:]),
            0
        )
        # Yearly Net Cash Flow ($/yr)
        lifetime_net_cash_flow = lifetime_annual_ebitda - lifetime_annual_loan_repayment - lifetime_taxes

        # Yearly Discounted Cash Flow ($/yr)
        discount_factors = (1 + self.opportunity_cost_capital) ** np.arange(1, self.plant_lifetime_yrs + 1)
        lifetime_discounted_cash_flow = lifetime_net_cash_flow / discount_factors

        # Net Present Value ($)
        npv = np.sum(lifetime_discounted_cash_flow)

        # Profitability Index (PI)
        pi = npv / capital_cost

        # Average Discounted Cash Flow ($/yr)
        average_discounted_cash_flow = np.mean(lifetime_discounted_cash_flow)

        # Average Taxes ($/yr)
        average_taxes = np.mean(lifetime_taxes)

        # Average Co-Product Revenue
        annual_co_product_revenue = self.mass_product * self.value_product
        total_co_product_revenue = annual_co_product_revenue * (1 + self.inflation_rate) ** np.arange(self.plant_lifetime_yrs)
        average_co_product_revenue = np.mean(total_co_product_revenue)

        # Payback time with discount cash flow (years)
        if round(npv) == 0:
            payback_time = self.plant_lifetime_yrs
        else:
            cumulative_dcf = np.cumsum(lifetime_discounted_cash_flow)
            payback_time = next((i + 1 for i, val in enumerate(cumulative_dcf) if val >= 0), self.plant_lifetime_yrs)

        # Prepare results dictionary
        results = {
            "Estimated CDR (0.8 mol CO2:NaOH) (tCO2/yr)": round(self.estimated_cdr, 2),
            "Method of Acid Disposal": self.acid_disposal_method, 
            "Yearly Energy Cost ($/yr)": round(self.annual_energy_cost, 2),
            "Capital Cost for ED System ($)": round(self.ed_system_cost, 2),
            "Capital Cost for BOP System ($)": round(self.bop_cost, 2),
            "Import Tariff Cost for Imported Equipment ($)": round(self.import_tariff, 2),
            "Transportation Cost for Equipment ($)": round(self.transportation_cost, 2),
            "Yard Improvements and Civil Works ($)": round(self.yard_improvements_cost, 2),
            "Laboratory Equipment Cost ($)": round(self.lab_equipment_cost, 2),
            "Piping System Cost ($)": round(self.piping_system_cost, 2),
            "Total Land Area Required (m²)": round(self.land_area_m2, 2),
            "Cost of Land ($/m²)": round(self.land_cost_per_m2, 2),
            "Annual Raw Materials Cost ($/yr)": round(self.annual_raw_materials_cost, 2),
            "Annual Labor Cost ($/yr)": round(self.annual_labor_cost, 2),
            "Annual Lab/QC/R&D Cost ($/yr)": round(self.annual_lab_qc_rd_cost, 2),
            "Annual Consumables Cost ($/yr)": round(self.annual_consumables_cost, 2),
            "Annual Waste Treatment or Disposal Cost ($/yr)": round(self.waste_mass * self.waste_disposal_cost, 2),
            "Annual Transport Cost ($/yr)": round(self.annual_transport_cost, 2),
            "Annual Miscellaneous Cost (Marketing & Administration) ($/yr)": round(self.annual_misc_cost, 2),
            "Annual Running Royalties Cost ($/yr)": round(self.annual_royalty_cost, 2),
            "Startup Costs ($)": round(self.startup_cost, 2),
            "Upfront R&D Costs ($)": round(self.upfront_rd_cost, 2),
            "Upfront Royalties Costs ($)": round(self.upfront_royalty_cost, 2),
            "Number of Membrane Replacements over Plant Lifetime": self.num_membrane_replacements,
            "Plant Lifetime (Years)": self.plant_lifetime_yrs,
            "Learning Rate (%)": round(self.learning_rate * 100, 2),
            "Rate of Inflation (%)": round(self.inflation_rate * 100, 2),
            "Recovery Period of Investment (Years)": self.recovery_period_yrs,
            "Rate of Interest (%)": round(self.interest_rate * 100, 2),
            "Corporate Tax Rate (%)": round(self.corporate_tax_rate * 100, 2),
            "Salvage Rate (%)": round(self.salvage_value_percent * 100, 2),
            "Opportunity Cost of Capital (%)": round(self.opportunity_cost_capital * 100, 2),
            "Direct Plant Costs ($)": round(direct_plant_costs, 2),
            "Indirect Plant Costs ($)": round(indirect_plant_cost, 2),
            "Direct Fixed Capital Cost ($)": round(direct_fixed_capital_cost, 2),
            "Working Capital Cost ($)": round(working_capital_cost, 2),
            "Capital Cost (CAPEX) ($)": round(capital_cost, 2),
            "Annual Operating Cost ($/yr)": round(annual_operating_cost, 2),
            "Average Annual Operating Cost ($/yr)": round(average_opex, 2),
            "Average Co-Product Revenue ($/yr)": round(average_co_product_revenue, 2),
            "Average Discounted Cash Flow ($/yr)": round(average_discounted_cash_flow, 2),
            "Average Taxes ($/yr)": round(average_taxes, 2),
            "Profitability Index (PI)": round(pi, 2),
            "Carbon Credit Value ($/tCO2)": round(carbon_credit_value, 2),
            "Net Present Value (NPV) ($)": round(npv, 2),
            "Discounted Payback Time (Years)": round(payback_time, 2),
        }

        if  save_outputs:
            save_paths = [output_dir + "figures/", output_dir + "data/"]

            for savepath in save_paths:
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

            results_df = pd.DataFrame(results,index=[0]).T
            results_df = results_df.reset_index()
            results_df.columns = ["Parameter", "Values"]

            results_df.to_csv(
                save_paths[1] + "OAE_resultsCosts.csv", index=False
            )
                
    def run_costs_without_energy(self):
        return


    
if __name__ == "__main__":
    test = OAEInputs()

    res1 = initialize_power_chemical_ranges(
        oae_config=OAEInputs(),
        pump_config=PumpInputs(),
        seawater_config=SeaWaterInputs(),
        rca=RCALoadingCalculator(oae=OAEInputs(),
                                 seawater=SeaWaterInputs())
    )

    #EXAMPLE: Sin function for power input
    days = 365
    exTime = np.zeros(24 * days)  # Example time in hours
    for i in range(len(exTime)):
        exTime[i] = i + 1
    maxPwr = 500 * 10**4 # W
    Amp = maxPwr/2
    periodT = 24 
    movUp = Amp
    movSide = -1*math.pi/2
    exPwr = np.zeros(len(exTime))
    for i in range(len(exTime)):
        exPwr[i] = Amp*math.sin(2*math.pi/periodT*exTime[i] + movSide) + movUp
        if int(exTime[i]/24) % 5 == 1:
            exPwr[i] = exPwr[i] * 0.1

    # results = simulate_ocean_alkalinity_enhancement(
    #     ranges=res1,
    #     oae_config=OAEInputs(),
    #     seawater_config=SeaWaterInputs(),
    #     rca=RCALoadingCalculator(oae=OAEInputs(),
    #                              seawater=SeaWaterInputs()),
    #     power_profile=exPwr,
    #     initial_tank_volume_m3=0,
    # )

    ranges, res = run_ocean_alkalinity_enhancement_physics_model(
        power_profile_w=exPwr,
        power_capacity_w=maxPwr,
        initial_tank_volume_m3=0,
        oae_config=OAEInputs(),
        pump_config=PumpInputs(),
        seawater_config=SeaWaterInputs(),
        rca=RCALoadingCalculator(oae=OAEInputs(),
                                 seawater=SeaWaterInputs()),
        save_plots=True,
        show_plots=True,
        save_outputs=True,

    )

    costs= OAECosts(mass_product=res.M_rev_yr,
                  value_product=res.X_rev_yr,
                  waste_mass=res.M_disposed_yr,
                  waste_disposal_cost=res.X_disp,
                  estimated_cdr=res.M_co2est,
                  base_added_seawater_max_power=res.mol_OH_yr_MaxPwr,
                #   mass_rca=res.slurry_mass_max,
                  )

    costs.run(save_outputs=True)