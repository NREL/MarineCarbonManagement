"""Model of electrochemical mCC system"""

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
from typing import Tuple

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
        P_edMax (float): Total ED system power (W). Default is 2.5 MW.
        N_edMax (int): Maximum number of ED units. Default is 10.
        E_HCl (float): Energy required per mole of HCl (kWh/mol). Computed if not given.
        E_NaOH (float): Energy required per mole of NaOH (kWh/mol). Computed if not given.
        Q_edMax (float): Max ED system flow rate (m³/s). Default is based on empirical values.
        Q_OMax (float): Max overall intake flow (m³/s). Derived from ED flow if not provided.
        frac_EDflow (float): Fraction of total flow treated by ED. Computed if not given.
        frac_baseFlow (float): Fraction of ED-treated flow that becomes base. Default is ~0.571.
        frac_acidFlow (float): Fraction of ED-treated flow producing acid. Computed as 1 - frac_baseFlow.
        c_a (float): Concentration of generated acid (mol/L). Default is 0.9.
        c_b (float): Concentration of generated base (mol/L). Default is 0.7.
        use_storage_tanks (bool): Whether storage tanks are used. Default is True.
        store_hours (float): Storage duration (h). Set to 0 if tanks are disabled.
        acid_disposal_method (str): Acid disposal strategy. Must be one of ["sell acid"].
    """
    P_ed1: float = field(default=None)
    Q_ed1: float = field(default=None)
    N_edMin: int = field(default=1, validator=gt_zero)
    P_edMax: float = field(default=250*10**4, validator=gt_zero)
    N_edMax: int = field(default=10, validator=gt_zero)
    E_HCl: float = field(default=None)
    E_NaOH: float = field(default=None)
    Q_edMax: float = field(default=(660 + 495) / 1000 / 60, validator=gt_zero)
    Q_OMax: float = field(default=None)
    frac_EDflow: float = field(default=None)
    frac_baseFlow: float = field(default=660 / (660 + 495), validator=range_val(0, 1))
    frac_acidFlow: float = field(default=None)
    c_a: float = 0.9
    c_b: float = 0.7
    use_storage_tanks: bool = field(default=True)
    store_hours: float = field(default=12)
    acid_disposal_method: str = field(default="sell acid", validator=contains(["sell acid"]))

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
            self.Q_OMax = self.Q_edMax / 0.01
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
        tempC (float): Average seawater temperature in °C. Default is 25°C.
        sal (float): Average salinity in ppt. Default is equivalent to 1.2 mol/kg.
        dic_i (float): Initial DIC concentration in mol/L. Default is 2.2e-3 mol/L.
        pH_i (float): Initial pH of seawater. Default is 8.1.

    Derived Attributes:
        kw (float): Water dissociation constant at the specified temperature and salinity.
        SAL_i (float): Salinity converted to mol NaCl/m³.
        h_i (float): Initial hydrogen ion concentration in mol/L.
        dic_iu (float): Initial DIC in µmol/kg.
        ta_i (float): Total alkalinity in mol/L.
    """
    tempC: float = 25.0
    sal: float = sal_m_to_ppt(1.2)
    dic_i: float = 2*2.2e-3
    pH_i: float = 8.1

    kw: float = field(init=False)
    SAL_i: float = field(init=False)
    h_i: float = field(init=False)
    dic_iu: float = field(init=False)
    ta_i: float = field(init=False)

    def __attrs_post_init__(self):
        tempK = self.tempC + 273.15  # Temperature in Kelvin

        # Calculate water dissociation constant (kw)
        self.kw = math.exp(
            -13847.26 / tempK
            + 148.9652
            - 23.6521 * math.log(tempK)
            + (
                (118.67 / tempK - 5.977 + 1.0495 * math.log(tempK)) * self.sal**0.5
                - 0.01615 * self.sal
            )
        )

        # Convert salinity to mol NaCl/m³
        self.SAL_i = sal_ppt_to_m(self.sal) * 1000

        # Calculate initial hydrogen ion concentration
        self.h_i = 10**-self.pH_i

        # Convert DIC to µmol/kg
        self.dic_iu = m_to_umol_per_kg(self.dic_i)

        # Use PyCO2SYS to calculate total alkalinity
        results = pyco2.sys(
            par1=self.dic_iu,
            par2=self.pH_i,
            par1_type=2,  # DIC
            par2_type=3,  # pH
            salinity=self.sal,
            temperature=self.tempC,
        )
        self.ta_i = umol_per_kg_to_m(results["alkalinity"])

@define
class PumpInputs:
    """
    A class to define the input parameters for various pumps in the system.

    Attributes:
        y_pump (float): The constant efficiency of the pump. Default is 0.9.
        p_o_min_bar (float): The minimum pressure (in bar) for seawater intake with filtration. Default is 0.1.
        p_o_max_bar (float): The maximum pressure (in bar) for seawater intake with filtration. Default is 0.5.
        p_ed_min_bar (float): The minimum pressure (in bar) for ED (Electrodialysis) units. Default is 0.1.
        p_ed_max_bar (float): The maximum pressure (in bar) for ED (Electrodialysis) units. Default is 0.5.
        p_a_min_bar (float): The minimum pressure (in bar) for pumping acid. Default is 0.1.
        p_a_max_bar (float): The maximum pressure (in bar) for pumping acid. Default is 0.5.
        p_i_min_bar (float): The minimum pressure (in bar) for pumping seawater for acid addition. Default is 0.
        p_i_max_bar (float): The maximum pressure (in bar) for pumping seawater for acid addition. Default is 0.
        p_b_min_bar (float): The minimum pressure (in bar) for pumping base. Default is 0.1.
        p_b_max_bar (float): The maximum pressure (in bar) for pumping base. Default is 0.5.
        p_f_min_bar (float): The minimum pressure (in bar) for released seawater. Default is 0.
        p_f_max_bar (float): The maximum pressure (in bar) for released seawater. Default is 0.
        """

    y_pump: float = 0.9
    p_o_min_bar: float = 0.1
    p_o_max_bar: float = 0.5
    p_ed_min_bar: float = 0.1
    p_ed_max_bar: float = 0.5
    p_a_min_bar: float = 0.1
    p_a_max_bar: float = 0.5
    p_i_min_bar: float = 0
    p_i_max_bar: float = 0
    p_b_min_bar: float = 0.1
    p_b_max_bar: float = 0.5
    p_f_min_bar: float = 0
    p_f_max_bar: float = 0

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
class OAERangeOutputs:
    """
    A class to represent the ocean alkalinity enhancement (OAE) device power and chemical ranges under each scenario.

    Attributes:
        S1 (dict): Chemical and power ranges for scenario 1 (e.g., tank filled).
            - "volAcid": Volume of acid (L).
            - "volBase": Volume of base (L).
            - "mol_OH": mol OH added to seawater at each time (mol).
            - "mol_HCl": mol HCl excess acid generated (mol).
            - "pH_f": Final pH of the solution.
            - "dic_f": Final dissolved inorganic carbon concentration (mol/L).
            - "ta_f": Final total alkalinity concentrations (mol/L).
            - "sal_f": Final salinity of the solution (ppt).
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
        V_a3_min (float): Minimum volume of acid required for S3.
        V_b3_min (float): Minimum volume of base required for S3.
        N_range (int): Number of ED units active in S1, S3, S4.
        S2_tot_range (int): Number of ED units active in S2.
        pump_power_min (float): Minimum pump power in MW.
        pump_power_max (float): Maximum pump power in MW.
        sep_power_min (float): Minimum separation power in MW.
        sep_power_max (float): Maximum separation power in MW.
        comp_power_min (float): Minimum compression power in MW.
        comp_power_max (float): Maximum compression power in MW.
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
        "mol_OH",
        "mol_HCl",
        "pH_f",
        "dic_f",
        "ta_f",
        "sal_f",
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
        S1["Qin"][i] = round(p.pumpO.Q,2)  # (m3/s) Intake

        # Acid and Base Concentrations
        p.pumpA.Q = p.pumpED.Q * oae_config.frac_acidFlow  # Acid flow rate
        C_a = (1 / p.pumpA.Q) * (
            P_EDi / (3600 * (E_HCl * 1000)) - (p.pumpED.Q * h_i * 1000)
        )  # (mol/m3) Acid concentration from ED units
        if C_a > seawater_config.SAL_i:
            C_a = seawater_config.SAL_i  # Limit acid concentration to seawater salinity
            warnings.warn(
                f"{__name__}: Acid concentration exceeds seawater salinity. Limiting to {C_a:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_a = (seawater_config.SAL_i - C_a) * p.pumpB.Q # (mol/s) NaCl needed to maintain salinity
        S1["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        S1["volAcid"][i] = round(p.pumpA.Q * 3600,2) # (m3) all acid made is excess
        S1["mol_HCl"][i] = C_a * p.pumpA.Q * 3600 # (mol) Excess acid generated
        p.pumpB.Q = p.pumpED.Q - p.pumpA.Q  # Base flow rate
        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        if C_b > seawater_config.SAL_i:
            C_b = seawater_config.SAL_i
            warnings.warn(
                f"{__name__}: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
        S1["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units
        S1["mol_OH"][i] = C_b * p.pumpB.Q * 3600 # (mol) OH added to seawater

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
        SAL_f = (SAL_b * p.pumpB.Q + seawater_config.SAL_i * p.pumpI.Q) / (
            p.pumpB.Q + p.pumpI.Q
        )
        S1["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # (ppt) Salinity after base addition

        # Define input conditions for mixing the base and the brine
        kwargs = dict(
            par1 = ta_fu, # Total alkalinity in umol/kg
            par2 = seawater_config.dic_iu, # DIC in umol/kg
            par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
            par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = S1["sal_f"][i],  # Salinity of the sample (ppt)
            temperature = seawater_config.tempC,  # Temperature at input conditions (C)
        )
        results = pyco2.sys(**kwargs)
        S1["pH_f"][i] = results["pH"] # (unitless) pH after base addition
        S1["dic_f"][i] = dic_i # (mol/L) DIC after base addition

        # Outtake
        p.pumpF.Q = p.pumpI.Q + p.pumpB.Q  # (m3/s) Outtake flow rate
        S1["Qout"][i] = round(p.pumpF.Q,2)  # (m3/s) Outtake

        # Power ranges for S1
        S1["pwrRanges"][i] = round(
            P_EDi
            + p.pumpED.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpB.power()
            + p.pumpF.power()
        ,2)

        P_minS1_tot = min(S1["pwrRanges"])

        ############################### S3: Chem Ranges: ED not active, tanks not zeros ##################
        P_EDi = 0  # ED Unit is off
        p.pumpED.Q = 0  # ED Unit is off
        p.pumpO.Q = (
            (1 / oae_config.frac_EDflow - 1) * (i + N_edMin) * Q_ed1
        )  # Flow rates for intake based on equivalent ED units that would be active
        S3["Qin"][i] = round(p.pumpO.Q,2)
        p.pumpI.Q = p.pumpO.Q  # since no flow is going to the ED unit
        p.pumpB.Q = (
            (i + N_edMin) * Q_ed1 * oae_config.frac_baseFlow
        )  # Flow rate for base pump based on equivalent ED units that would be active
        p.pumpA.Q = 0 # No waste acid is pumped in this case

        # Change in volume due to acid and base use
        S3["volAcid"][i] = round(-p.pumpA.Q * 3600,2)  # (m3) volume of acid lost by the tank
        S3["volBase"][i] = round(-p.pumpB.Q * 3600,2)  # (m3) volume of base lost by the tank

        # The concentration of acid and base produced does not vary with flow rate
        # Also does not vary with power since the power for the ED units scale directly with the flow rate
        C_b = (1 / p.pumpB.Q_min) * (
            P_ed1 * N_edMin / (3600 * (E_NaOH * 1000))
            - (p.pumpED.Q_min * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        if C_b > seawater_config.SAL_i:
            C_b = seawater_config.SAL_i
            warnings.warn(
                f"{__name__}: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
        
        S3["c_b"][i] = C_b / 1000  # (mol/L) Base concentration used in S3
        S3["mol_OH"][i] = p.pumpB.Q * C_b * 3600 # (mol) OH added to seawater in S3

        # Find TA Before Base Addition
        TA_i = ta_i * 1000  # (mol/m3)
        # Find TA After Base Addition
        TA_f = (TA_i * p.pumpI.Q + C_b * p.pumpB.Q) / (
            p.pumpI.Q + p.pumpB.Q
        )  # (mol/m3)

        S3["ta_f"][i] = TA_f / 1000 # (mol/L) Total alkalinity after base addition

        # Find effluent chem after base addition
        ta_fu = m_to_umol_per_kg(S3["ta_f"][i]) 
        SAL_f = (SAL_b * p.pumpB.Q + seawater_config.SAL_i * p.pumpI.Q) / (
            p.pumpB.Q + p.pumpI.Q
        )
        S3["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # (ppt) Salinity after base addition

        # Define input conditions for mixing the base and the brine
        kwargs = dict(
            par1 = ta_fu, # Total alkalinity in umol/kg
            par2 = seawater_config.dic_iu, # DIC in umol/kg
            par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
            par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = S3["sal_f"][i],  # Salinity of the sample (ppt)
            temperature = seawater_config.tempC,  # Temperature at input conditions (C)
        )
        results = pyco2.sys(**kwargs)
        S3["pH_f"][i] = results["pH"] # (unitless) pH after base addition
        S3["dic_f"][i] = dic_i # (mol/L) DIC after base addition

        # Outtake
        p.pumpF.Q = p.pumpI.Q + p.pumpB.Q  # (m3/s) Outtake flow rate
        S3["Qout"][i] = round(p.pumpF.Q,2)

        # Power ranges for S3
        S3["pwrRanges"][i] = round(
            P_EDi
            + p.pumpED.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpB.power()
            + p.pumpF.power()
        ,2)
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
        if C_a > seawater_config.SAL_i:
            C_a = seawater_config.SAL_i  # Limit acid concentration to seawater salinity
            warnings.warn(
                f"{__name__}: Acid concentration exceeds seawater salinity. Limiting to {C_a:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_a = (seawater_config.SAL_i - C_a) * p.pumpB.Q # (mol/s) NaCl needed to maintain salinity
        S4["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED4.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        if C_b > seawater_config.SAL_i:
            C_b = seawater_config.SAL_i
            warnings.warn(
                f"{__name__}: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
        S4["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units

        # Excess acid generated
        S4["volAcid"][i] = round(p.pumpA.Q * 3600, 2) # (m3) volume of excess acid after timestep
        S4["mol_HCl"][i] = round(C_a * p.pumpA.Q * 3600,2)  # (mol) moles of excess acid generated

        # Base added to the tank
        # n_bT = C_b * p.pumpB.Q  # (mol/s) rate of base moles added to tank
        S4["volBase"][i] = round(p.pumpB.Q * 3600,2)  # volume of base in tank after time step

        # Intake (ED4 pump not O pump is used)
        p.pumpO.Q = 0  # Need intake for ED & min CC
        S4["Qin"][i] = round(p.pumpED4.Q,2)  # (m3/s) Intake

        # Other pumps not used
        p.pumpI.Q = 0  # Intake remaining after diversion to ED

        # Outtake
        p.pumpF.Q = 0  # Outtake flow rate
        S4["Qout"][i] = round(p.pumpF.Q,2)  # (m3/s) Outtake

        # Since no OAE is conducted the final DIC and pH is the same as the initial
        S4["pH_f"][i] = pH_i
        S4["dic_f"][i] = dic_i  # (mol/L)
        S4["sal_f"][i] = seawater_config.sal # (ppt) 
        S4["ta_f"][i] = ta_i # (mol/L)

        # Power ranges for S4
        S4["pwrRanges"][i] = round(
            P_EDi
            + p.pumpED4.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpB.power()
            + p.pumpF.power()
        ,2)
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
        if C_a > seawater_config.SAL_i:
            C_a = seawater_config.SAL_i
            warnings.warn(
                f"{__name__}: Acid concentration exceeds seawater salinity. Limiting to {C_a:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_a = (seawater_config.SAL_i - C_a) * p.pumpB.Q  # (mol/s) NaCl needed to maintain salinity
        S2["c_a"][i] = C_a / 1000  # (mol/L) Acid concentration from ED units
        S2["mol_HCl"][i] = C_a * p.pumpA.Q * 3600  # (mol) Excess acid generated

        C_b = (1 / p.pumpB.Q) * (
            P_EDi / (3600 * (E_NaOH * 1000)) - (p.pumpED.Q * (kw / h_i) * 1000)
        )  # (mol/m3) Base concentration from ED units
        if C_b > seawater_config.SAL_i:
            C_b = seawater_config.SAL_i
            warnings.warn(
                f"{__name__}: Base concentration exceeds seawater salinity. Limiting to {C_b:.2f} mol/m³. Check the ED efficiency input value",
                UserWarning,
            )
        n_sal_b = (seawater_config.SAL_i - C_b) * p.pumpB.Q # mol/s of nacl after creation of base
        SAL_b = n_sal_b / p.pumpB.Q # (mol/m3) concentration of nacl after base formation
    
        S2["c_b"][i] = C_b / 1000  # (mol/L) Base concentration from ED units

        # Excess acid
        S2["volAcid"][i] = round(p.pumpA.Q * 3600, 2) # (m3) all acid is excess

        # Amount of base added for OAE
        Q_bOAE = S2_ranges[i, 0] * Q_ed1 * oae_config.frac_baseFlow  # flow rate used for OAE
        S2["mol_OH"][i] = C_b * Q_bOAE * 3600  # (mol) OH added to seawater

        # Base addition to tank
        Q_bT = p.pumpB.Q - Q_bOAE  # (m3/s) flow rate of base to tank
        #n_bT = C_b * Q_bT  # (mol/s) rate of base moles added to tank
        S2["volBase"][i] = round(Q_bT * 3600,2)  # (m3) base added to tank

        # Base addition to seawater
        p.pumpI.Q = (1 / oae_config.frac_EDflow - 1) * (Q_bOAE / oae_config.frac_baseFlow)

        # Seawater Intake
        p.pumpO.Q = p.pumpI.Q + p.pumpED.Q   # total seawater intake
        S2["Qin"][i] = round(p.pumpO.Q,2)  # (m3/s) intake

        # Find TA Before Base Addition
        TA_i = ta_i * 1000  # (mol/m3)

        # Find TA After Base Addition
        TA_f = (TA_i * p.pumpI.Q + C_b * Q_bOAE)/(p.pumpI.Q + Q_bOAE) # (mol/m3)
        S2["ta_f"][i] = TA_f/1000 # (mol/L)

        # Find effluent chem after base addition
        ta_fu = m_to_umol_per_kg(S2["ta_f"][i]) 
        SAL_f = (SAL_b * p.pumpB.Q + seawater_config.SAL_i * p.pumpI.Q) / (
            p.pumpB.Q + p.pumpI.Q
        )
        S2["sal_f"][i] = sal_m_to_ppt(SAL_f/1000) # (ppt) Salinity after base addition

        # Define input conditions for mixing the base and the brine
        kwargs = dict(
            par1 = ta_fu, # Total alkalinity in umol/kg
            par2 = seawater_config.dic_iu, # DIC in umol/kg
            par1_type = 1,  # The first parameter supplied is of type "1", which is "TA"
            par2_type = 2,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = S2["sal_f"][i],  # Salinity of the sample (ppt)
            temperature = seawater_config.tempC,  # Temperature at input conditions (C)
        )
        results = pyco2.sys(**kwargs)
        S2["pH_f"][i] = results["pH"] # (unitless) pH after base addition
        S2["dic_f"][i] = dic_i # (mol/L) DIC after base addition

        # Outtake
        p.pumpF.Q = p.pumpI.Q + Q_bOAE  # (m3/s) Outtake flow rate
        S2["Qout"][i] = round(p.pumpF.Q,2)  # (m3/s) Outtake

        # Power ranges for S2
        S2["pwrRanges"][i] = round(
            P_EDi
            + p.pumpED.power()
            + p.pumpO.power()
            + p.pumpA.power()
            + p.pumpI.power()
            + p.pumpB.power()
            + p.pumpF.power()
        ,2)
        P_minS2_tot = min(S2["pwrRanges"])

    ##################### S5: Chem Ranges: When all input power is excess ############################
    S5["volAcid"] = 0  # No acid generated
    S5["volBase"] = 0  # No base generated
    S5["mol_OH"] = 0  # No base addition
    S5["mol_HCl"] = 0  # No excess acid generated
    S5["pH_f"] = pH_i  # No changes in sea pH
    S5["dic_f"] = dic_i  # (mol/L) No changes in sea DIC
    S5["ta_f"] = ta_i  # (mol/L) No changes in sea TA
    S5["sal_f"] = seawater_config.sal  # (ppt) No changes in sea salinity
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

    # Define Tank Max Volumes (note there are two but they have the same volume)
    V_bT_max = round(
        p.pumpED.Q_min * oae_config.frac_baseFlow * oae_config.store_hours * 3600
    ,2)  # (m3) enables enough storage for 1 day or the hours from storeTime

    # Volume needed for S3
    V_b3_min = round(p.pumpED.Q_min * oae_config.frac_baseFlow * 3600, 2)  # enables minimum OAE for 1 timestep

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
                - volAcid (array): Volume of excess acid at each time step (m³).
                - volBase (array): Volume of base added or removed from tanks at each time step (m³).
                - tank_vol_b (array): Volume of base in the tank at each time step (m³).
                - mol_OH (array): Moles of OH added to seawater at each time step (mol).
                - mol_HCl (array): Moles of excess acid generated at each time step (mol).
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
        volOAEbase_yr (float): Average volume of effluent base added to seawater over the year (m³).
        mol_OH_yr_MaxPwr (float): Total moles of OH added to seawater over the year under maximum power conditions (mol).
        mol_HCl_yr (float): Total moles of excess acid generated over the year (mol).
        volXSacid_yr (float): Total volume of excess acid generated over the year (m³).
        pH_HCl_excess (float): pH of seawater after excess acid addition.
        m_adSolid_yr (float): Total mass of alkaline solid added over the year (g).
        M_rev_yr (float): Mass of Products Made (tonnes/yr).
        X_rev_yr (float): Value of Products Made ($/yr).
        M_co2est (float): Estimated mass of products captured over the year (tonne).
        M_co2cap (float): Mass of CO2 captured over the year (tonne).
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
    volOAEbase_yr: float
    mol_OH_yr_MaxPwr: float
    mol_HCl_yr: float
    volXSacid_yr: float
    pH_HCl_excess: float
    m_adSolid_yr: float
    M_rev_yr: float
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
    power_profile,
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
        power_profile (np.ndarray): Array representing the available power at each time step (W).
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

    # Define the array names
    keys = [
        "N_ed",  # Number of ED units active
        "P_xs",  # (W) Excess power at each time
        "volAcid",  # (m³) Volume of acid added/removed to/from tanks at each time
        "volBase",  # (m³) Volume of base added/removed to/from tanks at each time
        "tank_vol_b",  # (m³) Volume of base in the tank at each time
        "mol_OH",  # (mol) Moles of OH added to seawater at each time
        "mol_HCl",  # (mol) Moles of excess acid generated at each time
        "pH_f",  # Final pH at each time
        "dic_f",  # (mol/L) Final DIC at each time
        "ta_f",  # (mol/L) Final total alkalinity at each time
        "sal_f",  # (ppt) Final salinity at each time
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
            OAE_outputs["volAcid"][i] = ranges.S1["volAcid"][i_ed]
            OAE_outputs["volBase"][i] = ranges.S1["volBase"][i_ed]
            OAE_outputs["mol_OH"][i] = ranges.S1["mol_OH"][i_ed]
            OAE_outputs["mol_HCl"][i] = ranges.S1["mol_HCl"][i_ed]
            OAE_outputs["pH_f"][i] = ranges.S1["pH_f"][i_ed]
            OAE_outputs["dic_f"][i] = ranges.S1["dic_f"][i_ed]
            OAE_outputs["ta_f"][i] = ranges.S1["ta_f"][i_ed]
            OAE_outputs["sal_f"][i] = ranges.S1["sal_f"][i_ed]
            OAE_outputs["c_a"][i] = ranges.S1["c_a"][i_ed]
            OAE_outputs["c_b"][i] = ranges.S1["c_b"][i_ed]
            OAE_outputs["Qin"][i] = ranges.S1["Qin"][i_ed]
            OAE_outputs["Qout"][i] = ranges.S1["Qout"][i_ed]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S1["alkaline_solid_added"][i_ed]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S1["alkaline_to_acid"][i_ed]
            OAE_outputs["S_t"][i] = 1

            # Update Tank Volumes
            tank_vol_b[i + 1] = round(tank_vol_b[i] + OAE_outputs["volBase"][i],2)
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = round(ranges.V_bT_max,2)

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
            OAE_outputs["volAcid"][i] = ranges.S2["volAcid"][i_s2]
            OAE_outputs["volBase"][i] = ranges.S2["volBase"][i_s2]
            OAE_outputs["mol_OH"][i] = ranges.S2["mol_OH"][i_s2]
            OAE_outputs["mol_HCl"][i] = ranges.S2["mol_HCl"][i_s2]
            OAE_outputs["pH_f"][i] = ranges.S2["pH_f"][i_s2]
            OAE_outputs["dic_f"][i] = ranges.S2["dic_f"][i_s2]
            OAE_outputs["ta_f"][i] = ranges.S2["ta_f"][i_s2]
            OAE_outputs["sal_f"][i] = ranges.S2["sal_f"][i_s2]
            OAE_outputs["c_a"][i] = ranges.S2["c_a"][i_s2]
            OAE_outputs["c_b"][i] = ranges.S2["c_b"][i_s2]
            OAE_outputs["Qin"][i] = ranges.S2["Qin"][i_s2]
            OAE_outputs["Qout"][i] = ranges.S2["Qout"][i_s2]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S2["alkaline_solid_added"][i_s2]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S2["alkaline_to_acid"][i_s2]
            OAE_outputs["S_t"][i] = 2

            # Update Tank Volume
            tank_vol_b[i + 1] = round(tank_vol_b[i] + OAE_outputs["volAcid"][i],2)
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = round(ranges.V_bT_max,2)

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
            OAE_outputs["mol_OH"][i] = ranges.S3["mol_OH"][i_ed]
            OAE_outputs["mol_HCl"][i] = ranges.S3["mol_HCl"][i_ed]
            OAE_outputs["pH_f"][i] = ranges.S3["pH_f"][i_ed]
            OAE_outputs["dic_f"][i] = ranges.S3["dic_f"][i_ed]
            OAE_outputs["ta_f"][i] = ranges.S3["ta_f"][i_ed]
            OAE_outputs["sal_f"][i] = ranges.S3["sal_f"][i_ed]
            OAE_outputs["c_a"][i] = ranges.S3["c_a"][i_ed]
            OAE_outputs["c_b"][i] = ranges.S3["c_b"][i_ed]
            OAE_outputs["Qin"][i] = ranges.S3["Qin"][i_ed]
            OAE_outputs["Qout"][i] = ranges.S3["Qout"][i_ed]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S3["alkaline_solid_added"][i_ed]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S3["alkaline_to_acid"][i_ed]
            OAE_outputs["S_t"][i] = 3

            # Update Tank Volume
            tank_vol_b[i + 1] = round(tank_vol_b[i] + OAE_outputs["volBase"][i],2)
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = round(ranges.V_bT_max,2)

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
            OAE_outputs["volAcid"][i] = ranges.S4["volAcid"][i_ed]
            OAE_outputs["volBase"][i] = ranges.S4["volBase"][i_ed]
            OAE_outputs["mol_OH"][i] = ranges.S4["mol_OH"][i_ed]
            OAE_outputs["mol_HCl"][i] = ranges.S4["mol_HCl"][i_ed]
            OAE_outputs["pH_f"][i] = ranges.S4["pH_f"][i_ed]
            OAE_outputs["dic_f"][i] = ranges.S4["dic_f"][i_ed]
            OAE_outputs["ta_f"][i] = ranges.S4["ta_f"][i_ed]
            OAE_outputs["sal_f"][i] = ranges.S4["sal_f"][i_ed]
            OAE_outputs["c_a"][i] = ranges.S4["c_a"][i_ed]
            OAE_outputs["c_b"][i] = ranges.S4["c_b"][i_ed]
            OAE_outputs["Qin"][i] = ranges.S4["Qin"][i_ed]
            OAE_outputs["Qout"][i] = ranges.S4["Qout"][i_ed]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S4["alkaline_solid_added"][i_ed]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S4["alkaline_to_acid"][i_ed]
            OAE_outputs["S_t"][i] = 4

            # Update Tank Volume
            tank_vol_b[i + 1] = round(tank_vol_b[i] + OAE_outputs["volBase"][i],2)
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = round(ranges.V_bT_max,2)

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
            OAE_outputs["volAcid"][i] = ranges.S5["volAcid"]
            OAE_outputs["volBase"][i] = ranges.S5["volBase"]
            OAE_outputs["mol_OH"][i] = ranges.S5["mol_OH"]
            OAE_outputs["mol_HCl"][i] = ranges.S5["mol_HCl"]
            OAE_outputs["pH_f"][i] = ranges.S5["pH_f"]
            OAE_outputs["dic_f"][i] = ranges.S5["dic_f"]
            OAE_outputs["ta_f"][i] = ranges.S5["ta_f"]
            OAE_outputs["sal_f"][i] = ranges.S5["sal_f"]
            OAE_outputs["c_a"][i] = ranges.S5["c_a"]
            OAE_outputs["c_b"][i] = ranges.S5["c_b"]
            OAE_outputs["Qin"][i] = ranges.S5["Qin"]
            OAE_outputs["Qout"][i] = ranges.S5["Qout"]
            OAE_outputs["alkaline_solid_added"][i] = ranges.S5["alkaline_solid_added"]
            OAE_outputs["alkaline_to_acid"][i] = ranges.S5["alkaline_to_acid"]
            OAE_outputs["S_t"][i] = 5

            # Update Tank Volume
            tank_vol_b[i + 1] = round(tank_vol_b[i] + OAE_outputs["volBase"][i],2)
            OAE_outputs["tank_vol_b"][i] = tank_vol_b[i + 1]

            # Ensure Tank Volume Can't be More Than Max
            if tank_vol_b[i + 1] > ranges.V_bT_max:
                tank_vol_b[i + 1] = round(ranges.V_bT_max,2)

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
    mol_OH_yr = round(mol_OH_total, 2) # mol/yr

    # Approximation for CDR Scale
    N_co2est = 0.8 * mol_OH_yr # (mol CO2/yr) Estimated moles of CO2 absorbed
    M_co2est = N_co2est * 44 /1000000 # (tCO2/yr) Estimated mass of CO2 absorbed

    # Approximation for CDR capacity
    # Yearly alkalinity addition under constant max power conditions
    mol_OH_yr_MaxPwr = OH_max_addition_mol * 8760  # mol/yr
    N_co2cap = 0.8 * mol_OH_yr_MaxPwr # (mol CO2/yr) Estimated moles of CO2 absorbed
    M_co2cap = N_co2cap * 44 /1000000 # (tCO2/yr) Estimated mass of CO2 absorbed

    # Average pH, DIC, and sal of effluent when OAE is done
    if np.any(OAE_outputs["pH_f"] != seawater_config.pH_i):
        pH_avg = round(np.mean(OAE_outputs["pH_f"][OAE_outputs["pH_f"] != seawater_config.pH_i]), 2)
    else:
        pH_avg = 0.0
    if np.any(OAE_outputs["dic_f"] != seawater_config.dic_i):
        dic_avg = round(np.mean(OAE_outputs["dic_f"][OAE_outputs["dic_f"] != seawater_config.dic_i]), 2)
    else:
        dic_avg = 0.0
    
    if np.any(OAE_outputs["sal_f"] != seawater_config.sal):
        sal_avg = round(np.mean(OAE_outputs["sal_f"][OAE_outputs["sal_f"] != seawater_config.sal]), 2)
    else:
        sal_avg = 0.0
    if np.any(OAE_outputs["ta_f"] != seawater_config.ta_i):
        ta_avg = round(np.mean(OAE_outputs["ta_f"][OAE_outputs["ta_f"] != seawater_config.ta_i]), 2)
    else:
        ta_avg = 0.0
    
    # pH of excess acid
    pH_HCl_excess = round(-math.log10(ranges.S1["c_a"][0]),2)

    # Average yearly acid production
    mol_HCl_yr = sum(OAE_outputs["mol_HCl"][0:8760])

    # Average volume of acid to dispose of
    volXSacid_yr = sum(OAE_outputs["volAcid"][0:8760])

    # Average mass of alkaline solids needed for neutralization
    m_adSolid_yr = sum(OAE_outputs["alkaline_solid_added"][0:8760])

    # Average mass of sold products and value per ton
    if oae_config.acid_disposal_method == "sell acid":
        M_rev_yr = round(volXSacid_yr * R_H2O/10**3,2) # mass of sold acid in tons/yr 
        X_rev = 9 # $/ton for dilute acid

    else:
        # Assume acid is sold
        M_rev_yr = round(volXSacid_yr * R_H2O/10**3,2) # mass of sold acid in tons/yr 
        X_rev = 9 # $/ton for dilute acid

    # Average volume of alkaline seawater added to ocean
    volOAEbase_yr = sum(OAE_outputs["Qout"][0:8760]*3600) # (m3/yr) Volume of alkaline seawater added to ocean

    # Overall capacity factor (times system is on)
    OAE_timeFrac = nON/len(OAE_outputs["N_ed"])

    # OAE capacity factor (compare OAE with max if max power always available)
    OAEcapFact = mol_OH_yr/mol_OH_yr_MaxPwr

    # Print and determine the energy capacity factor (compare energy availability with max if max power always available)
    EcapFact = sum(power_profile[0:8760]) / (max(power_profile)*8760) 

    return OAEOutputs(
        OAE_outputs=OAE_outputs,
        mol_OH_yr=mol_OH_yr,
        pH_avg=pH_avg,
        dic_avg=dic_avg,
        ta_avg=ta_avg,
        sal_avg=sal_avg,
        volOAEbase_yr=volOAEbase_yr,
        mol_OH_yr_MaxPwr=mol_OH_yr_MaxPwr,
        mol_HCl_yr=mol_HCl_yr,
        volXSacid_yr=volXSacid_yr,
        pH_HCl_excess=pH_HCl_excess,
        m_adSolid_yr=m_adSolid_yr,
        M_rev_yr=M_rev_yr,
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
    """
    days: np.ndarray  # Days of the year (1-365)
    volume_seawater_out: np.ndarray  # Volume of seawater treated and released (m³)
    flow_rate_seawater_out: np.ndarray  # Flow rate of seawater treated and released (m³/s)
    pH_avg_out: np.ndarray  # Average pH of treated seawater
    dic_out: np.ndarray  # Dissolved inorganic carbon concentration in treated seawater (mol/L)
    sal_out: np.ndarray  # Salinity of treated seawater (ppt)
    ta_out: np.ndarray  # Total alkalinity of treated seawater (mol/L)

def run_ocean_alkalinity_enhancement_physics_model(
    power_profile_w,
    initial_tank_volume_m3,
    oae_config: OAEInputs,
    pump_config: PumpInputs,
    seawater_config: SeaWaterInputs,
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
    )
    res = simulate_ocean_alkalinity_enhancement(
        ranges=ranges,
        oae_config=oae_config,
        seawater_config=seawater_config,
        power_profile=power_profile_w,
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
    for i in range(365):
        iDays[i] = i+1
        VswOut[i] = 3600*sum(res.OAE_outputs["Qout"][24*i:24*(i+1)-1])
        QswOut[i] = VswOut[i]/(24*60*60) # average output flowrate in m3/s
        pHon = 0
        pHsum = 0
        dicSum = 0
        salSum = 0
        taSum = 0
        for j in range(24*i, 24*(i+1)):
            if res.OAE_outputs["Qout"][j] > 0:
                pHon = pHon+1
                pHsum = pHsum + res.OAE_outputs["pH_f"][j]
                dicSum = dicSum + res.OAE_outputs["dic_f"][j]
                salSum = salSum + res.OAE_outputs["sal_f"][j]
                taSum = taSum + res.OAE_outputs["ta_f"][j]
        if pHon > 0:
            pHavgOut[i] = round(pHsum/pHon,2) # average effluent pH when plant is active at least once in a day
            dicOut[i] = dicSum/pHon # average DIC when plant can be active at least once in a day
            salOut[i] = round(salSum/pHon,2)
            taOut[i] = taSum/pHon # average TA when plant can be active at least once in a day
        else:
            pHavgOut[i] = seawater_config.pH_i
            dicOut[i] = seawater_config.dic_i
            salOut[i] = seawater_config.sal
            taOut[i] = seawater_config.ta_i

    if save_plots or save_outputs:
        save_paths = [output_dir + "figures/", output_dir + "data/"]

        for savepath in save_paths:
            if not os.path.exists(savepath):
                os.makedirs(savepath)

    if save_outputs:
            # Design Inputs
            design_inputs = {
                "Maximum Power Need for ED System (W)": oae_config.P_edMax,
                "Maximum Flow Rate for ED System (m3/s)": oae_config.Q_edMax,
                "Percentage of ED Flow that Becomes Base (%)": round(oae_config.frac_baseFlow*100,2),
                "Concentration of Acid Made by ED (M)": oae_config.c_a,
                "Concentration of Base Made by ED (M)": oae_config.c_b,
                "Minimum Number of ED Units Used": oae_config.N_edMin,
                "Maximum Number of ED Units Used": oae_config.N_edMax,
                "Acid Production Efficiency (Wh/mol HCl)": oae_config.E_HCl,
                "Base Production Efficiency (Wh/mol NaOH)": oae_config.E_NaOH,
                "Method of Acid Disposal": oae_config.acid_disposal_method,
                "Average Seawater Temperature (C)": seawater_config.tempC,
                "Average Seawater Salinity (ppt)": seawater_config.sal,
                "Initial Seawater pH": seawater_config.pH_i,
                "Initial Seawater DIC (M)": seawater_config.dic_i,
            }
            diDF = pd.DataFrame(design_inputs, index=[0]).T
            diDF = diDF.reset_index()
            diDF.columns = ["Parameter", "Values"]
            diDF.to_csv(save_paths[1] + "OAE_timeDependentResults.csv", index=False)

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
                "Volume of Excess Acid (m3)": res.OAE_outputs["volAcid"],
                "Base Tank Volume (m3)": res.OAE_outputs["tank_vol_b"],
                "Base Added Volume (m3)": res.OAE_outputs["volBase"],
                "Seawater Flow Rate Into Plant (m3/s)": res.OAE_outputs["Qin"],
                "Seawater Flow Rate Out of Plant (m3/s)": res.OAE_outputs["Qout"],
                "pH of Effluent Seawater": res.OAE_outputs["pH_f"],
                "DIC of Effluent Seawater (mol/L)": res.OAE_outputs["dic_f"],
                "TA of Effluent Seawater (mol/L)": res.OAE_outputs["ta_f"],
                "Salinity of Effluent Seawater (ppt)": res.OAE_outputs["sal_f"],
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
            scenVolAcid = np.concatenate(
                [getattr(ranges, key)["volAcid"] for key in ["S1", "S2", "S3", "S4"]]
            )
            scenVolBase = np.concatenate(
                [getattr(ranges, key)["volBase"] for key in ["S1", "S2", "S3", "S4"]]
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

            # Create dictionary and save CSV
            scenDict = {
                "Scenario": scenNames,
                "ED Units Used for OAE (or Equivalent for S3)": scenEDoae,
                "ED Units Used to Fill Tanks": scenEDtank,
                "Power Needed (W)": scenPwr,
                "Rate of Base Added to Seawater (molOH/hr)":scenNB,
                "Rate of Excess Acid Generated (molHCl/hr)":scenNA,
                "Volume of Base Added to Tanks (m3)":scenVolBase,
                "Volume of Excess Acid Produced (m3)":scenVolAcid, 
                "Mass of Alkaline Solid Used of Acid Disposal (g)":scenMadSolid, 
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
                "Average pH of Effluent": res.pH_avg,
                "Average DIC of Effluent (M)": res.dic_avg, 
                "Average TA of Effluent (M)": res.ta_avg,
                "Average Salinity of Effluent (ppt)": res.sal_avg,
                "Average Volume of Effluent (m3/yr)": res.volOAEbase_yr,
                "Min Total Power Need for OAE (W)": min(ranges.S3["pwrRanges"]),
                "Max Total Power Need for OAE (W)": max(ranges.S1["pwrRanges"]),
                "Min OAE Rate (molOH/hr)": min(ranges.S1["mol_OH"]),
                "Max OAE Rate (molOH/hr)": max(ranges.S1["mol_OH"]),
                "Base Added to Seawater Under 100% Max Power (molOH/yr)": res.mol_OH_yr_MaxPwr,
                "OAE Capacity Factor (%)": res.oae_capacity_factor,
                "Fraction of Time OAE is Performed (%)": res.overall_capacity_factor,
                "Max Tank Fill (m3)": max(res.OAE_outputs["tank_vol_b"]),
                "Max Tank Fill (%)": res.max_tank_fill_percent,
                "Min ED Power (W)": oae_config.P_ed1
                * oae_config.N_edMin,
                "Max ED Power (W)": oae_config.P_ed1
                * oae_config.N_edMax,
                "Min Pump Power (W)": ranges.pump_power_min,
                "Max Pump Power (W)": ranges.pump_power_max,
                "Min Intake Pump Flow Rate (m3/s)": ranges.pumps.pumpO.Q_min,
                "Max Intake Pump Flow Rate (m3/s)": ranges.pumps.pumpO.Q_max,
                "Average Moles of Excess Acid Generated (molHCl/yr)": res.mol_HCl_yr,
                "Average Volume of Excess Acid Produced (m3/yr)": res.volXSacid_yr,
                "pH of Excess Acid": res.pH_HCl_excess,
                "Average Mass of Alkaline Solid Used for Acid Disposal (g/yr)": res.m_adSolid_yr,
                "Ratio of Moles of Alkaline Solid to Acid for Neutralization": ranges.S1["alkaline_to_acid"][0],
                "Mass of Products Made (tonnes/yr)": res.M_rev_yr,
                "Value of Products Made ($/yr)": res.X_rev_yr,
                "Estimated CDR (0.8 mol CO2:NaOH) (tCO2/yr)": res.M_co2est,
                "Estimated Max CDR Scale (0.8 mol CO2:NaOH) (tCO2/yr)": res.M_co2cap,
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
            }
            biogeochemDF = pd.DataFrame(biogeochem_results)
            biogeochemDF.to_csv(
                save_paths[1] + "OAE_biogeochemResults.csv", index=False
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
        if oae_config.acid_disposal_method is not None:
            ax1.plot(time, res.OAE_outputs["alkaline_solid_added"][time]/10**6, "brown", linewidth=2.5)
            ax1.set_ylabel(f"Power (MW) & Effluent Alkaline \n Seawater (pH {res.pH_avg}) (m³/s) & Mass of \n {oae_config.acid_disposal_method} for AD (t/hr)", fontsize=labelsize)
        else:
            ax1.set_ylabel(f"Power (MW) & \n Effluent Alkaline Seawater (pH {res.pH_avg}) (m³/s)", fontsize=labelsize)
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
        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig(
                save_paths[0] + "OAE_Time-Dependent_Results.png", bbox_inches="tight"
            )

        return (ranges, res)
    
if __name__ == "__main__":
    test = OAEInputs()

    res1 = initialize_power_chemical_ranges(
        oae_config=OAEInputs(),
        pump_config=PumpInputs(),
        seawater_config=SeaWaterInputs()
    )

    # EXAMPLE: Sin function for power input
    days = 365
    exTime = np.zeros(24 * days)  # Example time in hours
    for i in range(len(exTime)):
        exTime[i] = i + 1
    maxPwr = 400 * 10**4 # W
    Amp = maxPwr/2
    periodT = 24 
    movUp = Amp
    movSide = -1*math.pi/2
    exPwr = np.zeros(len(exTime))
    for i in range(len(exTime)):
        exPwr[i] = Amp*math.sin(2*math.pi/periodT*exTime[i] + movSide) + movUp
        if int(exTime[i]/24) % 5 == 1:
            exPwr[i] = exPwr[i] * 0.1

    results = simulate_ocean_alkalinity_enhancement(
        ranges=res1,
        oae_config=OAEInputs(),
        seawater_config=SeaWaterInputs(),
        power_profile=exPwr,
        initial_tank_volume_m3=0,
    )

    run_ocean_alkalinity_enhancement_physics_model(
        power_profile_w=exPwr,
        initial_tank_volume_m3=0,
        oae_config=OAEInputs(),
        pump_config=PumpInputs(),
        seawater_config=SeaWaterInputs(),
        save_plots=True,
        show_plots=True,
        save_outputs=True,

    )