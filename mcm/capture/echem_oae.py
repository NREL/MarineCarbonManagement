"""Model of electrochemical mCC system"""

__author__ = "James Niffenegger, Kaitlin Brunik"
__copyright__ = "Copyright 2024, National Renewable Energy Laboratory"
__maintainer__ = "Kaitlin Brunik"
__email__ = ("james.niffenegger", "kaitlin.brunik@nrel.gov")

import math
import warnings
import numpy as np
import pandas as pd
import PyCO2SYS as pyco2
import matplotlib.pyplot as plt
import statistics as stat

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
    P_edMax: float = field(default=250e4, validator=gt_zero)
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
    dic_i: float = 2.2e-3
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
        p_vacCO2_min_bar (float): The minimum vacuum pressure (in bar). Default is 0.1.
        p_vacCO2_max_bar (float): The maximum vacuum pressure (in bar). Default is 0.2.
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
    p_vacCO2_min_bar: float = 0.4
    p_vacCO2_max_bar: float = 0.8

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
                f"{self.__class__.__name__}: Flow Rate is {(self.Q_min - Q) / self.Q_min * 100:.2f}% less than the range provided for pump power. Defaulting to minimum flow rate: {self.Q_min:.2f} (m³/s).",
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
    )  # features of seawater intake pump
    pumpED = Pump(
        Q_ed1 * N_edMin, 
        Q_ed1 * N_edMax, 
        p.p_ed_min_bar, 
        p.p_ed_max_bar, 
        p.y_pump
    )  # features of ED pump
    pumpA = Pump(
        pumpED.Q_min * oae_config.frac_acidFlow, 
        pumpED.Q_max * oae_config.frac_baseFlow,
        p.p_a_min_bar, 
        p.p_a_max_bar, 
        p.y_pump
    )  # features of acid pump
    pumpI = Pump(
        pumpED.Q_min * (1/oae_config.frac_EDflow - 1), 
        pumpED.Q_max * (1/oae_config.frac_EDflow - 1), 
        p.p_i_min_bar, 
        p.p_i_max_bar, 
        p.y_pump
    )  # features of pump for seawater acidification
    pumpB = Pump(
        pumpED.Q_min - pumpA.Q_min,
        pumpED.Q_max - pumpA.Q_max,
        p.p_b_min_bar,
        p.p_b_max_bar,
        p.y_pump,
    )  # features of base pump
    pumpF = Pump(
        pumpI.Q_min + pumpB.Q_min,
        pumpI.Q_max + pumpB.Q_max,
        p.p_f_min_bar,
        p.p_f_max_bar,
        p.y_pump,
    )  # features of seawater output pump (note min can be less if all acid and base are used)
    pumpED4 = Pump(
        Q_ed1 * N_edMin, 
        Q_ed1 * N_edMax, 
        p.p_o_min_bar, 
        p.p_o_max_bar, 
        p.y_pump
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
            (i + N_edMin) * Q_ed1 / 2
        )  # Flow rate for base pump based on equivalent ED units that would be active
        p.pumpA.Q = p.pumpA.Q # No waste acid is pumped in this case

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

if __name__ == "__main__":
    test = OAEInputs()

    res1 = initialize_power_chemical_ranges(
        oae_config=OAEInputs(),
        pump_config=PumpInputs(),
        seawater_config=SeaWaterInputs()
    )