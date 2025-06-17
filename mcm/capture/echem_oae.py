"""Model of electrochemical mCC system"""

__author__ = "James Niffenegger, Kaitlin Brunik"
__copyright__ = "Copyright 2024, National Renewable Energy Laboratory"
__maintainer__ = "Kaitlin Brunik"
__email__ = ("james.niffenegger", "kaitlin.brunik@nrel.gov")

import math
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
R_H2O = 1030 # kg/m^3, density of water

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
    N_edMin: int = 1
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

if __name__ == "__main__":
    test = OAEInputs()