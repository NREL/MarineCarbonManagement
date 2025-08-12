# Constants
RHO = 1030  # kg/m³
MM_NACL = 58.44  # g/mol

# Convert salinity from molarity (M) to parts per thousand (ppt)
def sal_m_to_ppt(sal_m):
    return sal_m * 1000 / RHO * MM_NACL

# Convert salinity from parts per thousand (ppt) to molarity (M)
def sal_ppt_to_m(sal_ppt):
    return sal_ppt / MM_NACL * RHO / 1000

# Convert concentration from molarity (M) to micromoles per kilogram (µmol/kg)
def m_to_umol_per_kg(x_m):
    return x_m * 1e6 * 1000 / RHO

# Convert concentration from micromoles per kilogram (µmol/kg) to molarity (M)
def umol_per_kg_to_m(x_u):
    return x_u * RHO / 1000 / 1e6
