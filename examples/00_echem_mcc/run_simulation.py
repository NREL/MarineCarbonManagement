import numpy_financial as npf
import ProFAST
import numpy as np
from hopp.simulation import HoppInterface
from hopp.utilities import load_yaml
from hopp.utilities.keys import set_nrel_key_dot_env
from mcm import echem_mcc


# ----------------------------------- Hybrid Power Simulation ----------------
# Set API key using the .env for HOPP simulation
set_nrel_key_dot_env()
hopp_config = load_yaml("./inputs/hopp_config.yaml")

#load floris config for wind technology simulation
floris_config = load_yaml("./inputs/floris/floris_input_osw_15MW.yaml")
# load floris config into hopp config
hopp_config["technologies"]["wind"]["floris_config"] = floris_config

# Add load profile. Minimum power required to power the DOC scenarios.
baseload_limit_kw = float(6885.25838478)
desired_load = baseload_limit_kw*np.ones((8760))/1000
hopp_config['site']['desired_schedule'] = desired_load #MW

# Set OM costs in HOPP and financial model
if ("wind_om_per_kw" in hopp_config["config"]["cost_info"]) and (
            hopp_config["technologies"]["wind"]["fin_model"]["system_costs"][
                "om_capacity"
            ][0]
            != hopp_config["config"]["cost_info"]["wind_om_per_kw"]
        ):

            for i in range(
                len(
                    hopp_config["technologies"]["wind"]["fin_model"][
                        "system_costs"
                    ]["om_capacity"]
                )
            ):
                hopp_config["technologies"]["wind"]["fin_model"]["system_costs"][
                    "om_fixed"
                ][i] = hopp_config["config"]["cost_info"]["wind_om_per_kw"]


if ("wind_om_per_mwh" in hopp_config["config"]["cost_info"]) and (
            hopp_config["technologies"]["wind"]["fin_model"]["system_costs"][
                "om_production"
            ][0]
            != hopp_config["config"]["cost_info"]["wind_om_per_mwh"]
        ):
            # Use this to set the Production-based O&M amount [$/MWh]
            for i in range(
                len(
                    hopp_config["technologies"]["wind"]["fin_model"][
                        "system_costs"
                    ]["om_production"]
                )
            ):
                hopp_config["technologies"]["wind"]["fin_model"]["system_costs"][
                    "om_production"
                ][i] = hopp_config["config"]["cost_info"]["wind_om_per_mwh"]

if ("battery_om_per_kw" in hopp_config["config"]["cost_info"]) and (
            hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
                "om_capacity"
            ][0]
            != hopp_config["config"]["cost_info"]["battery_om_per_kw"]
        ):

            for i in range(
                len(
                    hopp_config["technologies"]["battery"]["fin_model"][
                        "system_costs"
                    ]["om_capacity"]
                )
            ):
                hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
                    "om_fixed"
                ][i] = hopp_config["config"]["cost_info"]["battery_om_per_kw"]


if ("battery_om_per_mwh" in hopp_config["config"]["cost_info"]) and (
            hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
                "om_production"
            ][0]
            != hopp_config["config"]["cost_info"]["battery_om_per_mwh"]
        ):
            # Use this to set the Production-based O&M amount [$/MWh]
            for i in range(
                len(
                    hopp_config["technologies"]["battery"]["fin_model"][
                        "system_costs"
                    ]["om_production"]
                )
            ):
                hopp_config["technologies"]["battery"]["fin_model"]["system_costs"][
                    "om_production"
                ][i] = hopp_config["config"]["cost_info"]["battery_om_per_mwh"]



# Instantiate HOPP Interface for Hybrid Simulation
hi = HoppInterface(hopp_config)

# Wave Energy Converter (WEC) Cost Inputs
cost_model_inputs = {
	'reference_model_num':3,
	'water_depth': 482,
	'distance_to_shore': 50,
	'number_rows': 18, #Adjust for analysis
	'device_spacing':600,
	'row_spacing': 600,
	'cable_system_overbuild': 20
}
hi.system.wave.create_mhk_cost_calculator(cost_model_inputs)

# Simulate for 30 years
hi.simulate(30)

wind_speed = [W[2] for W in hi.system.site.wind_resource._data["data"]]
print('Wind speed (m/s)', np.round(np.average(wind_speed),decimals=4))

hybrid_plant = hi.system

hybrid_energy = hi.system.grid._system_model.Outputs.system_pre_interconnect_kwac[0:8760]
hybrid_energy2 = hi.system.grid._system_model.Outputs.gen[0:8760]

# WEC Capex 
cost_dict = hybrid_plant.wave.mhk_costs.cost_outputs
wcapex = (
    cost_dict["structural_assembly_cost_modeled"]
    + cost_dict["power_takeoff_system_cost_modeled"]
    + cost_dict["mooring_found_substruc_cost_modeled"]
)

# WEC Balance of System (BOS)
wbos = (
    cost_dict["development_cost_modeled"]
    + cost_dict["eng_and_mgmt_cost_modeled"]
    + cost_dict["plant_commissioning_cost_modeled"]
    + cost_dict["site_access_port_staging_cost_modeled"]
    + cost_dict["assembly_and_install_cost_modeled"]
    + cost_dict["other_infrastructure_cost_modeled"]
)

# WEC Electrical Infrastructure
welec_infrastruc_costs = (
    cost_dict["array_cable_system_cost_modeled"]
    + cost_dict["export_cable_system_cost_modeled"]
    + cost_dict["other_elec_infra_cost_modeled"]
)

gen_inflation = 0.025 # General Inflation

# Adjust Hybrid Energy System Costs to 2023 cost year
wind_capex = hybrid_plant.wind.total_installed_cost #$2022
wind_capex = -npf.fv(
            gen_inflation,
            1,
            0.0,
            wind_capex,
        ) # Adjust to $2023
print("wind capex", wind_capex)

wave_capex = wcapex + wbos + welec_infrastruc_costs #$2020
wave_capex = -npf.fv(
            gen_inflation,
            3,
            0.0,
            wave_capex,
        ) # Adjust to $2023
print("wave capex", wave_capex)

if 'battery' in hopp_config['technologies']:
       battery_capex = hybrid_plant.battery.total_installed_cost #$2022\
       battery_capex = -npf.fv(
            gen_inflation,
            1,
            0.0,
            battery_capex,
        ) # Adjust to $2023
else:
    battery_capex = 0
print("battery capex", battery_capex)

#Opex
wind_opex = hybrid_plant.wind.om_total_expense[0] #$2022
wind_opex = -npf.fv(
            gen_inflation,
            1,
            0.0,
            wind_opex,
        ) # Adjust to $2023
print("wind opex",wind_opex)

wave_opex = cost_dict["maintenance_cost"] + cost_dict["operations_cost"] #$2020
wave_opex = -npf.fv(
            gen_inflation,
            3,
            0.0,
            wave_opex,
        ) # Adjust to $2023
print("wave opex",wave_opex)

if 'battery' in hopp_config['technologies']:
       battery_opex = hybrid_plant.battery.om_total_expense[0] #$2022\
       battery_opex = -npf.fv(
            gen_inflation,
            1,
            0.0,
            battery_opex,
        ) # Adjust to $2023
else:
    battery_opex = 0
print("battery opex", battery_opex)

# ----------------------------------- Hybrid Financial Calculations ----------------
pf = ProFAST.ProFAST()
pf.set_params(
        "commodity",
        {
            "name": "electricity",
            "unit": "kWh",
            "initial price": 100,
            "escalation": gen_inflation,
        },
    )

pf.set_params("capacity",np.sum(hybrid_energy) / 365.0,)  # kWh/day
pf.set_params("maintenance", {"value": 0, "escalation": gen_inflation})
pf.set_params("analysis start year", 2031)
pf.set_params("operating life", 30)
pf.set_params("installation months", 36)
pf.set_params("installation cost",
        {
            "value": 0,
            "depr type": "Straight line",
            "depr period": 4,
            "depreciable": False,
        },
    )
pf.set_params("demand rampup", 0)
pf.set_params("long term utilization", 1)
pf.set_params("credit card fees", 0)
pf.set_params("sales tax", 0)
pf.set_params("license and permit", {"value": 00, "escalation": gen_inflation})
pf.set_params("rent", {"value": 0, "escalation": gen_inflation})
pf.set_params("property tax and insurance",0.015,)
pf.set_params("admin expense",0.00,)
pf.set_params("total income tax rate",0.2574,)
pf.set_params("capital gains tax rate",0.099,)
pf.set_params("sell undepreciated cap", True)
pf.set_params("tax losses monetized", True)
pf.set_params("general inflation rate", gen_inflation)
pf.set_params("leverage after tax nominal discount rate",0.105,)
pf.set_params('debt equity ratio of initial financing',1.5) # D2E ratio at start
pf.set_params("debt type", "Revolving debt")
pf.set_params("loan period if used", 0)
pf.set_params("debt interest rate",0.05)
pf.set_params("cash onhand", 1)

    # ----------------------------------- Add capital items to ProFAST ----------------
pf.add_capital_item(
    name="Wind System",
    cost=wind_capex,
    depr_type="MACRS",
    depr_period=5,
    refurb=[0],
)
pf.add_capital_item(
    name="Wave System",
    cost=wave_capex,
    depr_type="MACRS",
    depr_period=5,
    refurb=[0],
)

if 'battery' in hopp_config['technologies']:
    pf.add_capital_item(
        name="Battery System",
        cost=battery_capex,
        depr_type="MACRS",
        depr_period=5,
        refurb=[0],
    )

    # -------------------------------------- Add fixed costs--------------------------------
pf.add_fixed_cost(
    name="Wind O&M Cost",
    usage=1.0,
    unit="$/year",
    cost=wind_opex,
    escalation=gen_inflation,
)

pf.add_fixed_cost(
    name="Wave O&M Cost",
    usage=1.0,
    unit="$/year",
    cost=wave_opex,
    escalation=gen_inflation,
)

if 'battery' in hopp_config['technologies']:
    pf.add_fixed_cost(
        name="Battery O&M Cost",
        usage=1.0,
        unit="$/year",
        cost=battery_opex,
        escalation=gen_inflation,
    )

sol = pf.solve_price()


# -------------------------------------- Hybrid Energy System Results --------------------------------
lcoe = sol["price"] # Levelized Cost of Energy
print("\nProFAST LCOE: ", "%.2f" % (lcoe * 1e3), "$/MWh")


aeps = hybrid_plant.annual_energies
npvs = hybrid_plant.net_present_values
cf = hybrid_plant.capacity_factors

print("Annual Energy Production")
print(aeps) # Hybrid Energy System Annual Energy Production

print("Capacity Factors") 
print(cf) # Hybrid Energy System Capacity Factors

print(hybrid_plant.capacity_factors.grid)


# ------------------ Run Marince Carbon Capture Electrodialysis Simulation --------------------------------
co_2_outputs, range_outputs, ed_outputs = echem_mcc.run_electrodialysis_physics_model(
        power_profile_w=np.array(hybrid_energy)*1000,
        initial_tank_volume_m3=0,
        electrodialysis_config=echem_mcc.ElectrodialysisInputs(),
        pump_config=echem_mcc.PumpInputs(),
        seawater_config=echem_mcc.SeaWaterInputs(
                                                sal=33,
                                                tempC=12
                                                    ),
        save_outputs=True,
        save_plots=True,
        plot_range=[3910, 4030],
)


# -------------------------------------- Physics Results for mCC --------------------------------
print("mCC per year", ed_outputs.mCC_total)
print("eneryg cap fac", ed_outputs.energy_capacity_factor)
print("power ranges")
# print("s1",range_outputs.S1['pwrRanges'])
# print("s2",range_outputs.S2['pwrRanges'])
# print("s3",range_outputs.S3['pwrRanges'])
# print("s4",range_outputs.S4['pwrRanges'])
print("min power scenario", min(np.concatenate([range_outputs.S1['pwrRanges'], range_outputs.S2['pwrRanges'],range_outputs.S3['pwrRanges'],range_outputs.S4['pwrRanges']])))
print("max power scenario", max(np.concatenate([range_outputs.S1['pwrRanges'], range_outputs.S2['pwrRanges'],range_outputs.S3['pwrRanges'],range_outputs.S4['pwrRanges']])))

print("min s1/s2 for wv power scenario", min(np.concatenate([range_outputs.S1['pwrRanges'], range_outputs.S2['pwrRanges']])))


ed_costs = echem_mcc.electrodialysis_cost_model(
    echem_mcc.ElectrodialysisCostInputs(
        electrodialysis_inputs=echem_mcc.ElectrodialysisInputs(),
        mCC_yr=ed_outputs.mCC_yr,
        total_tank_volume=range_outputs.V_aT_max + range_outputs.V_bT_max,
        infrastructure_type="new",
        max_theoretical_mCC=max(range_outputs.S1["mCC"]),
    ),
    save_outputs=True,
    
)


# -------------------------------------- Cost Results for mCC --------------------------------

print(ed_costs)

total_year_cost = ed_costs.yearly_capital_cost + ed_costs.yearly_operational_cost + lcoe*np.sum(hybrid_energy)

cost_per_ton_co2 = total_year_cost / ed_outputs.mCC_total

cost_without_e = (ed_costs.yearly_capital_cost + ed_costs.yearly_operational_cost)/ed_outputs.mCC_total

print("Cost per tonne CO2", cost_per_ton_co2)
print("Costs without E", cost_without_e)