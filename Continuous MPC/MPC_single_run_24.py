import pandas as pd
from pathlib import Path
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import sys
import json
from dataclasses import dataclass
from Load_forecast_specific_24 import Load_Forecaster_Params, Load_Forecast_Provider
from PV_forecast import PV_Forecast, PV_Forecast_Params
from apscheduler.schedulers.blocking import BlockingScheduler
import os
import time
import matplotlib.pyplot as plt
from termcolor import colored
import random

"""
Prices still have to be arranged! This will only run in afternoon xd
Still have to arrange warm start



MOET NU EEN MAKEN DIE JE GEWOON OVER VERLEDEN KAN LATEN RUNNEN EN KIJKEN VOOR DE IMPACT VAN KORTE HORIZON.



"""

@dataclass
class Battery_data:
    Vmax: float = 11.4 # kWh 
    eta: float = 0.95
    cmax: float = 10.0 # 10 kW Charge
    dmax: float = 10.0 # 10.0 kW Discharge
    beta_min: float = 0.2
    beta_max: float = 0.95
    beta0_0: float = 0.2
    c_deg: float = 1e-3 ### This whole implementation could be replaced by degradation model.


@dataclass
class Contract_data:
    inkoopvergoeding: float = 0.02 ### Zonneplan
    verkoopvergoeding: float = -0.02
    import_belasting: float = 0.111 ### Pery kWh
    jaarlijkske_belastingvermindering : float = 629.0 ### Per year




def ts_for_load_forecast(params):
    ts_now = pd.Timestamp.now(tz="Europe/Brussels")
    ts_last_bin = ts_now.floor(params.period)
    time_diff = ts_now - ts_last_bin
    if time_diff < pd.Timedelta(minutes=10): ### Should be set to two or something
        ts_to_forecast = ts_last_bin - pd.Timedelta(params.period)
        return ts_to_forecast, ts_now
    else:
        print(f"Current time {ts_now} is not within 2 minutes of the last bin {ts_last_bin}. Not forecasting now.")
        return None, ts_now


class Entsoe_Reader():
    def __init__(self, entsoe_path): ### File auto-updated by background process that retrieves DA prices every day and stores them to Data/DA Prices/DA_prices_cache.parquet. Updated every day at 12:55 and checks every 30 seconds until updated.
        self.entsoe_path = entsoe_path

    

    def read_prices(self):
        return pd.read_parquet(self.entsoe_path, columns=["DA_Price"])["DA_Price"] ### Has to return as serie, otherwise error.
    



class MPC:
    def __init__(self, battery_data: Battery_data, contract_data: Contract_data, params_load: Load_Forecaster_Params, PV_curtailment: bool, beta_0: float, salderen: bool):
        self.battery_data = battery_data
        self.contract_data = contract_data
        self.params_load = params_load
        self.PV_curtailment = PV_curtailment
        self.beta_0 = beta_0
        self.salderen = salderen




    def build_MPC(self, load_forecast, pv_forecast, entsoe_prices):
        m = pyo.ConcreteModel()
        m.T = pyo.Set(initialize=list(load_forecast.index), ordered=True)
        
        ### Parameters
        m.load_serie = pyo.Param(m.T, initialize=load_forecast.to_dict(), within=pyo.NonNegativeReals) ### In kWh
        m.pv_serie = pyo.Param(m.T, initialize=pv_forecast.to_dict(), within=pyo.NonNegativeReals) ### In kWh
        m.price_serie = pyo.Param(m.T, initialize=entsoe_prices.to_dict(), within=pyo.Reals) ### In €/kWh.

        m.Vmax = pyo.Param(initialize=self.battery_data.Vmax)
        m.eta = pyo.Param(initialize=self.battery_data.eta)
        m.cmax = pyo.Param(initialize=self.battery_data.cmax)
        m.dmax = pyo.Param(initialize=self.battery_data.dmax)
        m.beta_min = pyo.Param(initialize=self.battery_data.beta_min)
        m.beta_max = pyo.Param(initialize=self.battery_data.beta_max)
        m.beta0 = pyo.Param(initialize=self.beta_0) ### Has to be changed at every time!!
        m.c_deg = pyo.Param(initialize=self.battery_data.c_deg)
        m.inkoopvergoeding = pyo.Param(initialize=self.contract_data.inkoopvergoeding)
        m.verkoopvergoeding = pyo.Param(initialize=self.contract_data.verkoopvergoeding)
        m.import_belasting = pyo.Param(initialize=self.contract_data.import_belasting)
        m.jaarlijkske_belastingvermindering = pyo.Param(initialize=self.contract_data.jaarlijkske_belastingvermindering)

        
        m.period_s = pyo.Param(initialize=pd.Timedelta(self.params_load.period).total_seconds())
        m.max_grid_import = pyo.Param(initialize=10.0 * (pd.Timedelta(self.params_load.period).total_seconds() / 3600))
        m.max_grid_export = pyo.Param(initialize=10.0 * (pd.Timedelta(self.params_load.period).total_seconds() / 3600))
        ratio = len(load_forecast.index) * pd.Timedelta(self.params_load.period).total_seconds() / (365 * 24 * 3600) ### Make sure to count last interval too.
        m.ratio_interval_to_year = pyo.Param(initialize=ratio)

        ### Vars
        m.grid_import = pyo.Var(m.T, within=pyo.NonNegativeReals) ### In kWh!!
        m.grid_export = pyo.Var(m.T, within=pyo.NonNegativeReals) ### In kWh!!
        m.Pcharge = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, m.cmax)) ### Note that this is in kW!
        m.Pdischarge = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, m.dmax)) ### Note that this is in kW!
        m.SOC = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(m.beta_min, m.beta_max))
        m.PV_used = pyo.Var(m.T, within=pyo.NonNegativeReals)

        ### Binaries
        m.charge_indicator = pyo.Var(m.T, within=pyo.Binary)
        m.exclusive_grid = pyo.Var(m.T, within=pyo.Binary)


        ### Constraints
        def energy_balance_rule(m, t):
            #dt = (t - m.T.prev(t)).total_seconds() / 3600 ### Convert to hours
            dt =  m.period_s / 3600 ### Use this
            return m.grid_import[t] - m.grid_export[t] + (m.Pdischarge[t] * dt) - (m.Pcharge[t] * dt) + m.PV_used[t] - m.load_serie[t] == 0
        m.energy_balance = pyo.Constraint(m.T, rule=energy_balance_rule)

        def grid_exclusive_import(m,t):
            return m.grid_import[t] <= m.max_grid_import * m.exclusive_grid[t]
        m.grid_exclusive_import_constr = pyo.Constraint(m.T, rule=grid_exclusive_import)

        def grid_exclusive_export(m,t):
            return m.grid_export[t] <= m.max_grid_export * (1 - m.exclusive_grid[t])
        m.grid_exclusive_export_constr = pyo.Constraint(m.T, rule=grid_exclusive_export)

        def charge_exclusive(m,t):
            return m.Pcharge[t] <= m.cmax * m.charge_indicator[t]
        m.charge_exclusive_constr = pyo.Constraint(m.T, rule=charge_exclusive)

        def discharge_exclusive(m,t):
            return m.Pdischarge[t] <= m.dmax * (1 - m.charge_indicator[t])
        m.discharge_exclusive_constr = pyo.Constraint(m.T, rule=discharge_exclusive)


        def SOC_update_rule(m, t):
            if t == m.T.first():
                return m.SOC[m.T.first()] == m.beta0
            else: 
                #dt = (t - m.T.prev(t)).total_seconds() / 3600 ### Convert to hours
                dt = m.period_s / 3600 ### Use this
                prev_t = m.T.prev(t)
                return m.SOC[t] == m.SOC[prev_t] + (m.Pcharge[prev_t] * m.eta - m.Pdischarge[prev_t] / m.eta) * dt / m.Vmax ### SOC is the soc of the battery at the beginning of the interval.
        m.SOC_update_constr = pyo.Constraint(m.T, rule=SOC_update_rule)

        def SOC_terminal(m):
            last_T = m.T.last()
            dt = m.period_s / 3600 ### Convert to hours, have to use it here
            soc_plus_one = m.SOC[last_T] + (m.Pcharge[last_T] * m.eta - m.Pdischarge[last_T] / m.eta) * dt / m.Vmax
            return pyo.inequality(m.beta_min, soc_plus_one, m.beta_max)
        m.SOC_terminal_constr = pyo.Constraint(rule=SOC_terminal)


        def PV_curtailment_rule(m, t):
            if self.PV_curtailment:
                return pyo.inequality(0, m.PV_used[t], m.pv_serie[t])
            else:
                return m.PV_used[t] == m.pv_serie[t]
        m.PV_curtailment_constr = pyo.Constraint(m.T, rule=PV_curtailment_rule)

        ### Objective
        def OBJ(m):
            total = 0
            dt = m.period_s / 3600 ### Convert to hours, have to use it here
            for t in m.T:
                #dt = (t - m.T.prev(t)).total_seconds() / 3600 ### Convert to hours Probably not needed as grid import/export already in kWh

                if not self.salderen:
                    total += ((m.price_serie[t] + m.inkoopvergoeding + m.import_belasting) * m.grid_import[t]) - ((m.price_serie[t] + m.verkoopvergoeding) * m.grid_export[t]) + (m.c_deg * (m.Pcharge[t] + m.Pdischarge[t]) * dt)
                else:
                    total += ((m.price_serie[t] + m.inkoopvergoeding) * m.grid_import[t]) - ((m.price_serie[t] + m.verkoopvergoeding) * m.grid_export[t]) + (m.c_deg * (m.Pcharge[t] + m.Pdischarge[t]) * dt)
                
            if not self.salderen:
                total -= m.jaarlijkske_belastingvermindering * m.ratio_interval_to_year
                return total
            else: 
                tax_term = m.import_belasting * sum(m.grid_import[t] - m.grid_export[t] for t in m.T) ### Import and export already in kWh
                return total - (m.jaarlijkske_belastingvermindering * m.ratio_interval_to_year) + tax_term
        m.OBJ = pyo.Objective(rule=OBJ, sense=pyo.minimize)

        return m


    def solve_MPC(self, model):
        solver = SolverFactory("gurobi")
        solver.options["Threads"] = max(1, os.cpu_count() - 1) ### Use all but one core to avoid freezing the system. Adjust as needed.
        results = solver.solve(model, load_solutions=True, tee=False)
        #return results


            



############################


def SOC_retrieve(file_path, ts_new, disturb):
    if not file_path.exists():
        print(colored(f"SOC data file not found at {file_path}. Returning default value of 0.2 (empty).", "red"))
        return 0.2
    df = pd.read_parquet(file_path)
    ts_to_retrieve = ts_new.floor("15min")
    if ts_to_retrieve in df.index:
        if not disturb:
            return df.loc[ts_to_retrieve, "SOC"]
        else:
            disturbance = random.gauss(0, 0.02)
            disturbed_soc = df.loc[ts_to_retrieve, "SOC"] + disturbance
            return min(max(disturbed_soc, 0.2), 0.95) ### Chat for stability.
    else:
        print(colored(f"Timestamp {ts_to_retrieve} not found in SOC data.\n Will use default value of 0.2 (empty)", "red"))
        return 0.2

    



        



def MPC_run(config_file, rf_model_folder, correction, Entsoe_cache_path, logging):
    ### Load
    PV_column = "AC_active_power" ### Define here in order to avoid memory leak.
    meter_column = "HW_active_power_w"
    MODEL_SAVE_FOLDER_LOAD = Path(Path(__file__).parent / "Models" / "Load")
    params_load = Load_Forecaster_Params( ### Rest defined in load forecast file, should load this from config file later such that when retrained these can be changed. Or make an externally managed parameter here or something.
        start_training="2026-01-10",
        end_training="2026-03-30",
        start_eval="2026-03-30",
        end_eval="2026-04-02",
        start_test=None,#"2026-03-25",
        end_test=None,#"2026-04-03",
        OUTPUT_DIR=MODEL_SAVE_FOLDER_LOAD,
    )
    provider_load = Load_Forecast_Provider(params_load)

    ts_to_forecast, ts_now = ts_for_load_forecast(params_load)
    if not isinstance(ts_to_forecast, pd.Timestamp):
        return
    output_folder = Path(Path(__file__).parent / "MPC_outputs" / ts_now.strftime("%Y-%m-%d_%H-%M-%S"))
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if isinstance(ts_to_forecast, pd.Timestamp):
        print(f"Current time {pd.Timestamp.now(tz='Europe/Brussels')} is within 2 minutes of the last bin. Forecasting from previous bin {ts_to_forecast}.\n")

        times = provider_load.training__pipeline(FILE_rt_all, json_folder, meter_column, PV_column, index_column="Timestamp_package") ### Optional, to see timing info.
        load_forecast = provider_load.predict_from_timestamp(ts_to_forecast)

    ### PV
    with open(config_file, "r", encoding="utf-8") as f:
        params_config = json.load(f)
    params_solar = PV_Forecast_Params(
        house_id=params_config["house_id"],
        latitude=params_config["LAT"],
        longitude=params_config["LON"],
        rf_model_folder=rf_model_folder,
    )
    pv_forecast = PV_Forecast(params_solar)
    pv_forecast.run(correction)
    forecasted_energy_15min = pv_forecast.forecasted_energy["forecast_corrected"]

    ### ENTSOE
    provider_entsoe = Entsoe_Reader(Entsoe_cache_path)
    entsoe_prices = provider_entsoe.read_prices() ### Maybe still make a parameter here for FILL=True

    ### Print all ranges
    print(f"Load forecast range: {load_forecast.index.min()} to {load_forecast.index.max()}")
    print(f"PV forecast range: {forecasted_energy_15min.index.min()} to {forecasted_energy_15min.index.max()}")
    print(f"ENTSOE prices range: {entsoe_prices.index.min()} to {entsoe_prices.index.max()}")


    ### Cut to right series, based on load
    
    ref_index = load_forecast.index
    pv_forecast_cut = forecasted_energy_15min.reindex(ref_index)
    entsoe_prices_cut = entsoe_prices.reindex(ref_index)
    if pv_forecast_cut.isna().any() or entsoe_prices_cut.isna().any():
        raise ValueError("PV or price series missing timestamps from load_forecast index")

    ### Build MPC
    carryover_file = Path(__file__).parent / "Data" / "SOC_carryover.parquet"
    carryover_file.parent.mkdir(parents=True, exist_ok=True)
    beta_0_retrieved = SOC_retrieve(carryover_file, ts_now, disturb=True) ######## This will of course not occur if there is a real battery.
    ### Make separate file with soc data
    df_to_add = pd.DataFrame({"SOC": [beta_0_retrieved]}, index=[ts_now.floor("15min")])
    df_to_add.index.name = "Timestamp"
    soc_file_path = Path(Path(__file__).parent / "Data" / "Rolling" / "SOC_data.parquet")
    soc_file_path.parent.mkdir(parents=True, exist_ok=True)
    if soc_file_path.exists():
        df_existing = pd.read_parquet(soc_file_path)
        df_updated = pd.concat([df_existing, df_to_add])
        df_updated.to_parquet(soc_file_path)
    else:
        df_to_add.to_parquet(soc_file_path)



    salderen = False
    mpc = MPC(battery_data=Battery_data(), contract_data=Contract_data(), params_load=params_load, PV_curtailment=True, beta_0=beta_0_retrieved, salderen=salderen)
    start_build = time.time()
    m = mpc.build_MPC(load_forecast, pv_forecast_cut, entsoe_prices_cut)
    end_build = time.time()
    print(f"Time taken to build MPC model: {end_build - start_build:.2f} seconds")
    start_solve = time.time()
    #results = mpc.solve_MPC(m)
    mpc.solve_MPC(m)
    end_solve = time.time()
    print(f"Time taken to solve MPC model: {end_solve - start_solve:.2f} seconds")
    ### Arrange carryover
    first_t = m.T.first()
    next_t = m.T.next(first_t)
    beta_0_next = pyo.value(m.SOC[next_t])

    df_next = pd.DataFrame({"SOC": [beta_0_next]}, index=[next_t]) ### Overwritten each time.
    df_next.index.name = "Timestamp"
    df_next.to_parquet(carryover_file)


    SOC_serie = pd.Series({t: pyo.value(m.SOC[t]) for t in m.T})
    Charge_serie = pd.Series({t: pyo.value(m.Pcharge[t]) for t in m.T}) ### In kW
    Discharge_serie = pd.Series({t: pyo.value(m.Pdischarge[t]) for t in m.T}) ### In kW
    Grid_import_serie = pd.Series({t: pyo.value(m.grid_import[t]) for t in m.T}) ### In kWh
    Grid_export_serie = pd.Series({t: pyo.value(m.grid_export[t]) for t in m.T}) ### In kWh
    PV_used_serie = pd.Series({t: pyo.value(m.PV_used[t]) for t in m.T}) ### In kWh

    if logging:
        #output_folder = Path(Path(__file__).parent / "MPC_outputs" / ts_now.strftime("%Y-%m-%d_%H-%M-%S"))
        #output_folder.mkdir(parents=True, exist_ok=True)

        ### Put all series in a dataframe and to parquet
        df = pd.DataFrame({
            "SOC": SOC_serie,
            "Charge_kW": Charge_serie,
            "Discharge_kW": Discharge_serie,
            "Grid_import_kWh": Grid_import_serie,
            "Grid_export_kWh": Grid_export_serie,
            "PV_used_kWh": PV_used_serie,
            "Load_kWh": load_forecast,
            "PV_forecast_kWh": pv_forecast_cut,
            "DA_price_EUR_per_kWh": entsoe_prices_cut  
        })
        df.to_parquet(output_folder / "MPC_solution.parquet")
    print(f"\n It is now {pd.Timestamp.now(tz='Europe/Brussels')}. SOC Serie solved by MPC:\n{SOC_serie}\n")

    







if __name__ == "__main__":

    #######################################
    ############ Load forecast ############
    #######################################
    if sys.platform == "darwin":
        FILE_rt_all = Path("/Users/brechtgoethals/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/inverter_logs_parquet/rt_all_2.parquet")
        json_folder = Path("/Users/brechtgoethals/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/inverter_logs") #  None
    ### if on linux:
    elif sys.platform == "linux":
        FILE_rt_all = Path("/home/brecht1949/inverter_logs_parquet/rt_all_2.parquet")
        json_folder = Path("/home/brecht1949/inverter_logs/")
    elif sys.platform == "win32":
        print("Windows not supported yet.")


    
    




    #######################################
    ############ ENTSOE Reader ############
    #######################################
    Entsoe_cache_path = Path(__file__).parent / "Data" / "DA Prices" / "DA_prices_cache_extended.parquet"
    if not Entsoe_cache_path.exists():
        Entsoe_cache_path = Path(__file__).parent / "Data" / "DA Prices" / "DA_prices_cache.parquet"
    


    

    #######################################
    ############# PV forecast #############.  ### NOTE: PV has to be reloaded manually. In load, this is already done automatically in the training pipeline.
    #######################################
    config_file = Path(__file__).parent / "config" / "config.json"
    RF_model_folder = Path(Path(__file__).parent.parent / "PV forecasting" / "Together" / "Model Save")
    CORRECTION = True



    """
    #### Loop
    LOGGING = True
    scheduler = BlockingScheduler(timezone="Europe/Brussels")
    scheduler.add_job(MPC_run, "cron", minute="0,15,30,45", second=5, id="MPC_job", kwargs={
        "config_file": config_file,
        "rf_model_folder": RF_model_folder,
        "correction": CORRECTION,
        "provider_entsoe": entsoe_reader,
        "logging": LOGGING
    })
    scheduler.start()
    """

    ### Manual call
    MPC_run(config_file, RF_model_folder, CORRECTION, Entsoe_cache_path, logging=True)