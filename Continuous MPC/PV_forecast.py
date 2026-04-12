import pvlib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
import joblib
import json

@dataclass
class PV_Forecast_Params:
    house_id: str
    latitude: float
    longitude: float
    rf_model_folder: Path

    amount_panels: int = 19 ### Number of panels in the system
    tilt: float = 34.0 ### Degrees.  ### Get these from other script
    azimuth: float = 130.0 ### Degrees (0=N, 90=E, 180=S, 270=W)
    eff_STC: float = 0.166 ### Standard Test Conditions efficiency
    area: float = 1640 * 990 / 1e6 ### mm to m^2
    noct: float = 45 ### °C
    temp_coeff: float = -0.0042 ### Efficiency temperature coefficient %/°C
    transmittance_absorptance: float = 0.9 ### Necessary for pvlib model but can leave this to default
    array_height: int = 1 ### 2 for taller buildings, 1 for smaller buildings. (Amount of stories of approx 3m.)
    mount_standoff: int = 4 ### In inches, 1inch = 2.54cm
    inverter_efficiency_nominal: float = 0.971 ### Nominal efficiency of the inverter
    inverter_max_dc_power: float = 4.840
    fitting_factor: float = 0.8





class PV_Forecast():
    def __init__(self, params_solar: PV_Forecast_Params):
        self.params = params_solar
        self.timezone = "Europe/Brussels"

        self.weather = self.Weather_model(self)
        self.physics = self.Physics_model(self)
        self.rf = self.RF_correction_model(self)

        self.forecasted_energy = None

        


    class Weather_model():
        def __init__(self, outer):
            self.outer = outer
            self.weather_df = pd.read_parquet(Path(__file__).parent.parent / r"Weather fetcher" / "Weather_data" / "weather_df_cached.parquet") ### Read from parquet file, given by Weather_forecast_save.py, which is automatically updated by service.


    class Physics_model():
        def __init__(self, outer):
            self.outer = outer
            self.__dict__.update(asdict(outer.params))
            self.data_dict = {}

        def irradiance_poa(self):
            weather_fetcher = self.outer.weather
            if weather_fetcher.weather_df is None:
                raise RuntimeError("Weather data is not available. Please fetch weather data before calculating POA irradiance.")
            times = weather_fetcher.weather_df.index
            tz = times.tz if times.tz is not None else self.outer.timezone


            location = pvlib.location.Location(self.outer.params.latitude, self.outer.params.longitude, tz=tz)
            solar_position_array = location.get_solarposition(times)
        
            irradiance_array = pvlib.irradiance.get_total_irradiance(
                surface_tilt=self.tilt,
                surface_azimuth=self.azimuth,
                solar_zenith=solar_position_array["apparent_zenith"],
                solar_azimuth=solar_position_array["azimuth"],
                dni = weather_fetcher.weather_df["direct_normal_irradiance"],  ### Use interpolation.
                ghi = weather_fetcher.weather_df["shortwave_radiation"],
                dhi = weather_fetcher.weather_df["diffuse_radiation"],
                albedo=0.15, ### From previous assignments
                dni_extra=pvlib.irradiance.get_extra_radiation(times),
                model="haydavies",
            )
            self.data_dict["irradiance_array"] = irradiance_array
            self.data_dict["solar_position_array"] = solar_position_array
        

        def module_temperature_and_efficiency(self): ### Geometry of installation not taken into account.
            weather_fetcher = self.outer.weather
            if "irradiance_array" not in self.data_dict:
                raise RuntimeError("Irradiance data is not available. Please calculate POA irradiance before calculating module temperature and efficiency.")
            Temperature_array = pvlib.temperature.noct_sam(
                poa_global=self.data_dict["irradiance_array"]["poa_global"],
                temp_air=weather_fetcher.weather_df["temperature_2m"].reindex(self.data_dict["irradiance_array"].index).interpolate("time"),
                wind_speed=weather_fetcher.weather_df["wind_speed_10m"].reindex(self.data_dict["irradiance_array"].index).interpolate("time") / 3.6, ### Convert from km/h to m/s
                noct=self.noct,
                module_efficiency=self.eff_STC,
                transmittance_absorptance=self.transmittance_absorptance, ### Optional, default is 0.9 anyways but I'll leave it here for clarity.
                array_height=self.array_height,
                mount_standoff=self.mount_standoff
            )
            Efficiency_array = self.eff_STC * (1 + self.temp_coeff * (Temperature_array - 25))
            self.data_dict["Efficiency_array"] = Efficiency_array
        
        def DC_power_output(self):
            power_output = self.data_dict["irradiance_array"]["poa_global"] * self.area * self.data_dict["Efficiency_array"] * self.amount_panels /1000 ### In kW
            self.data_dict["DC_power_output"] = power_output
        
        def DC_energy_yield(self):
            power_output = self.data_dict["DC_power_output"]
            dt_s = (power_output.index[1] - power_output.index[0]).total_seconds()  # ~5
            energy_kwh_15min = power_output.resample("15min").sum() * dt_s / 3600
            self.data_dict["DC_energy_yield"] = energy_kwh_15min

        
        def inverter_power_output(self):
            self.data_dict["inverter_power_output"] = pvlib.inverter.pvwatts(pdc=self.data_dict["DC_power_output"], pdc0=self.inverter_max_dc_power, eta_inv_nom=self.inverter_efficiency_nominal)
        
        def inverter_energy_yield(self):
            inverter_power_output = self.data_dict["inverter_power_output"]
            dt_s = (inverter_power_output.index[1] - inverter_power_output.index[0]).total_seconds()  # ~5
            inverter_energy_yield = inverter_power_output.resample("15min").sum() * dt_s / 3600 ### In kWh, resampled to 15min intervals
            self.data_dict["inverter_energy_yield"] = inverter_energy_yield
        
        def fitting_factor_power(self):
            self.data_dict["fitting_factor_power"] = self.data_dict["inverter_power_output"] * self.fitting_factor
        
        def fitting_factor_energy(self):
            fitting_factor_power = self.data_dict["fitting_factor_power"]
            dt_s = (fitting_factor_power.index[1] - fitting_factor_power.index[0]).total_seconds()  # ~5
            fitting_factor_energy = fitting_factor_power.resample("15min").sum() * dt_s / 3600 ### In kWh, resampled to 15min intervals
            self.data_dict["fitting_factor_energy"] = fitting_factor_energy

        def run(self):
            try:
                self.irradiance_poa()
                self.module_temperature_and_efficiency()
                self.DC_power_output()
                self.DC_energy_yield()
                self.inverter_power_output()
                self.inverter_energy_yield()
                self.fitting_factor_power()
                self.fitting_factor_energy()
            except Exception as e:
                raise RuntimeError(f"Physics model failed:\n{e}")
            
        

    class RF_correction_model():
        def __init__(self, outer):
            self.outer = outer
            self.bundle = joblib.load(outer.params.rf_model_folder / f"rf_model_PV_{self.outer.params.house_id}.joblib")
            self.rf = self.bundle["model"]
            self.feature_cols = self.bundle["feature_cols"]
            self.target_col = self.bundle["target_col"]
            self.features_df = pd.DataFrame()
            self.predicted_df = None

        def build_features(self):
            irradiance_array_ = self.outer.physics.data_dict["irradiance_array"]
            solar_position_array_ = self.outer.physics.data_dict["solar_position_array"]

            df = pd.DataFrame(index=irradiance_array_.index)
            az = np.deg2rad(solar_position_array_["azimuth"])
            zen = np.deg2rad(solar_position_array_["apparent_zenith"])
            df["solar_azimuth_sin"] = np.sin(az)
            df["solar_azimuth_cos"] = np.cos(az)
            df["solar_zenith_sin"] = np.sin(zen)

            ### Irradiance features
            df = df.join(irradiance_array_[["poa_global", "poa_direct", "poa_diffuse"]], how="left")

            ### Time features
            day_of_year = df.index.dayofyear
            hour_of_day = df.index.hour
            minute_of_hour = df.index.minute

            minute_of_day = hour_of_day * 60 + minute_of_hour
            df["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
            df["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

            day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
            df["day_of_year_sin"] = day_of_year_sin
            df["day_of_year_cos"] = day_of_year_cos

            #indices_to_use = df[irradiance_array["poa_global"] > 10].index
            #df = df.loc[indices_to_use]
            self.features_df = df


        def predict_correction_no_lag(self):
            irradiance_array_ = self.outer.physics.data_dict["irradiance_array"]
            X = self.features_df[self.feature_cols]
            pred = self.rf.predict(X)
            self.predicted_df = pd.DataFrame(pred, index=self.features_df.index, columns=[self.target_col])
            indices_to_zero = irradiance_array_["poa_global"] <= 10
            self.predicted_df.loc[indices_to_zero, self.target_col] = 0
        
        def run(self):
            try:
                self.build_features()
                self.predict_correction_no_lag()
            except Exception as e:
                raise RuntimeError(f"RF correction model failed:\n{e}")

    ### Outer class

    def combine(self, CORRECTION_):
        if CORRECTION_:
            ff = self.physics.data_dict.get("fitting_factor_energy")
            if ff is None:
                raise RuntimeError("Physics model output is not available. Please run the physics model before combining.")

            if self.rf.predicted_df is None:
                raise RuntimeError("RF correction model output is not available. Please run the RF correction model before combining.")

            rf_err = self.rf.predicted_df[self.rf.target_col]

            # Align on common timestamps (prevents silent NaNs if indices differ)
            ff, rf_err = ff.align(rf_err, join="inner")

            self.forecasted_energy = (ff - rf_err).to_frame("forecast_corrected")
        else:
            ff = self.physics.data_dict.get("fitting_factor_energy")
            if ff is None:
                raise RuntimeError("Physics model output is not available. Please run the physics model before combining.")
            self.forecasted_energy = ff.to_frame("forecast_corrected")


    def run(self, CORRECTION):
        try:

            self.physics.run()
            self.rf.run()
        except Exception as e:
            raise RuntimeError(f"PV forecast failed:\n{e}")
        
        try:
            self.combine(CORRECTION_=CORRECTION)
        except Exception as e:
            raise RuntimeError(f"Combining model outputs failed:\n{e}")











if __name__ == "__main__":
    config_file = Path(__file__).parent / "Config" / "config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        params_config = json.load(f)

    RF_model_folder = Path(Path(__file__).parent.parent / "PV forecasting" / "Together" / "Model Save")

    params_solar = PV_Forecast_Params(
        house_id=params_config["house_id"],
        latitude=params_config["LAT"],
        longitude=params_config["LON"],
        rf_model_folder=RF_model_folder,
    )
    pv_forecast = PV_Forecast(params_solar)
    CORRECTION = True
    pv_forecast.run(CORRECTION)
    forecasted_energy_15min = pv_forecast.forecasted_energy["forecast_corrected"]
    print(forecasted_energy_15min.to_string())



    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(pv_forecast.physics.data_dict["fitting_factor_energy"], label="Physics model forecast (kWh)", color="blue")
    plt.plot(pv_forecast.rf.predicted_df[pv_forecast.rf.target_col], label="RF predicted error (kWh)", color="red")
    plt.plot(pv_forecast.forecasted_energy["forecast_corrected"], label="Corrected forecast (kWh)", color="green")
    plt.legend()
    plt.title("PV Energy Yield Forecast Correction")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.grid()
    plt.tight_layout()
    plt.close()




    ### Per day totals
    daily_corrected_energy = pv_forecast.forecasted_energy["forecast_corrected"].resample("D").sum()
    """print("\n=== DAILY CORRECTED ENERGY TOTALS ===")
    for date, energy in daily_corrected_energy.items():
        print(f"{date.date()}: {energy:.3f} kWh")"""

    ### From measured
    import rt_all
    provider = rt_all.rt_all_provider("/Users/brechtgoethals/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/inverter_logs_parquet/rt_all_2.parquet")
    start_date = "2026-03-25" ### Cannot use minute format here
    end_date = "2026-04-04"
    provider.load_data(start_date, end_date)
    daily_measured_energy = provider.calculate_daily_energy()
    print("\n=== DAILY ENERGY TOTALS ===")
    ### Compare measured vs forecasted per day
    ### Calculate WAPE and Average error
    Sum_error = 0
    Sum_absolute_error = 0
    Sum_measured = 0
    for date, energy in daily_corrected_energy.items():
        date_str = date.strftime("%Y-%m-%d")
        measured_energy = daily_measured_energy.get(date_str, None)
        if measured_energy is not None:
            Error = energy - measured_energy
            Sum_error += Error
            Absolute_error = abs(Error)
            Sum_absolute_error += Absolute_error
            Sum_measured += measured_energy
            print(f"{date_str}: Corrected forecast = {energy:.3f} kWh, Measured = {measured_energy:.3f} kWh: Error = {Error:.3f} kWh ({(Error)/measured_energy:.2%}%)")
        else:
            print(f"{date_str}: Corrected forecast = {energy:.3f} kWh, Measured = N/A")

    print(f"\nOver {len(daily_corrected_energy)} days:")
    print(f"Overall WAPE: {Sum_absolute_error / Sum_measured:.2%}")
    print(f"Overall Average Error: {Sum_absolute_error / len(daily_corrected_energy):.3f} kWh")
    print(f"Overall Average Measured: {Sum_measured / len(daily_corrected_energy):.3f} kWh")
    print(f"Sum of errors: {Sum_error:.3f} kWh")


    print(f"\nCorrection on" if CORRECTION else "\nCorrection off")


    measured_energy_15min = provider.df_period_energy_15min
    error_energy_15min = forecasted_energy_15min - measured_energy_15min

    plt.figure(figsize=(12,6))
    plt.plot(measured_energy_15min.index, measured_energy_15min.values, label="Measured energy (kWh)", color="orange", alpha=0.5, linestyle="--")
    plt.plot(forecasted_energy_15min.index, forecasted_energy_15min.values, label="Corrected forecast energy (kWh)", color="green", alpha=0.5, linestyle="--")
    plt.plot(error_energy_15min.index, error_energy_15min.values, label="Error (kWh)", color="red", alpha=0.7)
    plt.legend()
    plt.title("15-minute Energy Comparison")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.grid()
    plt.tight_layout()
    plt.show()
