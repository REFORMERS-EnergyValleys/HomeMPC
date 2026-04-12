import os

import pandas as pd
from pathlib import Path
import joblib
from dataclasses import dataclass, asdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import sys
import json
import time

if sys.platform == "linux":
    import os
    os.environ["LOKY_MAX_CPU_COUNT"] = "6" ### Quick fix 

"""
lightgbm script to train on load forecasting.

"""

@dataclass
class Load_Forecaster_Params: ### Can be moved easily ### Should be called by Load_Forecast_Params(start_training="", end_training="", start_eval="", end_eval="") minimum
    ### All parameters without defaults must be put first
    start_training: str
    end_training: str
    start_eval: str
    end_eval: str
    start_test: str = None
    end_test: str = None

    HORIZON_STEPS: int = 24*4 ### 24 hours ahead with 15-minute data
    OUTPUT_DIR: Path = Path(Path(__file__).parent / "Models" / "Load")
    period: str = "15min"

    boosting_type: str = "dart" ### gbdt or dart
    objective: str = "regression"
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 200
    max_depth: int = 25
    min_child_samples: int = 2
    min_child_weight: float = 3e-4
    subsample: float = 0.9
    subsample_freq: int = 0
    colsample_bytree: float = 1.0 ### Use all features for each tree.
    reg_alpha: float = 0.0 ### L1 regularization
    reg_lambda: float = 0.0 ### L2 regularization
    random_state: int = 42
    verbose: int = 1
    early_stopping_rounds: int = 500
    max_bin: int = 555
    eval_metric: str = "rmse"





class Load_Forecast_Provider:
    def __init__(self, load_forecaster_params: Load_Forecaster_Params):
        self.params = load_forecaster_params

        

    def resample_and_keep_dense_bins(self, expected_samples_per_bin, meter_power_column, PV_power_column, min_frac):
        """
        Resample a high-frequency DataFrame and drop bins with too few samples.

        Assumes:
        - DatetimeIndex
        - roughly 1 Hz data if expected_samples_per_bin = 60*5 = 300 for 5-minute bins

        Returns:
        - resampled DataFrame (mean over each bin)
        - only bins with at least min_frac * expected_samples_per_bin rows kept
        """
        if not isinstance(self.load_df.index, pd.DatetimeIndex):
            raise ValueError("df must have a DatetimeIndex")



        min_required = expected_samples_per_bin * min_frac

        # how many raw rows fall in each 5-minute bin
        counts = self.load_df.resample(self.params.period).size()

        # resampled values
        df_resampled = self.load_df.resample(self.params.period).mean(numeric_only=True)

        # keep only bins with enough samples
        good_bins = counts >= min_required
        df_resampled["dense_bin"] = good_bins
        df_resampled.loc[~df_resampled["dense_bin"], meter_power_column] = np.nan ### Change from previous version, keep full bins but mark them and put nans to drop them later. 
        df_resampled.loc[~df_resampled["dense_bin"], PV_power_column] = np.nan

        return df_resampled
    

    def try_loading_model(self):
        model_path = Path(self.params.OUTPUT_DIR / "Load_Forecast_Model.joblib")
        if model_path.exists():
            parameters_saved_model_path = Path(self.params.OUTPUT_DIR / "parameters_saved_model.json")
            if parameters_saved_model_path.exists():
                with open(parameters_saved_model_path, "r") as f:
                    parameters_saved_model = json.load(f)
                ### Check if all parameters match
                for param in self.params.__dataclass_fields__.keys():
                    if param == "OUTPUT_DIR":
                        continue ### To avoid retraining after using different computers.
                    if param not in parameters_saved_model:
                        print(f"Parameter '{param}' not found in saved model parameters. Will retrain model.")
                        return False
                    saved_value = parameters_saved_model.get(param, None)
                    current_value = getattr(self.params, param)

                    if isinstance(current_value, Path): ### Replaced by Chat, previous comparison breaks upon None values
                        current_value = str(current_value)
                    if saved_value != current_value:
                        print(f"Parameter '{param}' mismatch: saved={saved_value}, current={current_value}. Will retrain model.")
                        return False
                        
                print("All parameters match the saved model. Loading model from disk.")
                self.model = joblib.load(model_path)
                return True
            else:
                print(f"No parameters_saved_model.json found at {parameters_saved_model_path}. Will retrain model.")
                return False
        else:
            print(f"No saved model found at {model_path}. Will train new model.")
            return False
        

    def safe_read_json(self, file_path):
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.endswith("\n"):
                    break
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
        ### To dataframe
        return pd.DataFrame(records)
    


    def training__load_data(self, FILE, json_folder, meter_power_column, PV_power_column, index_column=None):
        if FILE.suffix != ".parquet":
            raise ValueError("Unsupported file format. Please provide a .parquet file. This is for performance reasons.")

        cols_to_read = [meter_power_column, PV_power_column]
        if index_column is not None:
            cols_to_read = [index_column] + cols_to_read
        self.historic_load_df = pd.read_parquet(FILE, columns=cols_to_read)
        print(f"Loaded parquet file with shape {self.historic_load_df.shape}")

        if index_column is not None:
            self.historic_load_df = self.historic_load_df.set_index(index_column)

        if not isinstance(self.historic_load_df.index, pd.DatetimeIndex): ### Speedup by CHAT.
            print(f"Index is not a DatetimeIndex, converting to datetime and setting timezone...")
            self.historic_load_df.index = pd.DatetimeIndex(self.historic_load_df.index)
        if not self.historic_load_df.index.is_monotonic_increasing:
            print(f"Index is not sorted, sorting index...")
            self.historic_load_df = self.historic_load_df.sort_index()

        target_tz = self.historic_load_df.index.tz
        if target_tz is None:
            target_tz = "Europe/Brussels"
            self.historic_load_df.index = self.historic_load_df.index.tz_localize(target_tz)

        if json_folder is not None:
            now_tz = self.historic_load_df.index.tz
            ts = pd.Timestamp.now(tz=now_tz).strftime("%Y-%m-%d")
            file_name = f"rt_{ts}.jsonl"

            self.today_df = self.safe_read_json(Path(json_folder) / file_name)

            print(f"Loaded today's JSONL file with shape {self.today_df.shape}")

            if index_column is not None:
                self.today_df = self.today_df.set_index(index_column)

            self.today_df.index = pd.to_datetime(self.today_df.index, utc=True).tz_convert(target_tz)
            self.today_df = self.today_df.sort_index()

            historic_max = self.historic_load_df.index.max()
            self.today_df = self.today_df[self.today_df.index > historic_max]

            # keep only JSON rows newer than the parquet tail
            historic_max = self.historic_load_df.index.max()
            self.today_df = self.today_df[self.today_df.index > historic_max]

            # force JSON to have the same columns as parquet
            self.today_df = self.today_df.reindex(columns=self.historic_load_df.columns)

            # if there are no truly new rows, just use the parquet
            if self.today_df.empty:
                self.load_df = self.historic_load_df.copy()
            else:
                self.load_df = pd.concat([self.historic_load_df, self.today_df], axis=0).sort_index()
        else:
            self.today_df = pd.DataFrame()
            self.load_df = self.historic_load_df.copy()

        
        if meter_power_column not in self.load_df.columns:
            raise ValueError(f"Meter power column '{meter_power_column}' not found in the DataFrame.")

        if PV_power_column not in self.load_df.columns:
            raise ValueError(f"PV power column '{PV_power_column}' not found in the DataFrame.")
        

        self.load_df[meter_power_column] = pd.to_numeric(self.load_df[meter_power_column], errors="coerce") ### Also part of chatgpt fix due to depreceated package.
        self.load_df[PV_power_column] = pd.to_numeric(self.load_df[PV_power_column], errors="coerce")

        ### Delete all entries where hw_active_power_w is null
        self.load_df = self.load_df[self.load_df["HW_active_power_w"].notna()]

        ### Resample and delete bins with too few samples

        if self.params.period is not None: ### Expensive function
            self.load_df = self.resample_and_keep_dense_bins(int(pd.Timedelta(self.params.period).total_seconds()), meter_power_column, PV_power_column, min_frac=0.9)


        print(f"After resampling and dropping sparse bins, shape is {self.load_df.shape}")
        ### Timestamps where self.load_df[meter_power_column] exists and isna is False

        ### New changes in this version
        #self.load_df = self.load_df[self.load_df[meter_power_column].notnull()] #& self.load_df[PV_power_column].notnull()]
        self.load_df["Net_load_kWh"] = self.load_df[meter_power_column] + self.load_df[PV_power_column].fillna(0)
        period_hours = pd.Timedelta(self.params.period).total_seconds() / 3600
        self.load_df["Net_load_kWh"] = self.load_df["Net_load_kWh"] / 1000 * period_hours

        #print(f"After binning and dropping sparse bins, self.load_df['Net_load_kWh'] tail:\n{self.load_df['Net_load_kWh'].tail()}") ### DEBUG

        ### Raise if any negative values in Net_load_kWh
        if (self.load_df["Net_load_kWh"] < 0).any():
            print(f"Negative values at:\n{(self.load_df[self.load_df['Net_load_kWh'] < 0]['Net_load_kWh']).to_string()}")
            raise ValueError("Negative values found in Net_load_kWh. Please check the data and preprocessing steps.")


        print(f"Range: {self.load_df['Net_load_kWh'].index.min()} to {self.load_df['Net_load_kWh'].index.max()}")


        self.y = self.load_df["Net_load_kWh"] ### --> Load now available as self.y for forecasting




    def training__make_features(self):
        self.params.steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(self.params.period))
        idx = pd.DatetimeIndex(self.y.index)
        self.X = pd.DataFrame(index=idx)
        ### Lagging features
        self.params.lags = [
            1,2,3,4,
            self.params.steps_per_hour, 
            self.params.steps_per_hour*24, ### 1 day lag
            self.params.steps_per_hour*24*7 ### 1 week lag
        ]
        for lag in self.params.lags:
            self.X[f"lag_{lag}"] = self.y.shift(lag)
        self.X["diff_1"] = self.y.shift(1) - self.y.shift(2)

        ### Add zero lag, when called at 22:02 for example, lag_0 brings the data of 21:45 until 22:00!!
        self.X["lag_0"] = self.y

        ### Rolling features
        self.params.roll_windows = [ ### May add more later
            self.params.steps_per_hour, 
            self.params.steps_per_hour*24
        ]
        for roll in self.params.roll_windows:
            #self.X[f"roll_mean_{roll}"] = self.y.shift(1).rolling(roll).mean() ### Adjust to use the real last window
            #self.X[f"roll_std_{roll}"] = self.y.shift(1).rolling(roll).std()
            #self.X[f"roll_min_{roll}"] = self.y.shift(1).rolling(roll).min()
            #self.X[f"roll_max_{roll}"] = self.y.shift(1).rolling(roll).max()

            self.X[f"roll_mean_{roll}"] = self.y.rolling(roll).mean()
            self.X[f"roll_std_{roll}"] = self.y.rolling(roll).std()
            self.X[f"roll_min_{roll}"] = self.y.rolling(roll).min()
            self.X[f"roll_max_{roll}"] = self.y.rolling(roll).max()

        ### Time features
        dow = idx.dayofweek
        self.X["is_weekend"] = (dow >= 5).astype(int)
        self.X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        self.X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        hour = idx.hour
        minute = idx.minute
        min_of_day = hour * 60 + minute
        self.X["mod_sin"] = np.sin(2 * np.pi * min_of_day / (24 * 60))
        self.X["mod_cos"] = np.cos(2 * np.pi * min_of_day / (24 * 60))

        doy = idx.dayofyear
        self.X["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        self.X["doy_cos"] = np.cos(2 * np.pi * doy / 365)

        ### Specific features
        self.X["is_morning_peak"] = ((hour >= 7) & (hour < 10)).astype(int)
        self.X["is_evening_peak"] = ((hour >= 17) & (hour < 20)).astype(int)




    def training__make_future_features(self, origins):
        X_rows = []
        Y_rows = []
        future_times = []

        idx = self.y.index
        target_values = self.y.values ### Turns it into a numpy array

        for origin in origins:
            origin_features = self.X.iloc[origin]

            for h in range(1, self.params.HORIZON_STEPS + 1):
                future_idx = origin + h
                future_time = idx[future_idx]

                row = origin_features.to_dict()

                ### Time features
                future_dow = future_time.dayofweek
                row["future_is_weekend"] = int(future_dow >= 5)
                row["future_dow_sin"] = np.sin(2 * np.pi * future_dow / 7)
                row["future_dow_cos"] = np.cos(2 * np.pi * future_dow / 7)

                future_hour = future_time.hour
                future_minute = future_time.minute

                min_of_day = future_hour * 60 + future_minute
                row["future_mod_sin"] = np.sin(2 * np.pi * min_of_day / (1440))
                row["future_mod_cos"] = np.cos(2 * np.pi * min_of_day / (1440))
                
                row["future_doy_sin"] = np.sin(2 * np.pi * future_time.dayofyear / 365)
                row["future_doy_cos"] = np.cos(2 * np.pi * future_time.dayofyear / 365)

                ### Step features
                row["h_step"] = h
                row["h_step_sin"] = np.sin(2 * np.pi * h / self.params.HORIZON_STEPS)
                row["h_step_cos"] = np.cos(2 * np.pi * h / self.params.HORIZON_STEPS)
                row["h_norm"] = h / self.params.HORIZON_STEPS ### Normalized step, added by CHAT
                row["h_hours"] = h * (pd.Timedelta(self.params.period).total_seconds() / 3600) ### Step in hours, added by CHAT

                X_rows.append(row)
                Y_rows.append(target_values[future_idx])
                future_times.append(future_time)

        X_with_future = pd.DataFrame(X_rows)
        Y_with_future = pd.Series(Y_rows, name="target")
        future_times = pd.DatetimeIndex(future_times) ### For plotting later

        X_with_future["target"] = Y_with_future
        X_with_future["future_time"] = future_times

        X_with_future = X_with_future.dropna() ### This is how v2 differs from v1.

        Y_with_future = X_with_future.pop("target")
        future_times = pd.DatetimeIndex(X_with_future.pop("future_time"))

        ### DEBUG, set true for debugging.
        if False:
            print(X_with_future["h_step"].value_counts().sort_index())

        return X_with_future, Y_with_future, future_times
    



    def build_origins_by_date(self): ### By chat
        idx = self.y.index
        positions = np.arange(len(idx) - self.params.HORIZON_STEPS)

        origin_times = idx[positions]

        horizon_td = self.params.HORIZON_STEPS * pd.Timedelta(self.params.period)

        train_start = pd.Timestamp(self.params.start_training).tz_localize(idx.tz)
        train_end   = pd.Timestamp(self.params.end_training).tz_localize(idx.tz) - horizon_td ### Little extra protection by chat

        eval_start = pd.Timestamp(self.params.start_eval).tz_localize(idx.tz)
        eval_end   = pd.Timestamp(self.params.end_eval).tz_localize(idx.tz) - horizon_td

        if self.params.start_test is not None and self.params.end_test is not None:
            test_start = pd.Timestamp(self.params.start_test).tz_localize(idx.tz)
            test_end = pd.Timestamp(self.params.end_test).tz_localize(idx.tz) - horizon_td

        self.train_origins = positions[(origin_times >= train_start) & (origin_times < train_end)]
        self.eval_origins  = positions[(origin_times >= eval_start) & (origin_times < eval_end)]
        if self.params.start_test is not None and self.params.end_test is not None:
            self.test_origins  = positions[(origin_times >= test_start) & (origin_times < test_end)]
        else:
            self.test_origins = np.array([], dtype=int)






    def training__build_full_feature_set(self):
        self.X_train, self.y_train, self.train_future_times = self.training__make_future_features(self.train_origins)
        self.X_eval, self.y_eval, self.eval_future_times = self.training__make_future_features(self.eval_origins)

        print(f"Training set range: {self.train_future_times.min()} to {self.train_future_times.max()}")


    def training__fit_model(self):

        self.model = LGBMRegressor(
            boosting_type=self.params.boosting_type,
            objective=self.params.objective,
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            num_leaves=self.params.num_leaves,
            max_depth=self.params.max_depth,
            min_child_samples=self.params.min_child_samples,
            min_child_weight=self.params.min_child_weight,
            subsample=self.params.subsample,
            subsample_freq=self.params.subsample_freq,
            colsample_bytree=self.params.colsample_bytree,
            reg_alpha=self.params.reg_alpha,
            reg_lambda=self.params.reg_lambda,
            max_bin=self.params.max_bin,
            random_state=self.params.random_state,
            verbosity=self.params.verbose,
        )

        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_eval, self.y_eval)],
            eval_metric=self.params.eval_metric,
            callbacks=[ ### Early stopping done by chat
                early_stopping(self.params.early_stopping_rounds, first_metric_only=True, verbose=True),
                log_evaluation(50),
            ],
        )

        print("Best iteration:", self.model.best_iteration_)


    def training__save_model(self):
        self.params.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = Path(self.params.OUTPUT_DIR / "Load_Forecast_Model.joblib")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        parameters_saved_model_path = Path(self.params.OUTPUT_DIR / "parameters_saved_model.json")
        metadata = asdict(self.params) ### Done by Chat
        metadata["OUTPUT_DIR"] = str(metadata["OUTPUT_DIR"])
        with open(parameters_saved_model_path, "w") as f:
            json.dump(metadata, f, indent=2)



    def training__pipeline(self, FILE, json_folder, meter_power_column, PV_power_column, index_column=None):
        start_1 = time.time()
        self.training__load_data(FILE, json_folder, meter_power_column, PV_power_column, index_column)
        start_2 = time.time()
        self.training__make_features()
        start_3 = time.time()
        self.build_origins_by_date()
        start_4 = time.time()
        


        response = self.try_loading_model()
        start_5 = time.time()

        if response==True:
            print("Model loaded successfully. Skipping training.")
        else:
            print("Training model...")
            start_training_time = time.time()
            self.training__build_full_feature_set()
            self.training__fit_model()
            self.training__save_model()
            end_training_time = time.time()
            print(f"Training time: {end_training_time - start_training_time:.2f} seconds")
        return {
            "load_data_time": start_2 - start_1,
            "make_features_time": start_3 - start_2,
            "build_origins_time": start_4 - start_3,
            "load_model_time": start_5 - start_4,
        }
        


    def predict_from_timestamp(self, timestamp):
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Model is not trained or loaded.")

        idx = self.y.index
        ts = timestamp

        ### Few checks before
        if not isinstance(ts, pd.Timestamp):
            raise ValueError("Input timestamp must be a pandas Timestamp.")
        if ts.tz is None:
            raise ValueError(f"Input timestamp must be timezone-aware and set to the same timezone as the training data: {idx.tz}")
        if str(ts.tz) != str(idx.tz):
            raise ValueError(f"Input timestamp timezone {ts.tzinfo} does not match training data timezone {idx.tz}")

        origin_features = self.X.loc[ts] ### Fix from chat for stability
        if origin_features.isna().any():
            ###raise ValueError(f"Timestamp {ts} does not have enough history for lag/rolling features.") ### Applying this becomes problematic
            print(f"Warning: Timestamp {ts} has NaN values in features. This may be due to insufficient history for lag/rolling features. Proceeding with prediction, but results may be unreliable.")

        X_rows = []
        future_times = []

        step = pd.Timedelta(self.params.period)

        for h in range(1, self.params.HORIZON_STEPS + 1):
            future_time = ts + h * step
            row = origin_features.to_dict()

            future_dow = future_time.dayofweek
            row["future_is_weekend"] = int(future_dow >= 5)
            row["future_dow_sin"] = np.sin(2 * np.pi * future_dow / 7)
            row["future_dow_cos"] = np.cos(2 * np.pi * future_dow / 7)

            future_hour = future_time.hour
            future_minute = future_time.minute
            min_of_day = future_hour * 60 + future_minute
            row["future_mod_sin"] = np.sin(2 * np.pi * min_of_day / 1440)
            row["future_mod_cos"] = np.cos(2 * np.pi * min_of_day / 1440)

            row["future_doy_sin"] = np.sin(2 * np.pi * future_time.dayofyear / 365)
            row["future_doy_cos"] = np.cos(2 * np.pi * future_time.dayofyear / 365)

            row["h_step"] = h
            row["h_step_sin"] = np.sin(2 * np.pi * h / self.params.HORIZON_STEPS)
            row["h_step_cos"] = np.cos(2 * np.pi * h / self.params.HORIZON_STEPS)
            row["h_norm"] = h / self.params.HORIZON_STEPS
            row["h_hours"] = h * (step.total_seconds() / 3600)

            X_rows.append(row)
            future_times.append(future_time)

        X_pred = pd.DataFrame(X_rows)
        #print(f"Prediction dataframe head:\n{X_pred.head()}") ### DEBUG
        #print(f"\nPrediction dataframe tail:\n{X_pred.tail()}") ### DEBUG
        y_pred = self.model.predict(X_pred)

        return pd.Series(y_pred, index=pd.DatetimeIndex(future_times), name="forecast_24h")

    
    
    
    ######### TESTING SECTION. Not to bse used if not from file internally. Only to check performance on unseen data.
    def testing__build_full_feature_set(self):
        self.X_test, self.y_test, self.test_future_times = self.training__make_future_features(self.test_origins) ### Reuse function

    def testing__predict(self):
        self.y_test_pred = self.model.predict(self.X_test)

        
    
    def testing__evaluate(self):
        mae = mean_absolute_error(self.y_test, self.y_test_pred)
        rmse = mean_squared_error(self.y_test, self.y_test_pred) ** 0.5

        print("\nTEST RESULTS")
        print("MAE :", mae)
        print("RMSE:", rmse)

    import matplotlib.pyplot as plt

    def testing__plot_test_period(self): ### By chat
        import matplotlib.pyplot as plt
        plot_df = pd.DataFrame({
            "y_true": self.y_test.to_numpy(),
            "y_pred": self.y_test_pred,
            "time": self.test_future_times,
        })

        # combine overlapping predictions for the same timestamp
        plot_df = plot_df.groupby("time", as_index=True).mean()

        plt.figure(figsize=(14, 5))
        plt.plot(plot_df.index, plot_df["y_true"], label="Actual")
        plt.plot(plot_df.index, plot_df["y_pred"], label="Predicted")
        plt.title("Test period")
        plt.xlabel("Time")
        plt.ylabel("Net_load_kWh")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def testing__pipeline(self):
        self.testing__build_full_feature_set()
        self.testing__predict()
        self.testing__evaluate()
        self.testing__plot_test_period()
        ### Evaluate most important features
        feature_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        print("\nTop 20 feature importances:")
        print(feature_importances.head(20))







if __name__ == "__main__":
    ### if on mac:
    if sys.platform == "darwin":
        FILE = Path("/Users/brechtgoethals/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/inverter_logs_parquet/rt_all_2.parquet")
        json_folder = Path("/Users/brechtgoethals/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/inverter_logs") #  None
    ### if on linux:
    elif sys.platform == "linux":
        FILE = Path("/home/brecht1949/inverter_logs_parquet/rt_all_2.parquet")
        json_folder = Path("/home/brecht1949/inverter_logs/")
    elif sys.platform == "win32":
        FILE = Path(r"C:\Users\brech\OneDrive - Delft University of Technology\inverter_logs_parquet\rt_all_2.parquet")

    ts_now = pd.Timestamp.now(tz="Europe/Brussels")


    MODEL_SAVE_FOLDER = Path(Path(__file__).parent / "Models" / "Load")
    PV_column = "AC_active_power"
    meter_column = "HW_active_power_w"

    params = Load_Forecaster_Params(
        start_training="2026-01-10",
        end_training="2026-03-30",
        start_eval="2026-03-30",
        end_eval="2026-04-02",
        start_test=None,#"2026-03-25",
        end_test=None,#"2026-04-03",
        OUTPUT_DIR=MODEL_SAVE_FOLDER,
    )
    provider = Load_Forecast_Provider(params)

    
    
    """
    ### Make list of timestamps between xxx and xxx with timestamp 00:00:00 every day, and predict for those
    forecast_timestamps = pd.date_range(start="2026-03-31", end="2026-04-04", freq="15min", tz="Europe/Brussels")
    validate_timestamps = []
    invalid_timestamps = []
    for ts in forecast_timestamps:
        try:
            forecast = provider.predict_from_timestamp(ts)
            validate_timestamps.append(ts)
        except Exception as e:
            print(f"Could not predict for timestamp {ts}: {e}")
            invalid_timestamps.append(ts)

    print("Invalid timestamps for prediction:")
    for ts in invalid_timestamps:
        print(ts)
    print(f"Ratio valid timestamps: {len(validate_timestamps)/len(forecast_timestamps):.2%}")
    print(f"Invalid timestamps ratio: {len(invalid_timestamps)/len(forecast_timestamps):.2%}")
    """

    """
    timestamp_to_forecast = pd.Timestamp("2026-04-06 22:00:00", tz="Europe/Brussels")
    forecast = provider.predict_from_timestamp(timestamp_to_forecast)
    print(f"\nForecast for {timestamp_to_forecast} to {timestamp_to_forecast + pd.Timedelta(params.period) * params.HORIZON_STEPS}:")
    print(forecast)
    """
    
    #provider.training__pipeline(FILE, json_folder, meter_column, PV_column, index_column="Timestamp_package")
    #provider.testing__pipeline() ### If wanting to test, fill in test interval and run this line

    
    if sys.platform == "darwin": ### Should only be macos!!
        begin = time.time()
        timestamp_to_forecast = pd.Timestamp("2026-04-07 22:45:00", tz="Europe/Brussels")
        times = provider.training__pipeline(FILE, json_folder, meter_column, PV_column, index_column="Timestamp_package")
        print(f"\nPipeline times: {times}")
        start_prediction_time = time.time()
        forecast = provider.predict_from_timestamp(timestamp_to_forecast)
        stop_prediction_time = time.time()
        print(f"Prediction time: {stop_prediction_time - start_prediction_time:.2f} seconds")
        print(f"\nForecast for {timestamp_to_forecast} to {timestamp_to_forecast + pd.Timedelta(params.period) * params.HORIZON_STEPS}:")
        print(forecast)
        end = time.time()
        print(f"\nTotal time for training/loading and prediction: {end - begin:.2f} seconds")

    if sys.platform == "linux": ### SHould be linux, will name else for now for debugging!!!!
        ts_last_bin = ts_now.floor(params.period)
        time_diff = ts_now - ts_last_bin
        if time_diff < pd.Timedelta(minutes=4):
            ts_to_forecast = ts_last_bin - pd.Timedelta(params.period)
            print(f"Current time {ts_now} is within 2 minutes of the last bin {ts_last_bin}. Forecasting from previous bin {ts_to_forecast}.\n")
            
            start_time = time.time()
            print(f"Running training pipeline to ensure model is up to date with the latest data...")
            times = provider.training__pipeline(FILE, json_folder, meter_column, PV_column, index_column="Timestamp_package")
            print(f"\nPipeline times: {times}")
            print(f"Training pipeline completed. \nNow predicting for {ts_now} with origin {ts_to_forecast}...")

            start_prediction_time = time.time()
            forecast = provider.predict_from_timestamp(ts_to_forecast)
            stop_prediction_time = time.time()
            print(f"Prediction time: {stop_prediction_time - start_prediction_time:.2f} seconds")
            print(f"\nForecast for {ts_to_forecast} to {ts_to_forecast + pd.Timedelta(params.period) * params.HORIZON_STEPS}:")
            print(forecast)
            end_time = time.time()
            print(f"\nTotal time for training/loading and prediction: {end_time - start_time:.2f} seconds")
        else:
            print(f"Should be called within two minutes after last bin. Last one was at {ts_last_bin} and next one is at {ts_last_bin + pd.Timedelta(params.period)}.\nCurrent time is {ts_now}.")