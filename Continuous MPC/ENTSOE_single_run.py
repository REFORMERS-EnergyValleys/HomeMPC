import pandas as pd
from entsoe import EntsoePandasClient
from pathlib import Path
import time
from termcolor import colored
import sys

import requests



class Entsoe:
    def __init__(self, API_KEY, COUNTRY_CODE, TIMEZONE, cache_path, RESAMPLE=None):
        self.price_df = None
        self.client = EntsoePandasClient(api_key=API_KEY)
        self.country_code = COUNTRY_CODE
        self.RESAMPLE = RESAMPLE
        self.TZ = TIMEZONE
        self.cache_path = cache_path




    def get_DA_PRICES(self, start_date, end_date):
        while True:
            try:
                ENTSOE_PRICES = self.client.query_day_ahead_prices(self.country_code, start=start_date, end=end_date) / 1000  ### Returns a pandas series
                self.price_serie = ENTSOE_PRICES
                break
            except Exception as e:
                print(colored(f"Error fetching data from ENTSOE: {e}\n", "red"))
                print(colored(f"Retrying in 30 seconds...", "yellow"))
                time.sleep(30)

        self.price_serie.index = self.price_serie.index.tz_convert(self.TZ)
        if self.RESAMPLE is not None:
            self.price_serie = self.price_serie.resample(self.RESAMPLE).ffill()
        self.price_df = self.price_serie.to_frame(name="DA_Price") ### Convert serie to df for consistency
    


    def run_retrieve_ENTSOE(self, FIRST_RUN=False):
        start_date = pd.Timestamp.now(tz=self.TZ).floor("D")
        end_date = start_date + pd.Timedelta(days=2)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            print(f"Fetching day-ahead prices from {start_date} to {end_date}...")
            self.get_DA_PRICES(start_date, end_date)

            should_save = True
            if self.cache_path.exists():
                cached_df = pd.read_parquet(self.cache_path)
                should_save = not cached_df.equals(self.price_df)

            if should_save:
                self.price_df.to_parquet(self.cache_path)
                print(colored("Cache updated.", "green"))
                print(f"Current time: {pd.Timestamp.now(tz=self.TZ)}")
                break
            else:
                print(colored("No change in cached data.", "yellow"))
                print(f"Current time: {pd.Timestamp.now(tz=self.TZ)}")
                if FIRST_RUN:
                    break
                else:
                    print(colored("\nRetrying in 30 seconds...", "yellow"))
                    time.sleep(30)




if __name__ == "__main__":
    if sys.platform == "darwin":
        API_KEY = pd.read_csv("/Users/brechtgoethals/API_ENTSOE.txt", header=None).iloc[0,0]
    elif sys.platform == "linux":
        API_KEY = pd.read_csv("/home/brecht1949/API_ENTSOE.txt", header=None).iloc[0,0]

    COUNTRY_CODE = "BE"  # Example country code for Belgium
    TIMEZONE = "Europe/Brussels"
    cache_path = Path(__file__).parent / "Data" / "DA Prices" / "DA_prices_cache.parquet"

    entsoe_client = Entsoe(API_KEY, COUNTRY_CODE, TIMEZONE, cache_path)

    ### Manual run
    #entsoe_client.run_retrieve_ENTSOE(FIRST_RUN=True)

    entsoe_client.run_retrieve_ENTSOE(FIRST_RUN=False)
