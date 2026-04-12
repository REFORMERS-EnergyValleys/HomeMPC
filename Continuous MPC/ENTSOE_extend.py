import pandas as pd
from pathlib import Path
import requests




def Extend_DA_w_forecast(Entsoe_cache_path, country_code, timezone):
        UNIT = "EUR_PER_KWH"
        HOURS = 30 ### Just to be sure
        url = f"http://localhost:8000/prices_short?region={country_code}&hours={HOURS}&unit={UNIT}&timezone={timezone}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        ts = pd.to_datetime(data["s"], unit="s", utc=True).tz_convert(timezone) ### Has to be called like this in order to get the correct timezone!
        prices_forecast = pd.Series(data["t"], index=ts, name="price")

        original_df = pd.read_parquet(Entsoe_cache_path)["DA_Price"]
        combined = original_df.combine_first(prices_forecast)
        combined = combined.sort_index()
        ### Save as the same file but with _extended at the end
        combined.to_frame(name="DA_Price").to_parquet(Entsoe_cache_path.with_name(Entsoe_cache_path.stem + "_extended.parquet"))
        print(f"Extended DA prices, range: {combined.index.min()} to {combined.index.max()}")


if __name__ == "__main__":
    Country_code = "BE"
    Timezone = "Europe/Brussels"

    cache_path = Path(__file__).parent / "Data" / "DA Prices" / "DA_prices_cache.parquet"
    Extend_DA_w_forecast(cache_path, Country_code, Timezone)


