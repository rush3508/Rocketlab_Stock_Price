"""
Merges Electron launch events into the price DataFrame.
Adds features that capture the market impact of launches.
"""

import pandas as pd
import numpy as np
from data.fetch_launches import load_launches


def add_launch_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds launch-related columns to price_df (indexed by date):
      - is_launch_day    : 1 if a launch happened on that trading day
      - launch_success   : 1 = successful launch, -1 = failed, 0 = no launch
      - days_since_launch: trading days since the last launch
      - launches_30d     : number of launches in the previous 30 calendar days
    """
    df = price_df.copy()
    launches = load_launches()
    launches["date"] = pd.to_datetime(launches["date"]).dt.normalize()

    # Map launch events onto trading dates
    launch_map = launches.set_index("date")["success"]

    df["is_launch_day"] = df.index.isin(launch_map.index).astype(int)
    df["launch_success"] = 0

    for date in launch_map.index:
        # Find the nearest trading day on or after the launch
        mask = df.index >= date
        if mask.any():
            nearest = df.index[mask][0]
            df.loc[nearest, "is_launch_day"] = 1
            df.loc[nearest, "launch_success"] = 1 if launch_map[date] else -1

    # Days since last launch (in trading days)
    last_launch = pd.NaT
    days_since = []
    for idx in df.index:
        matching = launches[launches["date"] <= idx]
        if not matching.empty:
            last_launch = matching["date"].iloc[-1]
            delta = (idx - last_launch).days
        else:
            delta = np.nan
        days_since.append(delta)
    df["days_since_launch"] = days_since

    # Rolling 30-day launch count
    launch_dates_series = pd.Series(1, index=pd.DatetimeIndex(launches["date"]))
    launch_dates_series = launch_dates_series[~launch_dates_series.index.duplicated()]
    launch_counts = launch_dates_series.reindex(df.index, fill_value=0)
    df["launches_30d"] = (
        launch_counts.rolling("30D", min_periods=1).sum().values
    )

    return df
