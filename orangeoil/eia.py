import pandas as pd
import requests

from .config import CFG, require

EIA_WEEKLY = {
    "WCESTUS1": "stocks_crude",
    "WCRFPUS2": "prod_crude",
    "WCRIMUS2": "imports_crude",
    "WPULEUS3": "refinery_util",
}


def eia_fetch(route, series, freq, start):
    key = require("EIA_API_KEY")
    rows = []
    offset = 0
    while True:
        params = {
            "api_key": key,
            "frequency": freq,
            "data[0]": "value",
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": 5000,
            "facets[series][]": series,
        }
        response = requests.get(f"https://api.eia.gov/v2/{route}/data/", params=params, timeout=60)
        response.raise_for_status()
        data = response.json()["response"]["data"]
        rows += data
        if len(data) < 5000:
            break
        offset += 5000
    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def load_wti():
    df = eia_fetch("petroleum/pri/spt", ["RWTC"], "daily", CFG["start"])
    series = df.set_index("period")["value"].rename("wti").sort_index()
    return series[series > 1]


def load_eia_weekly():
    df = eia_fetch("petroleum/sum/sndw", list(EIA_WEEKLY), "weekly", CFG["start"])
    pivot = df.pivot(index="period", columns="series", values="value").rename(columns=EIA_WEEKLY)
    pivot.index = pivot.index + pd.Timedelta(days=CFG["eia_lag_days"])
    return pivot
