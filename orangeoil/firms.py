import io

import pandas as pd
import requests
from tqdm import tqdm

from .config import CFG, OIL_REGIONS, require
from .db import FIRMS_LIMITER


def firms_chunk(bbox, start, days=5):
    west, south, east, north = bbox
    age = (pd.Timestamp.utcnow().tz_localize(None) - start).days
    source = CFG["firms_src_hist"] if age > 60 else CFG["firms_src_nrt"]
    url = (
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{require('FIRMS_MAP_KEY')}/{source}/{west},{south},{east},{north}/{days}/{start:%Y-%m-%d}"
    )
    FIRMS_LIMITER.wait()
    response = requests.get(url, timeout=90)
    if response.status_code != 200 or "latitude" not in response.text[:400]:
        return None
    df = pd.read_csv(io.StringIO(response.text))
    if len(df) and "confidence" in df:
        df = df[df["confidence"].astype(str).str.lower().isin(["n", "h", "nominal", "high"])]
    index = pd.date_range(start, periods=days).strftime("%Y-%m-%d")
    if len(df):
        daily = df.groupby("acq_date").agg(n_det=("frp", "size"), frp_sum=("frp", "sum"))
        return daily.reindex(index).fillna(0)
    return pd.DataFrame({"n_det": 0, "frp_sum": 0.0}, index=index)


def firms_sync(con, start=None, end=None, step=5):
    start = pd.Timestamp(start or CFG["start"])
    end = pd.Timestamp(end or pd.Timestamp.utcnow().tz_localize(None).normalize())
    chunks = pd.date_range(start, end - pd.Timedelta(days=1), freq=f"{step}D")
    for region, bbox in OIL_REGIONS.items():
        for chunk in tqdm(chunks, desc=region, leave=False):
            days = min(step, (end - chunk).days)
            stop = chunk + pd.Timedelta(days=days)
            cached = con.execute(
                "SELECT COUNT(*) FROM firms_daily WHERE region=? AND date>=? AND date<?",
                (region, f"{chunk:%Y-%m-%d}", f"{stop:%Y-%m-%d}"),
            ).fetchone()[0]
            if cached >= days:
                continue
            daily = firms_chunk(bbox, chunk, days)
            if daily is None:
                continue
            con.executemany(
                "INSERT OR REPLACE INTO firms_daily VALUES (?,?,?,?)",
                [(i, region, int(r.n_det), float(r.frp_sum)) for i, r in daily.iterrows()],
            )
            con.commit()
    print("FIRMS synchronisé.")
