import os

from dotenv import load_dotenv

load_dotenv()


def require(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Variable d'environnement manquante : {name}")
    return value


CFG = {
    "start": "2021-01-01",
    "window": 30,
    "horizon": 5,
    "seed": 42,
    "db_path": "orangeoil.db",
    "model_path": "orangeoil.pt",
    "reports_dir": "reports",
    "laden_draught_m": 11.0,
    "eia_lag_days": 5,
    "firms_src_hist": "VIIRS_SNPP_SP",
    "firms_src_nrt": "VIIRS_SNPP_NRT",
}

OIL_REGIONS = {
    "permian": (-104.5, 30.5, -101.0, 33.5),
    "bakken": (-104.5, 47.0, -101.5, 49.0),
    "eagle_ford": (-100.0, 27.8, -97.0, 29.8),
    "w_siberia": (68.0, 58.0, 80.0, 64.0),
    "persian_gulf": (47.0, 25.0, 52.5, 31.0),
    "niger_delta": (5.0, 4.0, 7.5, 6.0),
    "libya": (15.0, 28.0, 22.0, 31.0),
}

CHOKEPOINTS = {
    "hormuz": (55.5, 25.5, 57.5, 27.2),
    "bab_el_mandeb": (42.5, 12.0, 44.0, 13.5),
    "suez": (32.0, 29.5, 33.0, 31.5),
    "malacca": (99.0, 0.5, 104.5, 4.0),
    "us_gulf": (-97.5, 26.0, -88.0, 30.5),
}
