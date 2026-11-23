import collections
import sqlite3
import time

from .config import CFG

SCHEMA = """
CREATE TABLE IF NOT EXISTS firms_daily(
    date TEXT, region TEXT, n_det INTEGER, frp_sum REAL, PRIMARY KEY(date, region)
);
CREATE TABLE IF NOT EXISTS ais_pos(
    mmsi INTEGER, ts TEXT, lat REAL, lon REAL, sog REAL, cog REAL, zone TEXT
);
CREATE TABLE IF NOT EXISTS ais_static(
    mmsi INTEGER, ts TEXT, name TEXT, ship_type INTEGER, draught REAL, dest TEXT
);
CREATE INDEX IF NOT EXISTS ix_pos ON ais_pos(mmsi, ts);
CREATE INDEX IF NOT EXISTS ix_static ON ais_static(mmsi, ts);
"""


def get_connection(path=None):
    con = sqlite3.connect(path or CFG["db_path"], check_same_thread=False)
    con.executescript(SCHEMA)
    return con


class RateLimiter:
    def __init__(self, n=4500, period=600):
        self.n = n
        self.period = period
        self.calls = collections.deque()

    def wait(self):
        now = time.time()
        while self.calls and now - self.calls[0] > self.period:
            self.calls.popleft()
        if len(self.calls) >= self.n:
            pause = self.period - (now - self.calls[0]) + 0.5
            print(f"[rate-limit] pause {pause:.0f}s")
            time.sleep(max(pause, 0))
        self.calls.append(time.time())


FIRMS_LIMITER = RateLimiter()
