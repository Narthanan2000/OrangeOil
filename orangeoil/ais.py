import asyncio
import json
import time

import folium
import numpy as np
import pandas as pd

from .config import CFG, CHOKEPOINTS, require


def zone_of(lat, lon):
    for zone, (west, south, east, north) in CHOKEPOINTS.items():
        if west <= lon <= east and south <= lat <= north:
            return zone
    return None


async def ais_collect(con, seconds=0, min_gap_s=300):
    import websockets

    boxes = [[[s, w], [n, e]] for (w, s, e, n) in CHOKEPOINTS.values()]
    subscription = {
        "APIKey": require("AISSTREAM_API_KEY"),
        "BoundingBoxes": boxes,
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }
    deadline = time.time() + seconds if seconds else float("inf")
    last_seen = {}
    last_static = {}
    positions = []
    statics = []
    count = 0

    def flush():
        nonlocal positions, statics
        if positions:
            con.executemany("INSERT INTO ais_pos VALUES (?,?,?,?,?,?,?)", positions)
        if statics:
            con.executemany("INSERT INTO ais_static VALUES (?,?,?,?,?,?)", statics)
        con.commit()
        positions, statics = [], []

    try:
        while time.time() < deadline:
            try:
                async with websockets.connect("wss://stream.aisstream.io/v0/stream", ping_interval=20) as ws:
                    await ws.send(json.dumps(subscription))
                    while time.time() < deadline:
                        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                        meta = message.get("MetaData", {})
                        kind = message.get("MessageType")
                        mmsi = meta.get("MMSI")
                        ts = str(meta.get("time_utc", ""))[:19]
                        if kind == "PositionReport":
                            body = message["Message"]["PositionReport"]
                            if time.time() - last_seen.get(mmsi, 0) < min_gap_s:
                                continue
                            zone = zone_of(body["Latitude"], body["Longitude"])
                            if zone is None:
                                continue
                            last_seen[mmsi] = time.time()
                            positions.append(
                                (mmsi, ts, body["Latitude"], body["Longitude"], body.get("Sog"), body.get("Cog"), zone)
                            )
                        elif kind == "ShipStaticData":
                            body = message["Message"]["ShipStaticData"]
                            key = (body.get("MaximumStaticDraught"), (body.get("Destination") or "").strip())
                            if last_static.get(mmsi) == key:
                                continue
                            last_static[mmsi] = key
                            name = (body.get("Name") or meta.get("ShipName", "")).strip()
                            statics.append((mmsi, ts, name, body.get("Type"), key[0], key[1]))
                        count += 1
                        if len(positions) + len(statics) >= 200:
                            flush()
            except Exception as error:
                print("AIS reconnexion :", repr(error))
                await asyncio.sleep(5)
    finally:
        flush()
        print(f"AIS : {count} messages traités.")


def run_collect(con, seconds=0):
    try:
        asyncio.run(ais_collect(con, seconds))
    except KeyboardInterrupt:
        print("Collecte arrêtée.")


def get_track(con, mmsi):
    return pd.read_sql("SELECT * FROM ais_pos WHERE mmsi=? ORDER BY ts", con, params=(mmsi,), parse_dates=["ts"])


def latest_static(con):
    df = pd.read_sql("SELECT * FROM ais_static ORDER BY ts", con, parse_dates=["ts"])
    if df.empty:
        return df
    return df.drop_duplicates("mmsi", keep="last").set_index("mmsi")


def ships_daily(con):
    pos = pd.read_sql("SELECT * FROM ais_pos", con, parse_dates=["ts"]).sort_values("ts")
    static = pd.read_sql("SELECT mmsi, ts, ship_type, draught FROM ais_static", con, parse_dates=["ts"]).sort_values("ts")
    if pos.empty or static.empty:
        return pd.DataFrame()
    df = pd.merge_asof(pos, static, on="ts", by="mmsi")
    df = df[(df.ship_type >= 80) & (df.ship_type <= 89)].copy()
    df["day"] = df.ts.dt.normalize()
    df["laden"] = (df.draught >= CFG["laden_draught_m"]).astype(float)
    df["still"] = (df.sog < 0.5).astype(float)
    grouped = df.groupby(["day", "zone"]).agg(
        n=("mmsi", "nunique"),
        draught=("draught", "mean"),
        laden=("laden", "mean"),
        still=("still", "mean"),
        sog=("sog", "mean"),
    )
    out = grouped.unstack("zone")
    out.columns = [f"ais_{zone}_{metric}" for metric, zone in out.columns]
    return out


def map_tankers(con, path="tankers_map.html", top_n=40):
    static = latest_static(con)
    pos = pd.read_sql("SELECT * FROM ais_pos ORDER BY ts", con, parse_dates=["ts"])
    if pos.empty or static.empty:
        print("Pas encore de données AIS.")
        return None
    tankers = static[(static.ship_type >= 80) & (static.ship_type <= 89)].index
    pos = pos[pos.mmsi.isin(tankers)]
    top = pos.mmsi.value_counts().head(top_n).index
    fmap = folium.Map(location=[20, 50], zoom_start=3, tiles="CartoDB dark_matter")
    for mmsi in top:
        track = pos[pos.mmsi == mmsi]
        draught = static.loc[mmsi, "draught"]
        laden = bool(draught and not np.isnan(draught) and draught >= CFG["laden_draught_m"])
        color = "red" if laden else "lime"
        tip = f"{static.loc[mmsi, 'name']} | draught {draught} m | -> {static.loc[mmsi, 'dest']}"
        points = track[["lat", "lon"]].values.tolist()
        folium.PolyLine(points, color=color, weight=2, tooltip=tip).add_to(fmap)
        folium.CircleMarker(points[-1], radius=3, color=color, fill=True).add_to(fmap)
    for zone, (west, south, east, north) in CHOKEPOINTS.items():
        folium.Rectangle([[south, west], [north, east]], color="orange", weight=1, tooltip=zone).add_to(fmap)
    fmap.save(path)
    print(f"Carte sauvegardée : {path}")
    return fmap
