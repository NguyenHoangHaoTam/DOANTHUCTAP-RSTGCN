import os, argparse, math
import numpy as np
import pandas as pd
from datetime import datetime
from math import radians, sin, cos, asin, sqrt

def parse_time(s):
    if pd.isna(s) or str(s).strip()=="" or str(s).startswith("#"):
        return None
    return datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R*c

def build_adj(stations_df, edges_df):
    codes = stations_df["station_code"].tolist()
    idx = {c:i for i,c in enumerate(codes)}
    N = len(codes)
    A = np.zeros((N,N), dtype=np.float32)
    for _, row in edges_df.iterrows():
        u, v = row["u"], row["v"]
        if u not in idx or v not in idx:
            continue
        i, j = idx[u], idx[v]
        lat1, lon1 = stations_df.loc[i, ["lat","lon"]]
        lat2, lon2 = stations_df.loc[idx[v], ["lat","lon"]]
        d = haversine(lat1, lon1, lat2, lon2)
        w = 1.0 / max(d, 1e-3)
        A[i,j] = w
        A[j,i] = w
    for i in range(N):
        A[i,i] = 1.0
    return A, idx

def build_time_series(stations_df, stops_df, idx_map, slot_minutes=60):
    stops_df = stops_df.copy()
    for col in ["arr_sched","arr_actual","dep_sched","dep_actual"]:
        stops_df[col] = stops_df[col].apply(parse_time)

    stops_df["arr_delay_min"] = (stops_df["arr_actual"] - stops_df["arr_sched"]).dt.total_seconds()/60.0
    stops_df = stops_df.dropna(subset=["arr_sched","arr_actual","station_code"])

    tmin = stops_df["arr_sched"].min().replace(minute=0, second=0, microsecond=0)
    tmax = stops_df["arr_sched"].max().replace(minute=0, second=0, microsecond=0)
    if tmax <= tmin:
        tmax = tmin
    slot = pd.Timedelta(minutes=slot_minutes)
    all_slots = pd.date_range(start=tmin, end=tmax, freq=slot, inclusive="both")
    if len(all_slots) < 2:
        all_slots = pd.date_range(start=tmin, periods=2, freq=slot)

    N = len(stations_df)
    T = len(all_slots)
    X = np.zeros((T, N, 1), dtype=np.float32)

    stops_df["slot_idx"] = stops_df["arr_sched"].apply(lambda t: int((t - tmin).total_seconds() // (slot_minutes*60)))
    grouped = stops_df.groupby(["station_code","slot_idx"])["arr_delay_min"].mean().reset_index()

    for _, r in grouped.iterrows():
        code = r["station_code"]
        if code not in idx_map:
            continue
        i = idx_map[code]
        t = int(r["slot_idx"])
        if 0 <= t < T:
            val = float(r["arr_delay_min"]) if not math.isnan(r["arr_delay_min"]) else 0.0
            X[t, i, 0] = val

    meta = {
        "slot_minutes": slot_minutes,
        "t_start": tmin.isoformat(),
        "t_end": tmax.isoformat(),
        "stations": stations_df["station_code"].tolist()
    }
    return X, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--stops", required=True)
    ap.add_argument("--slot", type=int, default=60)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    stations_df = pd.read_csv(args.stations, comment="#")
    edges_df = pd.read_csv(args.edges, comment="#")
    stops_df = pd.read_csv(args.stops, comment="#")

    A, idx = build_adj(stations_df, edges_df)
    X, meta = build_time_series(stations_df, stops_df, idx, slot_minutes=args.slot)

    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "adj.npy"), A)
    np.save(os.path.join(args.outdir, "dataset.npy"), X)
    with open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(" -", os.path.join(args.outdir, "adj.npy"), A.shape)
    print(" -", os.path.join(args.outdir, "dataset.npy"), X.shape)
    print(" -", os.path.join(args.outdir, "meta.json"))

if __name__ == "__main__":
    main()
