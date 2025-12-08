import argparse, json, math, os
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def parse_dt(ts):
    for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M",
                "%d/%m/%Y %H:%M:%S","%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(str(ts), fmt)
        except Exception:
            pass
    return pd.to_datetime(ts)

def diff_min(actual, sched):
    if pd.isna(actual) or pd.isna(sched):
        return np.nan
    a = parse_dt(actual); s = parse_dt(sched)
    if pd.isna(a) or pd.isna(s):
        return np.nan
    return (a - s).total_seconds() / 60.0

def main(a):
    os.makedirs(a.outdir, exist_ok=True)

    stations = pd.read_csv(a.stations)
    edges    = pd.read_csv(a.edges)
    stops    = pd.read_csv(a.stops)

    st_id = a.st_id_col or next((c for c in ["station_id","station","stop_id","id","station_code"]
                                 if c in stations.columns), None)
    lat_c = a.lat_col   or next((c for c in ["lat","latitude","y"] if c in stations.columns), None)
    lon_c = a.lon_col   or next((c for c in ["lon","lng","longitude","x"] if c in stations.columns), None)
    if not all([st_id, lat_c, lon_c]):
        raise ValueError(f"stations.csv thiếu cột. Cần station_id|station_code + lat + lon. Headers: {list(stations.columns)}")

    SIDS = stations[st_id].astype(str).tolist()
    id2idx = {sid:i for i,sid in enumerate(SIDS)}
    N = len(SIDS)

    u_c = next((c for c in ["u","src","from","source"] if c in edges.columns), None)
    v_c = next((c for c in ["v","dst","to","target"] if c in edges.columns), None)
    if not u_c or not v_c:
        raise ValueError("edges.csv cần cột u/v hoặc src/dst")
    A = np.zeros((N,N), dtype=np.float32)
    for _, r in edges.iterrows():
        u, v = str(r[u_c]), str(r[v_c])
        if u in id2idx and v in id2idx:
            i, j = id2idx[u], id2idx[v]
            A[i,j] = A[j,i] = 1.0

    coords = stations[[lat_c, lon_c]].to_numpy().astype(float)
    D = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i,j] = haversine(coords[i,0], coords[i,1], coords[j,0], coords[j,1])

    st_col   = next((c for c in ["station_id","station","stop_id","id","station_code"] if c in stops.columns), None)
    trip_col = next((c for c in ["trip_id","train_id","service_id","journey_id"] if c in stops.columns), None)
    if st_col is None or trip_col is None:
        raise ValueError(f"stop_times.csv cần station_code và train_id/trip_id. Headers: {list(stops.columns)}")

    arr_sched  = "arr_sched"  if "arr_sched"  in stops.columns else None
    arr_actual = "arr_actual" if "arr_actual" in stops.columns else None
    dep_sched  = "dep_sched"  if "dep_sched"  in stops.columns else None
    dep_actual = "dep_actual" if "dep_actual" in stops.columns else None

    arr_dly  = next((c for c in ["arr_delay","arrival_delay","arr_delay_min","arr_late"] if c in stops.columns), None)
    dep_dly  = next((c for c in ["dep_delay","departure_delay","dep_delay_min","dep_late"] if c in stops.columns), None)

    def best_ts_row(i):
        for col in [dep_actual, arr_actual, dep_sched, arr_sched]:
            if col and isinstance(stops.at[i, col], str) and stops.at[i, col].strip() != "":
                t = parse_dt(stops.at[i, col])
                if not pd.isna(t):
                    return t
        return pd.NaT
    ts_best = pd.Series({i: best_ts_row(i) for i in stops.index})

    if arr_dly:
        arr_delay_all = pd.to_numeric(stops[arr_dly], errors="coerce")
    else:
        arr_delay_all = pd.Series({
            i: diff_min(stops.at[i, arr_actual] if arr_actual else np.nan,
                        stops.at[i, arr_sched]  if arr_sched  else np.nan)
            for i in stops.index
        })

    if dep_dly:
        dep_delay_all = pd.to_numeric(stops[dep_dly], errors="coerce")
    else:
        dep_delay_all = pd.Series({
            i: diff_min(stops.at[i, dep_actual] if dep_actual else np.nan,
                        stops.at[i, dep_sched]  if dep_sched  else np.nan)
            for i in stops.index
        })

    df = pd.DataFrame(index=stops.index)
    df["station_id"] = stops[st_col].astype(str)
    df["trip_id"]    = stops[trip_col].astype(str)
    df["timestamp"]  = ts_best
    df["arr_delay"]  = arr_delay_all
    df["dep_delay"]  = dep_delay_all

    df = df[df["station_id"].isin(id2idx.keys())]
    df = df[~df["timestamp"].isna()].copy()

    slot = int(a.slot)
    df["slot_time"] = df["timestamp"].dt.floor(f"{slot}min")

    agg = df.groupby(["slot_time","station_id"]).agg(
        avg_arr_delay=("arr_delay","mean"),
        avg_dep_delay=("dep_delay","mean"),
        tot_arr_delay=("arr_delay", lambda s: np.nan_to_num(s).sum()),
        tot_dep_delay=("dep_delay", lambda s: np.nan_to_num(s).sum()),
        trips=("trip_id","nunique")
    ).reset_index()

    df_sorted = df.sort_values(["station_id","timestamp"])
    df_sorted["gap_min"] = df_sorted.groupby("station_id")["timestamp"].diff().dt.total_seconds()/60.0
    df_sorted["slot_time"] = df_sorted["timestamp"].dt.floor(f"{slot}min")
    headway = df_sorted.groupby(["slot_time","station_id"]).agg(headway=("gap_min","median")).reset_index()
    agg = agg.merge(headway, on=["slot_time","station_id"], how="left")

    agg["tot_arr_delay"] = agg["tot_arr_delay"].fillna(0.0)
    agg["tot_dep_delay"] = agg["tot_dep_delay"].fillna(0.0)
    agg["headway"] = agg["headway"].fillna(float(slot))

    if len(agg) == 0:
        raise ValueError("Không tạo được bản ghi nào sau khi gom nhóm. Kiểm tra dữ liệu stop_times.csv.")
    t_min, t_max = agg["slot_time"].min(), agg["slot_time"].max()
    times = pd.date_range(t_min, t_max, freq=f"{slot}min")
    T = len(times)
    feats = ["avg_arr_delay","avg_dep_delay","tot_arr_delay","tot_dep_delay","headway"]

    X = np.full((T, N, 5), np.nan, dtype=np.float32)
    table = agg.set_index(["slot_time","station_id"])
    for ti, tt in enumerate(times):
        for sid, ni in id2idx.items():
            if (tt, sid) in table.index:
                row = table.loc[(tt, sid), feats]
                X[ti, ni, 0] = float(row["avg_arr_delay"]) if not pd.isna(row["avg_arr_delay"]) else np.nan
                X[ti, ni, 1] = float(row["avg_dep_delay"]) if not pd.isna(row["avg_dep_delay"]) else np.nan
                X[ti, ni, 2] = float(row["tot_arr_delay"])
                X[ti, ni, 3] = float(row["tot_dep_delay"])
                X[ti, ni, 4] = float(row["headway"])

    df_trip = df.sort_values(["trip_id","timestamp"]).copy()
    df_trip["i"] = df_trip["station_id"].map(id2idx)
    pairs = Counter()
    for _, grp in df_trip.groupby("trip_id"):
        seq = grp["i"].tolist()
        for a_i, b_i in zip(seq, seq[1:]):
            if a_i is None or b_i is None: continue
            if A[a_i, b_i] > 0 or A[b_i, a_i] > 0:
                i, j = sorted((a_i, b_i))
                pairs[(i, j)] += 1
    Freq = np.zeros((N, N), dtype=np.float32)
    if pairs:
        m = max(pairs.values())
        for (i, j), c in pairs.items():
            v = c / m
            Freq[i, j] = Freq[j, i] = v

    np.save(os.path.join(a.outdir, "dataset.npy"), X)
    np.save(os.path.join(a.outdir, "adj.npy"), A.astype(np.float32))
    np.save(os.path.join(a.outdir, "dist.npy"), D)
    np.save(os.path.join(a.outdir, "freq.npy"), Freq)
    meta = {
        "slot_minutes": slot,
        "t_start": str(times[0]),
        "t_end": str(times[-1]),
        "stations": SIDS,
        "features": ["avg_arr_delay","avg_dep_delay","tot_arr_delay","tot_dep_delay","headway"],
        "shapes": {"dataset": list(X.shape), "adj": list(A.shape), "dist": list(D.shape), "freq": list(Freq.shape)}
    }
    with open(os.path.join(a.outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print(" - dataset.npy:", X.shape)
    print(" - adj.npy    :", A.shape)
    print(" - dist.npy   :", D.shape)
    print(" - freq.npy   :", Freq.shape)
    print(" - meta.json  :", a.outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stations", required=True)
    p.add_argument("--edges", required=True)
    p.add_argument("--stops", required=True)
    p.add_argument("--slot", type=int, default=60)
    p.add_argument("--outdir", default="data/processed")
    p.add_argument("--st-id-col", default=None, help="column name for station id in stations.csv")
    p.add_argument("--lat-col",   default=None, help="column name for latitude in stations.csv")
    p.add_argument("--lon-col",   default=None, help="column name for longitude in stations.csv")
    main(p.parse_args())
