# scripts/bulk_scrape.py
import argparse, os, re
from datetime import datetime, date
import pandas as pd
import numpy as np

# ===== Helpers =====
def parse_time_pair(cell: str):
    if not isinstance(cell, str):
        return None, None
    s = cell.strip()
    if s == "" or s == "-" or "Source" in s or "No Delay" in s:
        return None, None
    parts = [p.strip() for p in s.split("/") if p.strip()]
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]

def to_24h(dt_str: str, base_date: date):
    if not dt_str: return None
    try:
        t = datetime.strptime(dt_str.replace(" ", ""), "%I:%M%p").time()
        return datetime.combine(base_date, t).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None

def diff_min(actual_str: str, sched_str: str):
    if not actual_str or not sched_str:
        return np.nan
    try:
        a = datetime.strptime(actual_str, "%Y-%m-%d %H:%M")
        s = datetime.strptime(sched_str, "%Y-%m-%d %H:%M")
        return (a - s).total_seconds() / 60.0
    except Exception:
        return np.nan

def station_to_code(name: str):
    code = re.sub(r"[^A-Za-z0-9]", "", (name or "")).upper()
    return code[:12]

def flatten_cols(cols):
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            s = " ".join([str(x) for x in c if x and not str(x).startswith("Unnamed")]).strip()
        else:
            s = str(c)
        flat.append(s)
    return flat

def parse_delay_min(cell):
    if not isinstance(cell, str):
        return np.nan
    s = cell.strip().lower()
    if s == "" or s == "-" or "source" in s:
        return np.nan
    if "right time" in s or "no delay" in s or "on time" in s:
        return 0.0
    h = m = 0.0
    mh = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)", s)
    mm = re.search(r"(\d+)\s*(m|min|mins|minute|minutes)", s)
    if mh: h = float(mh.group(1))
    if mm: m = float(mm.group(1))
    if (h + m) > 0:
        return h*60.0 + m
    mm2 = re.search(r"(\d+)\s*(?:min|m)?\s*late", s)
    if mm2:
        return float(mm2.group(1))
    return np.nan

def read_status_table(url: str):
    # thử header=0
    tables = pd.read_html(url, header=0)
    for t in tables:
        t = t.copy()
        t.columns = flatten_cols(t.columns)
        low = [c.lower() for c in t.columns]
        has_station = any("station" in c for c in low)
        has_arr = any(("sch" in c and "arr" in c) or ("sch/act" in c and "arr" in c) for c in low)
        has_dep = any(("sch" in c and "dep" in c) or ("sch/act" in c and "dep" in c) for c in low)
        if has_station and (has_arr or has_dep):
            return t
    # fallback
    tables = pd.read_html(url)
    for t in tables:
        t = t.copy()
        t.columns = flatten_cols(t.columns)
        low = [c.lower() for c in t.columns]
        has_station = any("station" in c for c in low)
        has_arr = any(("sch" in c and "arr" in c) or ("sch/act" in c and "arr" in c) for c in low)
        has_dep = any(("sch" in c and "dep" in c) or ("sch/act" in c and "dep" in c) for c in low)
        if has_station and (has_arr or has_dep):
            return t
    raise ValueError("Không tìm thấy bảng Station / Sch/Act Arrival / Sch/Act Departure.")

def extract_train_code(url: str):
    # ví dụ: https://runningstatus.in/status/04828 -> 04828
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else "TRAIN"

def find_delay_columns(df):
    low_map = {c.lower(): c for c in df.columns}
    arr_col = dep_col = any_col = None
    for lc, orig in low_map.items():
        if "delay" in lc or "late" in lc:
            if "arr" in lc:
                arr_col = orig
            elif "dep" in lc:
                dep_col = orig
            else:
                any_col = orig
    return arr_col, dep_col, any_col

# ===== Bulk scrape & merge =====
def scrape_one(url: str, base_date: date, train_id: str):
    tbl = read_status_table(url)

    # map cột -> station, arr, dep
    rename = {}
    for c in tbl.columns:
        lc = c.lower()
        if "station" in lc and "code" not in lc: rename[c] = "station"
        elif ("arr" in lc) and ("sch" in lc or "sch/act" in lc): rename[c] = "arr"
        elif ("dep" in lc) and ("sch" in lc or "sch/act" in lc): rename[c] = "dep"
    tbl = tbl.rename(columns=rename)
    cols = list(tbl.columns)
    if "station" not in tbl.columns: raise ValueError("Thiếu cột Station.")
    if "arr" not in tbl.columns and len(cols)>=2: tbl = tbl.rename(columns={cols[1]:"arr"})
    if "dep" not in tbl.columns and len(cols)>=3: tbl = tbl.rename(columns={cols[2]:"dep"})

    arr_delay_col, dep_delay_col, any_delay_col = find_delay_columns(tbl)

    # stop_times
    rows = []
    codes_order = []
    for _, r in tbl.iterrows():
        name = str(r["station"]).strip()
        if not name or name == "—": continue
        code = station_to_code(name)
        codes_order.append(code)

        arr_s, arr_a = parse_time_pair(str(r.get("arr","")))
        dep_s, dep_a = parse_time_pair(str(r.get("dep","")))
        if arr_s: arr_s = to_24h(arr_s, base_date)
        if arr_a: arr_a = to_24h(arr_a, base_date)
        if dep_s: dep_s = to_24h(dep_s, base_date)
        if dep_a: dep_a = to_24h(dep_a, base_date)

        d_arr = diff_min(arr_a, arr_s)
        d_dep = diff_min(dep_a, dep_s)

        if (pd.isna(d_arr)) and arr_delay_col:
            d_arr = parse_delay_min(str(r.get(arr_delay_col, "")))
        if (pd.isna(d_dep)) and dep_delay_col:
            d_dep = parse_delay_min(str(r.get(dep_delay_col, "")))
        if pd.isna(d_arr) and any_delay_col:
            d_arr = parse_delay_min(str(r.get(any_delay_col, "")))
        if pd.isna(d_dep) and any_delay_col:
            d_dep = parse_delay_min(str(r.get(any_delay_col, "")))

        rows.append({
            "train_id": train_id,
            "station_code": code,
            "arr_sched": arr_s or "",
            "arr_actual": arr_a or "",
            "dep_sched": dep_s or "",
            "dep_actual": dep_a or "",
            "arr_delay": d_arr if pd.notna(d_arr) else "",
            "dep_delay": d_dep if pd.notna(d_dep) else "",
        })

    stops = pd.DataFrame(rows)[["train_id","station_code","arr_sched","arr_actual","dep_sched","dep_actual","arr_delay","dep_delay"]]

    # edges
    edges = pd.DataFrame({"u": codes_order[:-1], "v": codes_order[1:]})

    # stations (tọa độ placeholder)
    n = len(codes_order)
    lat0, lat1 = 20.0, 30.0
    lon_fixed = 78.0
    lats = np.linspace(lat0, lat1, num=max(n,1))[:n]
    stations = pd.DataFrame({
        "station_code": codes_order,
        "station_name": [c.title() for c in codes_order],
        "lat": lats,
        "lon": [lon_fixed]*n,
        "zone": ["NA"]*n
    }).drop_duplicates("station_code")

    return stops, edges, stations

def main(a):
    base_date = datetime.strptime(a.date, "%Y-%m-%d").date()
    urls = [u.strip() for u in a.urls.split(",") if u.strip()]
    os.makedirs(a.outdir, exist_ok=True)

    all_stops, all_edges, all_st = [], [], []
    for url in urls:
        code = extract_train_code(url)
        train_id = f"{code}_{a.date}"
        try:
            s,e,st = scrape_one(url, base_date, train_id)
            if len(s)==0:
                print(f"[WARN] Không lấy được stops từ {url}")
            all_stops.append(s)
            all_edges.append(e)
            all_st.append(st)
            print(f"[OK] {url} -> stops:{s.shape} edges:{e.shape} stations:{st.shape}")
        except Exception as ex:
            print(f"[ERR] {url}: {ex}")

    if not all_stops:
        raise SystemExit("Không có dữ liệu stops nào. Dừng.")

    # merge & dedup
    stops_merged = pd.concat(all_stops, ignore_index=True)
    stops_merged.drop_duplicates(
        subset=["train_id","station_code","arr_sched","dep_sched","arr_actual","dep_actual","arr_delay","dep_delay"],
        inplace=True
    )
    stops_merged.to_csv(os.path.join(a.outdir,"stop_times.csv"), index=False)

    edges_merged = pd.concat(all_edges, ignore_index=True)
    edges_merged = edges_merged.dropna()
    edges_merged["u"] = edges_merged["u"].astype(str)
    edges_merged["v"] = edges_merged["v"].astype(str)
    edges_merged.drop_duplicates(subset=["u","v"], inplace=True)
    edges_merged.to_csv(os.path.join(a.outdir,"edges.csv"), index=False)

    st_merged = pd.concat(all_st, ignore_index=True)
    st_merged = st_merged.drop_duplicates(subset=["station_code"], keep="first")
    st_merged = st_merged[["station_code","station_name","lat","lon","zone"]]
    st_merged.to_csv(os.path.join(a.outdir,"stations.csv"), index=False)

    print("[DONE] Export to", a.outdir)
    print(" - stop_times.csv:", stops_merged.shape)
    print(" - edges.csv     :", edges_merged.shape)
    print(" - stations.csv  :", st_merged.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="Danh sách URL, phân tách bằng dấu phẩy")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--outdir", default="data/templates_all")
    main(ap.parse_args())
