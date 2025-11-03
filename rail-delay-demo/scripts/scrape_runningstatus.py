# scripts/scrape_runningstatus.py
import argparse
import re
from datetime import datetime, date
import pandas as pd
import numpy as np
import os

# ---------- helpers ----------
def parse_time_pair(cell: str):
    """
    cell ví dụ: "03:11PM / 03:11PM" hoặc "Source" / "No Delay" / "-"
    Trả về (sched_str, actual_str) ở dạng *nguyên bản* (AM/PM), None nếu không có.
    """
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
    """ '03:11PM' -> 'YYYY-MM-DD HH:MM' """
    if not dt_str:
        return None
    try:
        t = datetime.strptime(dt_str.replace(" ", ""), "%I:%M%p").time()
        return datetime.combine(base_date, t).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None

def diff_min(actual_str: str, sched_str: str):
    """ (actual - scheduled) in minutes; returns np.nan if missing """
    if not actual_str or not sched_str:
        return np.nan
    try:
        a = datetime.strptime(actual_str, "%Y-%m-%d %H:%M")
        s = datetime.strptime(sched_str, "%Y-%m-%d %H:%M")
        return (a - s).total_seconds() / 60.0
    except Exception:
        return np.nan

def station_to_code(name: str):
    """ Tạo mã ga đơn giản từ tên: viết hoa, bỏ ký tự lạ, cắt ngắn. """
    if not isinstance(name, str):
        return None
    code = re.sub(r"[^A-Za-z0-9]", "", name or "").upper()
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

def read_status_table(url: str):
    """
    Lấy bảng có cột 'Station' và ít nhất một trong 'Sch/Act Arrival' / 'Sch/Act Departure'.
    Tự làm phẳng header nhiều tầng.
    """
    # thử header=0 trước
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

    # fallback: không ép header
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

    raise ValueError("Không tìm thấy bảng Station / Sch/Act Arrival / Sch/Act Departure trên trang.")

def parse_delay_min(cell):
    """
    Parse chuỗi delay -> phút.
    Hỗ trợ: '3h 27m late', '3 hr 27 min late', '15 min late', 'Right time', 'No Delay'
    Trả về float phút, hoặc np.nan nếu không nhận ra.
    """
    if not isinstance(cell, str):
        return np.nan
    s = cell.strip().lower()
    if s == "" or s == "-" or "source" in s:
        return np.nan
    if "right time" in s or "no delay" in s or "on time" in s:
        return 0.0

    # giờ
    h = 0.0
    m = 0.0
    mh = re.search(r"(\d+)\s*(h|hr|hrs|hour|hours)", s)
    mm = re.search(r"(\d+)\s*(m|min|mins|minute|minutes)", s)
    if mh:
        h = float(mh.group(1))
    if mm:
        m = float(mm.group(1))
    if (h + m) > 0:
        return h*60.0 + m

    # fallback: số phút đơn lẻ trước chữ 'late'
    mm2 = re.search(r"(\d+)\s*(?:min|m)?\s*late", s)
    if mm2:
        return float(mm2.group(1))

    return np.nan

def find_delay_columns(df):
    """
    Tìm các cột có thể chứa delay text cho arrival/departure.
    Trả về tuple: (arr_delay_col or None, dep_delay_col or None, any_delay_col or None)
    """
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

# ---------- main ----------
def main(a):
    os.makedirs(a.outdir, exist_ok=True)
    base_date = datetime.strptime(a.date, "%Y-%m-%d").date()
    train_id = a.train_id

    # 1) Đọc bảng trạng thái
    tbl = read_status_table(a.url)

    # 2) Đổi tên cột về 'station' | 'arr' | 'dep' (linh hoạt nhiều biến thể)
    rename = {}
    for c in tbl.columns:
        lc = c.lower()
        if "station" in lc and "code" not in lc:
            rename[c] = "station"
        elif ("arr" in lc) and ("sch" in lc or "sch/act" in lc):
            rename[c] = "arr"
        elif ("dep" in lc) and ("sch" in lc or "sch/act" in lc):
            rename[c] = "dep"
    tbl = tbl.rename(columns=rename)

    # nếu thiếu arr/dep, đoán theo vị trí sau 'station'
    if "station" not in tbl.columns:
        raise ValueError(f"Không xác định được cột Station. Columns: {list(tbl.columns)}")
    cols = list(tbl.columns)
    if "arr" not in tbl.columns and len(cols) >= 2:
        tbl = tbl.rename(columns={cols[1]: "arr"})
    if "dep" not in tbl.columns and len(cols) >= 3:
        tbl = tbl.rename(columns={cols[2]: "dep"})

    # cột chứa text delay (nếu có)
    arr_delay_col, dep_delay_col, any_delay_col = find_delay_columns(tbl)

    # 3) Tạo stop_times.csv
    rows = []
    for _, r in tbl.iterrows():
        name = str(r["station"]).strip()
        if name == "" or name.lower() == "nan" or name == "—":
            continue
        code = station_to_code(name)

        arr_s, arr_a = parse_time_pair(str(r.get("arr", "")))
        dep_s, dep_a = parse_time_pair(str(r.get("dep", "")))

        if arr_s: arr_s = to_24h(arr_s, base_date)
        if arr_a: arr_a = to_24h(arr_a, base_date)
        if dep_s: dep_s = to_24h(dep_s, base_date)
        if dep_a: dep_a = to_24h(dep_a, base_date)

        # tính delay: ưu tiên (actual - sched), fallback text 'Late'
        d_arr = diff_min(arr_a, arr_s)
        d_dep = diff_min(dep_a, dep_s)

        if (pd.isna(d_arr)) and arr_delay_col:
            d_arr = parse_delay_min(str(r.get(arr_delay_col, "")))
        if (pd.isna(d_dep)) and dep_delay_col:
            d_dep = parse_delay_min(str(r.get(dep_delay_col, "")))

        # nếu vẫn nan mà có cột delay chung -> dùng chung
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

    stops = pd.DataFrame(rows)
    stops = stops[["train_id","station_code","arr_sched","arr_actual","dep_sched","dep_actual","arr_delay","dep_delay"]]
    stops.to_csv(os.path.join(a.outdir, "stop_times.csv"), index=False)

    # 4) edges.csv – nối cặp ga liên tiếp
    codes = stops["station_code"].tolist()
    edges = pd.DataFrame({"u": codes[:-1], "v": codes[1:]})
    edges.to_csv(os.path.join(a.outdir, "edges.csv"), index=False)

    # 5) stations.csv – toạ độ giả lập (đủ cho demo)
    n = len(codes)
    lat0, lat1 = 20.0, 30.0
    lon_fixed = 78.0
    lats = np.linspace(lat0, lat1, num=n) if n > 0 else np.array([])
    st = pd.DataFrame({
        "station_code": codes,
        "station_name": [c.title() for c in codes],
        "lat": lats,
        "lon": [lon_fixed]*n,
        "zone": ["NA"]*n
    }).drop_duplicates("station_code")
    st.to_csv(os.path.join(a.outdir, "stations.csv"), index=False)

    print("[OK] Exported CSV to", a.outdir)
    print(" - stop_times.csv:", stops.shape)
    print(" - edges.csv     :", edges.shape)
    print(" - stations.csv  :", st.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="URL trang 'Live Train Running Status' (trang chi tiết 1 chuyến)")
    ap.add_argument("--date", required=True, help="Ngày (YYYY-MM-DD) để gắn vào giờ AM/PM")
    ap.add_argument("--train-id", required=True, help="Định danh chuyến, vd 04828_2025-11-02")
    ap.add_argument("--outdir", default="data/templates")
    main(ap.parse_args())
