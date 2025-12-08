import argparse
import re
import time
from datetime import datetime, date
import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup

TIME_RE = re.compile(r'(\d{1,2}):(\d{2})\s*([AP]M)', re.I)

def extract_times_and_delay(cell: str):
    """Trích xuất scheduled/actual time và delay từ cell text"""
    if not isinstance(cell, str):
        return None, None, np.nan
    s = cell.strip()
    if not s or s == "-" or s.lower() == "nan":
        return None, None, np.nan

    times = TIME_RE.findall(s)
    sched_str = None
    actual_str = None
    
    if len(times) >= 1:
        h, m, ampm = times[0]
        sched_str = f"{h}:{m}{ampm}"
    if len(times) >= 2:
        h, m, ampm = times[1]
        actual_str = f"{h}:{m}{ampm}"

    low = s.lower()
    if "right time" in low or "no delay" in low or "on time" in low:
        fb = 0.0
    else:
        h = m = 0
        mh = re.search(r'(\d+)\s*(h|hr|hrs|hour|hours)', low)
        mm = re.search(r'(\d+)\s*(m|min|mins|minute|minutes)', low)
        if mh: h = int(mh.group(1))
        if mm: m = int(mm.group(1))
        fb = float(h*60 + m) if (h or m) else np.nan
    
    return sched_str, actual_str, fb

def to_24h(dt_str: str, base_date: date):
    """Chuyển '03:11PM' -> 'YYYY-MM-DD HH:MM'"""
    if not dt_str: return None
    try:
        t = datetime.strptime(dt_str.replace(" ", ""), "%I:%M%p").time()
        return datetime.combine(base_date, t).strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        return None

def diff_min(actual_str: str, sched_str: str):
    """Tính delay = (actual - scheduled) in minutes"""
    if not actual_str or not sched_str:
        return np.nan
    try:
        a = datetime.strptime(actual_str, "%Y-%m-%d %H:%M")
        s = datetime.strptime(sched_str, "%Y-%m-%d %H:%M")
        return (a - s).total_seconds() / 60.0
    except Exception:
        return np.nan

def station_to_code(name: str):
    """Tạo mã ga từ tên ga"""
    if not isinstance(name, str):
        return "UNKNOWN"
    code = re.sub(r"[^A-Za-z0-9]", "", name).upper()
    return code[:12] if code else "UNKNOWN"

def scrape_with_requests(url: str, retry=3):
    """Scrape với requests + BeautifulSoup, fallback về pd.read_html"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(retry):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            if tables:
                for table in tables:
                    df = pd.read_html(str(table))[0]
                    cols_lower = [str(c).lower() for c in df.columns]
                    if any("station" in c for c in cols_lower):
                        return df
            
            tables = pd.read_html(url, header=0)
            for t in tables:
                cols_lower = [str(c).lower() for c in t.columns]
                if any("station" in c for c in cols_lower):
                    return t
            
            tables = pd.read_html(url)
            for t in tables:
                cols_lower = [str(c).lower() for c in t.columns]
                if any("station" in c for c in cols_lower):
                    return t
            
            raise ValueError("Không tìm thấy bảng có cột Station")
            
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            raise e
    
    raise ValueError(f"Không thể scrape sau {retry} lần thử")

def flatten_cols(cols):
    """Làm phẳng multi-level columns"""
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            s = " ".join([str(x) for x in c if x and not str(x).startswith("Unnamed")]).strip()
        else:
            s = str(c)
        flat.append(s)
    return flat

def read_status_table(url: str):
    """Đọc bảng status từ URL"""
    try:
        tbl = scrape_with_requests(url)
    except Exception as e:
        print(f"[WARN] Requests method failed: {e}, trying pd.read_html...")
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
    
    tbl.columns = flatten_cols(tbl.columns)
    return tbl

def extract_train_code(url: str):
    """Trích xuất mã tàu từ URL"""
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else "TRAIN"

def scrape_one(url: str, base_date: date, train_id: str):
    """Scrape một URL và trả về stops, edges, stations"""
    tbl = read_status_table(url)

    rename = {}
    for c in tbl.columns:
        lc = str(c).lower()
        if "station" in lc and "code" not in lc:
            rename[c] = "station"
        elif ("arr" in lc) and ("sch" in lc or "sch/act" in lc):
            rename[c] = "arr"
        elif ("dep" in lc) and ("sch" in lc or "sch/act" in lc):
            rename[c] = "dep"
    tbl = tbl.rename(columns=rename)
    
    cols = list(tbl.columns)
    if "station" not in tbl.columns:
        raise ValueError("Thiếu cột Station.")
    if "arr" not in tbl.columns and len(cols) >= 2:
        tbl = tbl.rename(columns={cols[1]: "arr"})
    if "dep" not in tbl.columns and len(cols) >= 3:
        tbl = tbl.rename(columns={cols[2]: "dep"})

    rows = []
    codes_order = []
    for _, r in tbl.iterrows():
        name = str(r["station"]).strip()
        if not name or name == "—" or name.lower() == "nan":
            continue
        code = station_to_code(name)
        codes_order.append(code)

        as_raw = str(r.get("arr", ""))
        ds_raw = str(r.get("dep", ""))

        arr_s_raw, arr_a_raw, arr_fb = extract_times_and_delay(as_raw)
        dep_s_raw, dep_a_raw, dep_fb = extract_times_and_delay(ds_raw)

        arr_s = to_24h(arr_s_raw, base_date) if arr_s_raw else ""
        arr_a = to_24h(arr_a_raw, base_date) if arr_a_raw else ""
        dep_s = to_24h(dep_s_raw, base_date) if dep_s_raw else ""
        dep_a = to_24h(dep_a_raw, base_date) if dep_a_raw else ""

        d_arr = diff_min(arr_a, arr_s)
        if np.isnan(d_arr): d_arr = arr_fb
        d_dep = diff_min(dep_a, dep_s)
        if np.isnan(d_dep): d_dep = dep_fb

        rows.append({
            "train_id": train_id,
            "station_code": code,
            "arr_sched": arr_s or "",
            "arr_actual": arr_a or "",
            "dep_sched": dep_s or "",
            "dep_actual": dep_a or "",
            "arr_delay": d_arr if not np.isnan(d_arr) else "",
            "dep_delay": d_dep if not np.isnan(d_dep) else "",
        })

    stops = pd.DataFrame(rows)
    if len(stops) == 0:
        return None, None, None

    edges = pd.DataFrame({"u": codes_order[:-1], "v": codes_order[1:]})

    n = len(codes_order)
    lat0, lat1 = 20.0, 30.0
    lon_fixed = 78.0
    lats = np.linspace(lat0, lat1, num=max(n, 1))[:n]
    stations = pd.DataFrame({
        "station_code": codes_order,
        "station_name": [c.title() for c in codes_order],
        "lat": lats,
        "lon": [lon_fixed] * n,
        "zone": ["NA"] * n
    }).drop_duplicates("station_code")

    return stops, edges, stations

def main(a):
    """Main function - hỗ trợ cả single URL và bulk URLs"""
    base_date = datetime.strptime(a.date, "%Y-%m-%d").date()
    os.makedirs(a.outdir, exist_ok=True)

    if hasattr(a, 'urls') and a.urls:
        urls = [u.strip() for u in a.urls.split(",") if u.strip()]
    elif hasattr(a, 'url') and a.url:
        urls = [a.url]
    else:
        raise ValueError("Cần cung cấp --url hoặc --urls")

    all_stops, all_edges, all_st = [], [], []
    
    for i, url in enumerate(urls):
        code = extract_train_code(url)
        train_id = f"{code}_{a.date}" if not hasattr(a, 'train_id') or not a.train_id else a.train_id
        
        try:
            print(f"[{i+1}/{len(urls)}] Scraping {url}...")
            s, e, st = scrape_one(url, base_date, train_id)
            
            if s is None or len(s) == 0:
                print(f"[WARN] Không lấy được stops từ {url}")
                continue
                
            all_stops.append(s)
            all_edges.append(e)
            all_st.append(st)
            print(f"[OK] {url} -> stops:{s.shape} edges:{e.shape} stations:{st.shape}")
            
            if i < len(urls) - 1:
                time.sleep(1)
                
        except Exception as ex:
            print(f"[ERR] {url}: {ex}")

    if not all_stops:
        raise SystemExit("❌ Không có dữ liệu stops nào. Dừng.")

    stops_merged = pd.concat(all_stops, ignore_index=True)
    edges_merged = pd.concat(all_edges, ignore_index=True)
    st_merged = pd.concat(all_st, ignore_index=True)

    stop_path = os.path.join(a.outdir, "stop_times.csv")
    edge_path = os.path.join(a.outdir, "edges.csv")
    station_path = os.path.join(a.outdir, "stations.csv")

    if os.path.exists(stop_path):
        old = pd.read_csv(stop_path)
        stops_merged = pd.concat([old, stops_merged], ignore_index=True)
    if os.path.exists(edge_path):
        old = pd.read_csv(edge_path)
        edges_merged = pd.concat([old, edges_merged], ignore_index=True)
    if os.path.exists(station_path):
        old = pd.read_csv(station_path)
        st_merged = pd.concat([old, st_merged], ignore_index=True)

    stops_merged.drop_duplicates(
        subset=["train_id", "station_code"],
        keep="last",
        inplace=True
    )
    edges_merged.drop_duplicates(subset=["u", "v"], inplace=True)
    st_merged.drop_duplicates(subset=["station_code"], keep="first", inplace=True)

    stops_merged.to_csv(stop_path, index=False)
    edges_merged.to_csv(edge_path, index=False)
    st_merged.to_csv(station_path, index=False)

    print("\n✅ [DONE] Dữ liệu đã được lưu vào", a.outdir)
    print(f"   - stop_times.csv: {stops_merged.shape}")
    print(f"   - edges.csv     : {edges_merged.shape}")
    print(f"   - stations.csv  : {st_merged.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrape dữ liệu từ runningstatus.in")
    ap.add_argument("--url", help="URL đơn lẻ (ví dụ: https://runningstatus.in/status/05511-on-20251102)")
    ap.add_argument("--urls", help="Danh sách URL phân tách bằng dấu phẩy")
    ap.add_argument("--date", required=True, help="Ngày (YYYY-MM-DD)")
    ap.add_argument("--train-id", help="Train ID tùy chỉnh (nếu không có sẽ tự động từ URL)")
    ap.add_argument("--outdir", default="data/templates_all", help="Thư mục output")
    main(ap.parse_args())
