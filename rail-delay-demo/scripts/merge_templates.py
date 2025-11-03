# scripts/merge_templates.py
import argparse, os, glob
import pandas as pd

def load_csv_safe(path, **kw):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None

def main(a):
    in_dirs = [d for d in a.inputs.split(",") if d.strip()]
    outdir = a.outdir
    os.makedirs(outdir, exist_ok=True)

    all_stops = []
    all_edges = []
    all_st = []

    for d in in_dirs:
        stops = load_csv_safe(os.path.join(d, "stop_times.csv"))
        edges = load_csv_safe(os.path.join(d, "edges.csv"))
        st    = load_csv_safe(os.path.join(d, "stations.csv"))
        if stops is not None and len(stops)>0:
            all_stops.append(stops)
        if edges is not None and len(edges)>0:
            all_edges.append(edges)
        if st is not None and len(st)>0:
            all_st.append(st)

    if not all_stops:
        raise SystemExit("Không tìm thấy stop_times.csv ở các thư mục đầu vào.")

    # --- merge stop_times ---
    stops_merged = pd.concat(all_stops, ignore_index=True)
    # đảm bảo cột đúng thứ tự
    for c in ["train_id","station_code","arr_sched","arr_actual","dep_sched","dep_actual"]:
        if c not in stops_merged.columns:
            stops_merged[c] = ""
    stops_merged = stops_merged[["train_id","station_code","arr_sched","arr_actual","dep_sched","dep_actual"]]
    stops_merged.to_csv(os.path.join(outdir, "stop_times.csv"), index=False)

    # --- merge edges (unique) ---
    if all_edges:
        edges_merged = pd.concat(all_edges, ignore_index=True)
        edges_merged = edges_merged.dropna()
        edges_merged["u"] = edges_merged["u"].astype(str)
        edges_merged["v"] = edges_merged["v"].astype(str)
        edges_merged = edges_merged.drop_duplicates(subset=["u","v"])
        edges_merged.to_csv(os.path.join(outdir, "edges.csv"), index=False)
    else:
        # vẫn có thể tạo từ chuỗi stop_times theo từng train nếu cần, nhưng tạm yêu cầu có edges từ scraper
        edges_merged = None

    # --- merge stations (keep-first theo station_code) ---
    if all_st:
        st_merged = pd.concat(all_st, ignore_index=True)
        st_merged = st_merged.drop_duplicates(subset=["station_code"], keep="first")
        # đảm bảo cột
        for c in ["station_code","station_name","lat","lon","zone"]:
            if c not in st_merged.columns:
                if c in ["lat","lon"]:
                    st_merged[c] = 0.0
                else:
                    st_merged[c] = ""
        st_merged = st_merged[["station_code","station_name","lat","lon","zone"]]
        st_merged.to_csv(os.path.join(outdir, "stations.csv"), index=False)
    else:
        st_merged = None

    print("[OK] Merged into", outdir)
    print(" - stop_times.csv:", stops_merged.shape)
    if edges_merged is not None:
        print(" - edges.csv     :", edges_merged.shape)
    if st_merged is not None:
        print(" - stations.csv  :", st_merged.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Danh sách thư mục, phân tách bằng dấu phẩy, ví dụ: data/templates/04828_... ,data/templates/12303_...")
    ap.add_argument("--outdir", default="data/templates_all")
    main(ap.parse_args())
