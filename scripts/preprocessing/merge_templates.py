import argparse, os
import pandas as pd

def load_csv_safe(path, **kw):
    """Đọc file CSV an toàn: trả None nếu không tồn tại hoặc lỗi."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None

def main(a):
    in_dirs = [d.strip() for d in a.inputs.split(",") if d.strip()]
    outdir = a.outdir
    os.makedirs(outdir, exist_ok=True)

    all_stops, all_edges, all_st = [], [], []

    for d in in_dirs:
        stops = load_csv_safe(os.path.join(d, "stop_times.csv"))
        edges = load_csv_safe(os.path.join(d, "edges.csv"))
        st    = load_csv_safe(os.path.join(d, "stations.csv"))

        if stops is not None and len(stops) > 0:
            all_stops.append(stops)
        if edges is not None and len(edges) > 0:
            all_edges.append(edges)
        if st is not None and len(st) > 0:
            all_st.append(st)

    if not all_stops:
        raise SystemExit("❌ Không tìm thấy stop_times.csv ở các thư mục đầu vào.")

    stops_merged = pd.concat(all_stops, ignore_index=True)

    out_stop = os.path.join(outdir, "stop_times.csv")
    if os.path.exists(out_stop):
        old = pd.read_csv(out_stop)
        print(f"[INFO] Đọc dữ liệu cũ từ {out_stop}: {old.shape}")
        stops_merged = pd.concat([old, stops_merged], ignore_index=True)

    if "train_id" in stops_merged.columns and "station_code" in stops_merged.columns:
        stops_merged.drop_duplicates(subset=["train_id", "station_code"], keep="last", inplace=True)

    stops_merged.to_csv(out_stop, index=False)
    print(f"[OK] Gộp và lưu: {out_stop} ({stops_merged.shape})")

    if all_edges:
        edges_merged = pd.concat(all_edges, ignore_index=True)
        edges_merged = edges_merged.dropna()
        edges_merged["u"] = edges_merged["u"].astype(str)
        edges_merged["v"] = edges_merged["v"].astype(str)

        out_edges = os.path.join(outdir, "edges.csv")
        if os.path.exists(out_edges):
            old = pd.read_csv(out_edges)
            edges_merged = pd.concat([old, edges_merged], ignore_index=True)

        edges_merged.drop_duplicates(subset=["u", "v"], inplace=True)
        edges_merged.to_csv(out_edges, index=False)
        print(f"[OK] Gộp edges.csv: {edges_merged.shape}")
    else:
        print("[WARN] Không tìm thấy edges.csv, bỏ qua.")

    if all_st:
        st_merged = pd.concat(all_st, ignore_index=True)
        st_merged.drop_duplicates(subset=["station_code"], keep="first", inplace=True)

        for c in ["station_code", "station_name", "lat", "lon", "zone"]:
            if c not in st_merged.columns:
                if c in ["lat", "lon"]:
                    st_merged[c] = 0.0
                else:
                    st_merged[c] = ""

        out_st = os.path.join(outdir, "stations.csv")
        if os.path.exists(out_st):
            old = pd.read_csv(out_st)
            st_merged = pd.concat([old, st_merged], ignore_index=True)
            st_merged.drop_duplicates(subset=["station_code"], keep="first", inplace=True)

        st_merged = st_merged[["station_code", "station_name", "lat", "lon", "zone"]]
        st_merged.to_csv(out_st, index=False)
        print(f"[OK] Gộp stations.csv: {st_merged.shape}")
    else:
        print("[WARN] Không tìm thấy stations.csv, bỏ qua.")

    print("\n✅ Hoàn tất gộp dữ liệu vào thư mục:", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Danh sách thư mục, phân tách bằng dấu phẩy, ví dụ: data/templates/04828_... ,data/templates/12303_...")
    ap.add_argument("--outdir", default="data/templates_all")
    main(ap.parse_args())
