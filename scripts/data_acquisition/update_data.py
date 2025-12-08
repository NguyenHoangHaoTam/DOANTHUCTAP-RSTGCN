"""
Script tự động cập nhật dữ liệu từ các link web
Hỗ trợ nhận URLs từ file text hoặc command line
"""
import argparse
import os
import sys
from datetime import datetime, date
import pandas as pd
from pathlib import Path

try:
    from .scrape_runningstatus import scrape_one, extract_train_code
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from .scrape_runningstatus import (
        scrape_one,
        extract_train_code,
    )

def read_urls_from_file(file_path: str):
    """Đọc danh sách URLs từ file text (mỗi dòng một URL)"""
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
        return urls
    except FileNotFoundError:
        print(f"[ERROR] Khong tim thay file: {file_path}")
        return []
    except Exception as e:
        print(f"[ERROR] Loi khi doc file: {e}")
        return []

def read_urls_from_input():
    """Nhận URLs từ input của người dùng (paste nhiều dòng)"""
    print("Dan cac URLs (moi dong mot URL). Nhan Enter 2 lan de ket thuc:")
    print("   (Hoac go 'done' va Enter de ket thuc)")
    urls = []
    empty_count = 0
    while True:
        try:
            line = input().strip()
            if line.lower() == 'done':
                break
            if not line:
                empty_count += 1
                if empty_count >= 2:
                    break
                continue
            empty_count = 0
            if line and not line.startswith('#'):
                urls.append(line)
        except EOFError:
            break
    return urls

def auto_detect_date_from_url(url: str):
    """Tự động phát hiện ngày từ URL (nếu có format on-YYYYMMDD)"""
    import re
    patterns = [
        r'on-(\d{4})(\d{2})(\d{2})',
        r'on-(\d{4})-(\d{2})-(\d{2})',
        r'/(\d{4})(\d{2})(\d{2})/',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            year, month, day = match.groups()
            try:
                return f"{year}-{month}-{day}"
            except:
                pass
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Auto update data from web links",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--urls-file",
        help="Path to text file containing URLs (one per line)"
    )
    parser.add_argument(
        "--urls",
        help="Comma-separated list of URLs"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - enter URLs directly from terminal"
    )
    parser.add_argument(
        "--date",
        help="Date for data (YYYY-MM-DD). Auto-detect from URL if not provided"
    )
    parser.add_argument(
        "--auto-date",
        action="store_true",
        help="Auto-detect date from URL (if URL has format on-YYYYMMDD)"
    )
    parser.add_argument(
        "--outdir",
        default="data/templates_all",
        help="Output directory (default: data/templates_all)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    urls = []
    
    if args.urls_file:
        print(f"[INFO] Doc URLs tu file: {args.urls_file}")
        urls = read_urls_from_file(args.urls_file)
        if not urls:
            print("[ERROR] Khong co URL nao trong file!")
            return
        print(f"[OK] Tim thay {len(urls)} URLs")
        
    elif args.urls:
        print("[INFO] Doc URLs tu command line")
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
        print(f"[OK] Tim thay {len(urls)} URLs")
        
    elif args.interactive:
        print("[INFO] Che do interactive")
        urls = read_urls_from_input()
        if not urls:
            print("[ERROR] Khong co URL nao duoc nhap!")
            return
        print(f"[OK] Da nhan {len(urls)} URLs")
        
    else:
        print("[ERROR] Can cung cap --urls-file, --urls, hoac --interactive")
        parser.print_help()
        return
    
    if not urls:
        print("[ERROR] Khong co URL nao de xu ly!")
        return
    
    base_date_str = args.date
    if args.auto_date or not base_date_str:
        detected_date = auto_detect_date_from_url(urls[0])
        if detected_date:
            base_date_str = detected_date
            print(f"[INFO] Tu dong phat hien ngay: {base_date_str}")
        elif not base_date_str:
            base_date_str = datetime.now().strftime("%Y-%m-%d")
            print(f"[INFO] Su dung ngay hom nay: {base_date_str}")
    
    try:
        base_date = datetime.strptime(base_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"[ERROR] Format ngay khong hop le: {base_date_str}. Can format YYYY-MM-DD")
        return
    
    os.makedirs(args.outdir, exist_ok=True)
    
    stop_path = os.path.join(args.outdir, "stop_times.csv")
    edge_path = os.path.join(args.outdir, "edges.csv")
    station_path = os.path.join(args.outdir, "stations.csv")
    
    old_stops = pd.DataFrame()
    old_edges = pd.DataFrame()
    old_stations = pd.DataFrame()
    
    if os.path.exists(stop_path):
        old_stops = pd.read_csv(stop_path)
        print(f"[INFO] Da co {len(old_stops)} dong du lieu stops cu")
    if os.path.exists(edge_path):
        old_edges = pd.read_csv(edge_path)
        print(f"[INFO] Da co {len(old_edges)} dong du lieu edges cu")
    if os.path.exists(station_path):
        old_stations = pd.read_csv(station_path)
        print(f"[INFO] Da co {len(old_stations)} dong du lieu stations cu")
    
    all_stops, all_edges, all_stations = [], [], []
    success_count = 0
    fail_count = 0
    
    print(f"\n[START] Bat dau scrape {len(urls)} URLs...\n")
    
    for i, url in enumerate(urls, 1):
        code = extract_train_code(url)
        train_id = f"{code}_{base_date_str}"
        
        url_date = auto_detect_date_from_url(url)
        if url_date:
            try:
                url_base_date = datetime.strptime(url_date, "%Y-%m-%d").date()
                train_id = f"{code}_{url_date}"
            except:
                url_base_date = base_date
        else:
            url_base_date = base_date
        
        try:
            print(f"[{i}/{len(urls)}] Scraping: {url}")
            stops, edges, stations = scrape_one(url, url_base_date, train_id)
            
            if stops is None or len(stops) == 0:
                print(f"   [WARN] Khong lay duoc du lieu tu URL nay")
                fail_count += 1
                continue
            
            all_stops.append(stops)
            all_edges.append(edges)
            all_stations.append(stations)
            success_count += 1
            
            print(f"   [OK] Thanh cong: stops={stops.shape[0]}, edges={edges.shape[0]}, stations={stations.shape[0]}")
            
            if i < len(urls):
                import time
                time.sleep(args.delay)
                
        except Exception as e:
            print(f"   [ERROR] Loi: {e}")
            fail_count += 1
            continue
    
    print(f"\n[RESULT] Ket qua: {success_count} thanh cong, {fail_count} that bai")
    
    if not all_stops:
        print("[ERROR] Khong co du lieu nao duoc lay ve. Dung.")
        return
    
    print("\n[MERGE] Dang merge du lieu...")
    new_stops = pd.concat(all_stops, ignore_index=True)
    new_edges = pd.concat(all_edges, ignore_index=True)
    new_stations = pd.concat(all_stations, ignore_index=True)
    
    if len(old_stops) > 0:
        stops_merged = pd.concat([old_stops, new_stops], ignore_index=True)
    else:
        stops_merged = new_stops
    
    if len(old_edges) > 0:
        edges_merged = pd.concat([old_edges, new_edges], ignore_index=True)
    else:
        edges_merged = new_edges
    
    if len(old_stations) > 0:
        stations_merged = pd.concat([old_stations, new_stations], ignore_index=True)
    else:
        stations_merged = new_stations
    
    print("[CLEAN] Dang xoa du lieu trung lap...")
    stops_merged.drop_duplicates(
        subset=["train_id", "station_code"],
        keep="last",
        inplace=True
    )
    edges_merged.drop_duplicates(subset=["u", "v"], inplace=True)
    stations_merged.drop_duplicates(subset=["station_code"], keep="first", inplace=True)
    
    print("[SAVE] Dang luu du lieu...")
    stops_merged.to_csv(stop_path, index=False)
    edges_merged.to_csv(edge_path, index=False)
    stations_merged.to_csv(station_path, index=False)
    
    print("\n" + "="*60)
    print("[SUCCESS] HOAN TAT!")
    print("="*60)
    print(f"Thu muc: {args.outdir}")
    print(f"stop_times.csv: {len(stops_merged)} dong (them {len(new_stops)} dong moi)")
    print(f"edges.csv     : {len(edges_merged)} dong (them {len(new_edges)} dong moi)")
    print(f"stations.csv  : {len(stations_merged)} dong (them {len(new_stations)} dong moi)")
    print("="*60)

if __name__ == "__main__":
    main()

