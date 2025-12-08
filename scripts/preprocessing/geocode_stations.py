import argparse
import pandas as pd
import time
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def geocode_station(name: str, geolocator, retry=3):
    """Geocode một ga với retry"""
    if not name or pd.isna(name):
        return None, None
    
    query = f"{name} railway station, India"
    
    for attempt in range(retry):
        try:
            location = geolocator.geocode(query, timeout=10)
            if location:
                return location.latitude, location.longitude
            location = geolocator.geocode(f"{name}, India", timeout=10)
            if location:
                return location.latitude, location.longitude
            return None, None
        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"[WARN] Không geocode được '{name}': {e}")
            return None, None
    
    return None, None

def main(a):
    """Main function"""
    stations = pd.read_csv(a.input_csv)
    
    if "station_name" not in stations.columns:
        print("[ERR] File CSV phải có cột 'station_name'")
        return
    
    geolocator = Nominatim(user_agent="rail_delay_demo")
    
    has_lat = "lat" in stations.columns
    has_lon = "lon" in stations.columns
    
    if has_lat and has_lon:
        missing = stations[(stations["lat"].isna()) | (stations["lon"].isna()) | 
                           (stations["lat"] == 0) | (stations["lon"] == 0)]
        print(f"[INFO] Tìm thấy {len(missing)} ga chưa có tọa độ")
    else:
        stations["lat"] = None
        stations["lon"] = None
        missing = stations
        print(f"[INFO] Geocoding {len(missing)} ga...")
    
    if len(missing) == 0:
        print("[OK] Tất cả ga đã có tọa độ!")
        return
    
    success = 0
    failed = []
    
    for idx, row in missing.iterrows():
        name = row["station_name"]
        print(f"[{idx+1}/{len(missing)}] Geocoding: {name}...", end=" ")
        
        lat, lon = geocode_station(name, geolocator)
        
        if lat and lon:
            stations.at[idx, "lat"] = lat
            stations.at[idx, "lon"] = lon
            print(f"✓ ({lat:.4f}, {lon:.4f})")
            success += 1
        else:
            print("✗")
            failed.append(name)
        
        time.sleep(1.1)
    
    stations.to_csv(a.output_csv, index=False)
    
    print(f"\n✅ [DONE] Đã geocode {success}/{len(missing)} ga")
    print(f"   - Lưu vào: {a.output_csv}")
    if failed:
        print(f"   - {len(failed)} ga thất bại: {', '.join(failed[:5])}...")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Geocode tọa độ các ga từ tên ga")
    ap.add_argument("--input-csv", required=True, help="File stations.csv đầu vào")
    ap.add_argument("--output-csv", help="File output (mặc định: ghi đè input)")
    args = ap.parse_args()
    
    if not args.output_csv:
        args.output_csv = args.input_csv
    
    main(args)

