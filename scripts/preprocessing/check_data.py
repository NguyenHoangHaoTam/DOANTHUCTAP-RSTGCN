"""Quick script to check data quality"""
import pandas as pd
import sys
import os

def main():
    data_dir = "data/templates_all"
    
    print("="*60)
    print("KIEM TRA DU LIEU")
    print("="*60)
    
    stop_path = os.path.join(data_dir, "stop_times.csv")
    if os.path.exists(stop_path):
        stops = pd.read_csv(stop_path)
        print(f"\n[STOP_TIMES]")
        print(f"  Total rows: {len(stops)}")
        print(f"  Columns: {list(stops.columns)}")
        print(f"  Unique train_ids: {stops['train_id'].nunique()}")
        
        dates = stops['train_id'].str.split('_').str[-1].unique()
        print(f"  Unique dates: {len(dates)} dates")
        print(f"  Sample dates: {sorted(dates)[:5]}")
        
        has_arr_delay = stops['arr_delay'].notna().sum()
        has_dep_delay = stops['dep_delay'].notna().sum()
        print(f"  Rows with arrival delay: {has_arr_delay}")
        print(f"  Rows with departure delay: {has_dep_delay}")
        
        print(f"\n  Sample rows:")
        print(stops.head(3).to_string(index=False))
    else:
        print(f"\n[ERROR] {stop_path} not found")
    
    edge_path = os.path.join(data_dir, "edges.csv")
    if os.path.exists(edge_path):
        edges = pd.read_csv(edge_path)
        print(f"\n[EDGES]")
        print(f"  Total rows: {len(edges)}")
        print(f"  Columns: {list(edges.columns)}")
        print(f"  Unique source stations (u): {edges['u'].nunique()}")
        print(f"  Unique target stations (v): {edges['v'].nunique()}")
    else:
        print(f"\n[ERROR] {edge_path} not found")
    
    station_path = os.path.join(data_dir, "stations.csv")
    if os.path.exists(station_path):
        stations = pd.read_csv(station_path)
        print(f"\n[STATIONS]")
        print(f"  Total rows: {len(stations)}")
        print(f"  Columns: {list(stations.columns)}")
        print(f"  Unique station codes: {stations['station_code'].nunique()}")
    else:
        print(f"\n[ERROR] {station_path} not found")
    
    print("\n" + "="*60)
    print("[SUCCESS] Kiem tra hoan tat!")
    print("="*60)

if __name__ == "__main__":
    main()

