import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    path = r"data\templates_all\stop_times_augmented.csv"
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")

    df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce")
    df["dep_delay"] = pd.to_numeric(df["dep_delay"], errors="coerce")
    df["avg_delay"] = df[["arr_delay","dep_delay"]].mean(axis=1)

    delay_by_station = (
        df.groupby("station_code")["avg_delay"]
        .mean()
        .fillna(0)
        .sort_values(ascending=False)
    )

    print("Top 10 ga trễ nhiều nhất:")
    print(delay_by_station.head(10).round(2))

    plt.figure(figsize=(8,5))
    plt.hist(delay_by_station, bins=30, edgecolor='black', color='skyblue')
    plt.title("Phân phối độ trễ trung bình (phút)")
    plt.xlabel("Độ trễ trung bình (phút)")
    plt.ylabel("Số ga")
    plt.grid(alpha=0.3)

    os.makedirs("runs", exist_ok=True)
    out1 = r"runs\delay_histogram.png"
    plt.savefig(out1, bbox_inches="tight")
    print("[OK] Saved histogram:", out1)

    top = delay_by_station.head(15)
    plt.figure(figsize=(10,6))
    top.plot(kind="bar", color="tomato", edgecolor="black")
    plt.title("Top 15 ga có độ trễ trung bình cao nhất")
    plt.ylabel("Phút trễ trung bình")
    plt.xlabel("Mã ga")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out2 = r"runs\top_delay_stations.png"
    plt.savefig(out2, bbox_inches="tight")
    print("[OK] Saved top stations:", out2)

if __name__ == "__main__":
    main()
