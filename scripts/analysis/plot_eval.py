import argparse, pandas as pd, matplotlib.pyplot as plt, os

def main(a):
    os.makedirs(os.path.dirname(a.out1), exist_ok=True)

    m = pd.read_csv(a.metrics_csv)
    m[["train_MAE","val_MAE"]].plot()
    plt.title("Learning Curve (MAE)")
    plt.xlabel("epoch"); plt.ylabel("MAE")
    plt.tight_layout(); plt.savefig(a.out1); plt.close()
    print("[OK] saved:", a.out1)

    df = pd.read_csv(a.pred_csv)
    if df.empty:
        raise SystemExit(f"[WARN] {a.pred_csv} rỗng.")

    if a.station is None:
        counts = df["station_code"].value_counts()
        st = counts.idxmax()
        cnt = int(counts.max())
        print(f"[INFO] auto station = {st} (rows={cnt})")
    else:
        st = a.station

    d2 = df[df["station_code"]==st].copy().sort_values("time")
    d2["time"] = pd.to_datetime(d2["time"])

    plt.figure()
    if len(d2) >= 2:
        plt.plot(d2["time"], d2["y_true"], label="true")
        plt.plot(d2["time"], d2["y_pred"], label="pred")
    else:
        plt.scatter(d2["time"], d2["y_true"], marker="o", label="true")
        plt.scatter(d2["time"], d2["y_pred"], marker="x", label="pred")
        if len(d2) == 1:
            yhat = float(d2["y_pred"].iloc[0])
            plt.axhline(yhat, linestyle="--", linewidth=1, label=f"pred={yhat:.2f}")

    plt.title(f"Station {st} - target over time")
    plt.xlabel("time"); plt.ylabel("minutes")
    plt.legend(); plt.tight_layout()
    plt.savefig(a.out2); plt.close()
    print("[OK] saved:", a.out2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--out1", default="runs/figs/learning_curve.png")
    ap.add_argument("--out2", default="runs/figs/station_timeseries.png")
    ap.add_argument("--station", default=None)
    main(ap.parse_args())
