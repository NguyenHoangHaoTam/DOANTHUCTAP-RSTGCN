import os, json, numpy as np, argparse
import matplotlib.pyplot as plt

def load_data(processed_dir):
    X = np.load(os.path.join(processed_dir, "dataset.npy"))[...,0]  # (T,N)
    meta = json.load(open(os.path.join(processed_dir, "meta.json"), "r", encoding="utf-8"))
    return X, meta

def train_val_split(X, train_ratio=0.7):
    T = X.shape[0]
    t_split = max(1, int(T*train_ratio))
    return (X[:t_split], X[t_split:]), t_split

def baseline_persistence(train, val):
    last_train = train[-1]
    preds = [last_train]
    for _ in range(1, val.shape[0]):
        preds.append(preds[-1])
    return np.stack(preds, axis=0)

def baseline_moving_average(train, val, k=3):
    k = min(k, len(train))
    ma = np.nanmean(train[-k:], axis=0)
    return np.tile(ma, (val.shape[0], 1))

def baseline_linear_trend(train, val):
    T, N = train.shape
    x  = np.arange(T, dtype=float)
    xv = np.arange(T, T+val.shape[0], dtype=float)
    preds = np.zeros_like(val)
    for i in range(N):
        y = train[:, i]
        if np.all(np.isnan(y)): 
            preds[:, i] = 0.0
            continue
        y = np.nan_to_num(y, nan=0.0)
        b1, b0 = np.polyfit(x, y, 1)
        preds[:, i] = b1*xv + b0
    preds = np.clip(preds, 0, None)
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r".\data\processed")
    ap.add_argument("--station", default="PRYJ", help="mã ga để vẽ (phải có trong meta['stations'])")
    args = ap.parse_args()

    X, meta = load_data(args.data)
    stations = meta["stations"]
    if args.station not in stations:
        raise SystemExit(f"Station '{args.station}' không có trong dataset. Có các mã: {stations}")

    i = stations.index(args.station)
    (train, val), t0 = train_val_split(X, 0.7)

    p1 = baseline_persistence(train, val)[:, i]
    p2 = baseline_moving_average(train, val, k=3)[:, i]
    p3 = baseline_linear_trend(train, val)[:, i]

    y_train = train[:, i]
    y_val   = val[:, i]

    # Vẽ
    plt.figure()
    plt.title(f"Delay (min) at {args.station}")
    plt.plot(np.arange(len(y_train)), y_train, label="train (truth)")
    plt.plot(np.arange(len(y_train), len(y_train)+len(y_val)), y_val, label="val (truth)")
    plt.plot(np.arange(len(y_train), len(y_train)+len(y_val)), p1, label="persistence")
    plt.plot(np.arange(len(y_train), len(y_train)+len(y_val)), p2, label="moving_avg_k3")
    plt.plot(np.arange(len(y_train), len(y_train)+len(y_val)), p3, label="linear_trend")
    plt.xlabel("time slot")
    plt.ylabel("minutes")
    plt.legend()
    os.makedirs(".\\figs", exist_ok=True)
    out = os.path.join(".\\figs", f"baseline_{args.station}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved figure:", out)

if __name__ == "__main__":
    main()
