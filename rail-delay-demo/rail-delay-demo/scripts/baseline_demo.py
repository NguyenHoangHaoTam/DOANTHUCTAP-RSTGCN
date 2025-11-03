import json, numpy as np, pandas as pd, os, argparse

def load_data(processed_dir):
    X = np.load(os.path.join(processed_dir, "dataset.npy"))  # (T, N, 1)
    A = np.load(os.path.join(processed_dir, "adj.npy"))
    meta = json.load(open(os.path.join(processed_dir, "meta.json"), "r", encoding="utf-8"))
    # reshape to (T, N)
    X = X[..., 0].astype(float)
    return X, A, meta

def train_val_split(X, train_ratio=0.7):
    T = X.shape[0]
    t_split = max(1, int(T * train_ratio))
    return (X[:t_split], X[t_split:]), t_split

def mae(y_true, y_pred):
    return float(np.nanmean(np.abs(y_true - y_pred)))

def baseline_persistence(train, val):
    # Dự báo = giá trị time-slot trước đó (per-station)
    last_train = train[-1]  # (N,)
    preds = [last_train]    # cho slot đầu của val
    # Sau đó dùng chính dự báo trước làm đầu vào (AR(1) "naive")
    for _ in range(1, val.shape[0]):
        preds.append(preds[-1])
    return np.stack(preds, axis=0)

def baseline_moving_average(train, val, k=3):
    # Trung bình k time-slots cuối của train, rồi giữ nguyên cho toàn bộ val
    hist = train[-k:]
    ma = np.nanmean(hist, axis=0)  # (N,)
    return np.tile(ma, (val.shape[0], 1))

def baseline_linear_trend(train, val):
    # Hồi quy tuyến tính theo thời gian cho từng ga (numpy.polyfit)
    T, N = train.shape
    x = np.arange(T, dtype=float)
    preds = np.zeros_like(val)
    xv = np.arange(T, T+val.shape[0], dtype=float)
    for i in range(N):
        y = train[:, i]
        if np.all(np.isnan(y)) or np.all(y==y[0]):
            preds[:, i] = y[-1] if not np.isnan(y[-1]) else 0.0
            continue
        # thay NaN bằng 0 cho đơn giản (dữ liệu demo ít)
        y = np.nan_to_num(y, nan=0.0)
        b1, b0 = np.polyfit(x, y, 1)  # y = b1*x + b0
        preds[:, i] = b1 * xv + b0
    # Đảm bảo không âm (độ trễ âm ít hữu ích)
    preds = np.clip(preds, 0, None)
    return preds

def eval_all(X):
    (train, val), t0 = train_val_split(X, train_ratio=0.7)
    if val.shape[0] == 0:
        raise ValueError("Dữ liệu quá ngắn (T quá nhỏ) — cần ít nhất 2–3 slots.")
    res = {}
    # Persistence
    p1 = baseline_persistence(train, val)
    res["persistence_MAE"] = mae(val, p1)
    # Moving average k=3
    p2 = baseline_moving_average(train, val, k=min(3, len(train)))
    res["moving_avg_k3_MAE"] = mae(val, p2)
    # Linear trend
    p3 = baseline_linear_trend(train, val)
    res["linear_trend_MAE"] = mae(val, p3)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r".\data\processed", help="folder chứa adj.npy, dataset.npy, meta.json")
    args = ap.parse_args()
    X, A, meta = load_data(args.data)
    print("Loaded:", X.shape, "T x N =", X.shape[0], "x", X.shape[1])
    res = eval_all(X)
    print("\n=== Baseline MAE (phút) — càng thấp càng tốt ===")
    for k, v in res.items():
        print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()
