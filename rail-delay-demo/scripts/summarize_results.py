# scripts/summarize_results.py
import argparse, os
import pandas as pd
import numpy as np

def metrics_from_csv(path):
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    abs_err = df["abs_err"].values
    y_true = df["y_true"].values
    m = np.isfinite(abs_err) & np.isfinite(y_true)
    if not m.any():
        return None
    mae = np.mean(np.abs(abs_err[m]))
    rmse = np.sqrt(np.mean((abs_err[m]) ** 2))
    mape = np.mean(np.abs(abs_err[m]) / (np.abs(y_true[m]) + 1.0))  # +1 để tránh chia 0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def main(a):
    rows = []
    for p in a.inputs.split(","):
        p = p.strip()
        if not p: 
            continue
        if not os.path.exists(p):
            print("[WARN] not found:", p); 
            continue
        m = metrics_from_csv(p)
        if m is None:
            print("[WARN] empty or invalid:", p)
            continue
        name = os.path.splitext(os.path.basename(p))[0]
        rows.append({"name": name, "csv": p, **m})

    if not rows:
        print("No valid input CSVs.")
        return

    out = pd.DataFrame(rows).sort_values("MAE")
    print("\n===== Summary (lower is better) =====")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if a.out_csv:
        os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
        out.to_csv(a.out_csv, index=False)
        print("\n[OK] saved:", a.out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, help="comma-separated paths to prediction CSVs")
    p.add_argument("--out-csv", default=None)
    main(p.parse_args())
