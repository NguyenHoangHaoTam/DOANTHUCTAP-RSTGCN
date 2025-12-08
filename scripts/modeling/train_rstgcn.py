import argparse, os, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_modules import load_processed, ThreeBranchDataset, train_val_split_T
from models.rstgcn import RSTGCN

def seed_all(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def masked_mae(p, y, m):
    return ((p - y).abs() * m).sum() / (m.sum() + 1e-8)

def masked_rmse(p, y, m):
    return torch.sqrt((((p - y) ** 2) * m).sum() / (m.sum() + 1e-8))

def masked_mape(p, y, m, eps=1.0):
    return (((p - y).abs() / (y.abs() + eps)) * m).sum() / (m.sum() + 1e-8)

def main(a):
    seed_all(7)

    X, A, D, F, meta = load_processed(a.data)
    T, N, Fdim = X.shape

    slot_min = int(meta.get("slot_minutes", 60))
    steps_per_day = max(1, int(round(1440 / slot_min)))

    tr_idx, va_idx = train_val_split_T(T, ratio=a.train_ratio)

    ds_tr = ThreeBranchDataset(
        X, window_h=a.window, target_feat_idx=a.target,
        t_indices=tr_idx, hours_per_day=steps_per_day
    )
    ds_va = ThreeBranchDataset(
        X, window_h=a.window, target_feat_idx=a.target,
        t_indices=va_idx, hours_per_day=steps_per_day
    )

    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise SystemExit(f"Dataset rỗng (train={len(ds_tr)}, val={len(ds_va)}). "
                         f"Hãy giảm --window hoặc tăng dữ liệu/giảm slot.")

    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model = RSTGCN(in_ch=Fdim, hid=a.hidden, A=A, D=D, Freq=F).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)

    os.makedirs(a.outdir, exist_ok=True)
    if a.metrics_csv:
        os.makedirs(os.path.dirname(a.metrics_csv), exist_ok=True)
        with open(a.metrics_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_MAE,val_MAE,val_RMSE,val_MAPE\n")

    best = float("inf")
    for ep in range(1, a.epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0

        for b in dl_tr:
            Xh = b["Xh"].to(device)
            Xd = b["Xd"].to(device)
            Xw = b["Xw"].to(device)
            y  = b["y"].to(device)
            m  = b["y_mask"].to(device).float()

            yhat = model(Xh, Xd, Xw)
            loss = masked_mae(yhat, y, m)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * y.size(0)
            n += y.size(0)

        tr_mae = tr_loss / max(1, n)

        model.eval()
        mae = rmse = mape = 0.0
        cnt = 0
        with torch.no_grad():
            for b in dl_va:
                Xh = b["Xh"].to(device)
                Xd = b["Xd"].to(device)
                Xw = b["Xw"].to(device)
                y  = b["y"].to(device)
                m  = b["y_mask"].to(device).float()

                yhat = model(Xh, Xd, Xw)
                B = y.size(0)
                mae  += masked_mae(yhat, y, m).item() * B
                rmse += masked_rmse(yhat, y, m).item() * B
                mape += masked_mape(yhat, y, m).item() * B
                cnt  += B

        va_mae  = mae / max(1, cnt)
        va_rmse = rmse / max(1, cnt)
        va_mape = mape / max(1, cnt)

        print(f"[Epoch {ep:03d}] train_MAE={tr_mae:.4f} | val_MAE={va_mae:.4f} RMSE={va_rmse:.4f} MAPE={va_mape:.4f}")

        if a.metrics_csv:
            with open(a.metrics_csv, "a", encoding="utf-8") as f:
                f.write(f"{ep},{tr_mae:.6f},{va_mae:.6f},{va_rmse:.6f},{va_mape:.6f}\n")

        if va_mae < best:
            best = va_mae
            torch.save(model.state_dict(), os.path.join(a.outdir, "rstgcn_best.pt"))

    with open(os.path.join(a.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "best_val_MAE": float(best),
            "target": a.target,
            "window": a.window,
            "slot_minutes": slot_min,
            "steps_per_day": steps_per_day
        }, indent=2))

    print("Done. Best val MAE:", best)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--outdir", default="runs/rstgcn_mvp")

    p.add_argument("--target", type=int, default=0,
                   help="0=avg_arr, 1=avg_dep, 2=tot_arr, 3=tot_dep, 4=headway")
    p.add_argument("--metrics-csv", default=None,
                   help="path to save epoch metrics as CSV (optional)")

    main(p.parse_args())
