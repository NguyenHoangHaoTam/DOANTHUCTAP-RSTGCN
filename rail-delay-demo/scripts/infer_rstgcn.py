# scripts/infer_rstgcn.py
import argparse, os, sys, numpy as np, torch
from torch.utils.data import DataLoader

# --- make imports robust whether called as module or as script ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

try:
    # when running: python -m scripts.infer_rstgcn
    from scripts.data_modules import load_processed, ThreeBranchDataset, train_val_split_T
except ModuleNotFoundError:
    # when running: python scripts/infer_rstgcn.py
    from data_modules import load_processed, ThreeBranchDataset, train_val_split_T

from models.rstgcn import RSTGCN

@torch.no_grad()
def main(a):
    X, A, D, F, meta = load_processed(a.data)
    T, N, Fdim = X.shape
    slot_min = int(meta.get("slot_minutes", 60))
    steps_per_day = max(1, int(round(1440 / slot_min)))
    tr_idx, va_idx = train_val_split_T(T, ratio=a.train_ratio)

    ds_va = ThreeBranchDataset(
        X, window_h=a.window, target_feat_idx=a.target,
        t_indices=va_idx, hours_per_day=steps_per_day
    )
    if len(ds_va) == 0:
        raise SystemExit("Validation set rỗng. Hãy giảm --window hoặc tăng dữ liệu.")

    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False)

    model = RSTGCN(in_ch=Fdim, hid=a.hidden, A=A, D=D, Freq=F)
    state = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    preds, trues, masks, tlist = [], [], [], []
    for b in dl_va:
        yhat = model(b["Xh"].float(), b["Xd"].float(), b["Xw"].float())
        preds.append(yhat.numpy())
        trues.append(b["y"].numpy())
        masks.append(b["y_mask"].numpy())
        tlist += b["t"]

    Yhat = np.concatenate(preds, axis=0)   # (B,N)
    Y    = np.concatenate(trues, axis=0)   # (B,N)
    M    = np.concatenate(masks, axis=0)   # (B,N)
    Tidx = np.array(tlist)                  # (B,)

    # t index -> timestamp tại t+1
    t0 = np.datetime64(meta["t_start"])
    slot = np.timedelta64(slot_min, "m")
    ts = np.array([t0 + (ti+1)*slot for ti in Tidx])
    stations = np.array(meta["stations"])

    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    import csv
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time","station_code","y_true","y_pred","abs_err"])
        for i in range(Yhat.shape[0]):
            for j in range(Yhat.shape[1]):
                if not M[i, j]:
                    continue
                y_true = float(Y[i, j]); y_pred = float(Yhat[i, j])
                w.writerow([str(ts[i]), stations[j], y_true, y_pred, abs(y_true - y_pred)])

    print("[OK] saved:", a.out_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)   # runs\...\rstgcn_best.pt
    p.add_argument("--out-csv", required=True)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--target", type=int, default=4)   # 4 = headway
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--train-ratio", type=float, default=0.7)
    main(p.parse_args())
