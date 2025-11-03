import numpy as np
from scripts.data_modules import load_processed, ThreeBranchDataset, train_val_split_T

# 1) Thống kê từng feature
X, A, D, F, meta = load_processed("data/processed")
T, N, Fdim = X.shape
names = ["avg_arr_delay","avg_dep_delay","tot_arr_delay","tot_dep_delay","headway"]
for i, n in enumerate(names):
    v = X[:, :, i]
    nn = np.isfinite(v)
    nonzero = (np.abs(v) > 1e-6) & nn
    print(f"{i}:{n:15s}  nan%={100*(~nn).mean():.2f}  nonzero%={100*nonzero.mean():.2f}  mean={np.nanmean(v):.3f}")

# 2) Đếm nhãn (mask) của dataset train/val theo từng target
slot_min = int(meta.get("slot_minutes", 60))
steps = max(1, int(round(1440 / slot_min)))
tr, va = train_val_split_T(T, 0.7)

def count_mask(target, window, indices):
    ds = ThreeBranchDataset(X, window_h=window, target_feat_idx=target, t_indices=indices, hours_per_day=steps)
    msum = 0
    for k in range(len(ds)):
        msum += int(ds[k]["y_mask"].numpy().sum())
    return len(ds), msum

for tgt in [0,1,2,3,4]:
    lt, mt = count_mask(tgt, 2, tr)
    lv, mv = count_mask(tgt, 2, va)
    print(f"target {tgt}: train_len={lt}, train_mask_true={mt} | val_len={lv}, val_mask_true={mv}")
