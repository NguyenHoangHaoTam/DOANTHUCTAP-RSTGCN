import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

def load_processed(data_dir):
    X = np.load(os.path.join(data_dir,"dataset.npy"))   # (T,N,5)
    A = np.load(os.path.join(data_dir,"adj.npy"))
    D = np.load(os.path.join(data_dir,"dist.npy"))
    F = np.load(os.path.join(data_dir,"freq.npy"))
    with open(os.path.join(data_dir,"meta.json"),"r",encoding="utf-8") as f:
        meta = json.load(f)
    return X, A, D, F, meta

def train_val_split_T(T, ratio=0.7):
    t = int(T*ratio)
    return list(range(t)), list(range(t, T))

class ThreeBranchDataset(Dataset):
    """
    3 nhánh thời gian:
      - recent:  window_h bước ngay trước t
      - daily:   cùng khung giờ ngày trước (offset = steps_per_day)  → AUTO: tắt nếu T không đủ
      - weekly:  cùng khung giờ tuần trước (offset = steps_per_day*7) → AUTO: tắt nếu T không đủ
    Target: avg_arr_delay (feature 0) tại t+1
    """
    def __init__(self, X, window_h=4, target_feat_idx=0, t_indices=None, hours_per_day=24):
        super().__init__()
        self.X = X.astype(np.float32)      # (T,N,F)
        self.T, self.N, self.F = X.shape
        self.Wh = int(window_h)
        # Ở đây 'hours_per_day' thực chất là "số bước trong 1 ngày" (đã tính trước ở train.py)
        self.D1 = int(hours_per_day)           # steps per day (vd slot=60 → 24; slot=15 → 96)
        self.W1 = int(self.D1 * 7)
        self.fy = int(target_feat_idx)

        # AUTO: nếu không đủ dữ liệu thì tắt daily/weekly
        self.has_daily  = (self.T - 1) - (self.D1 + self.Wh) >= 0
        self.has_weekly = (self.T - 1) - (self.W1 + self.Wh) >= 0

        # Chỉ cần đủ cho recent là tạo sample được
        min_t = self.Wh
        full = np.arange(min_t, self.T - 1)   # dự báo tại t+1
        self.idx = full if t_indices is None else full[np.isin(full, t_indices)]

    def __len__(self): 
        return int(len(self.idx))

    def __getitem__(self, k):
        t = int(self.idx[k])

        # recent window
        Xh = self.X[t - self.Wh : t, :, :]                 # (Wh,N,F)

        # daily branch
        if self.has_daily:
            idx_d = np.array([t - self.D1 - s for s in range(self.Wh, 0, -1)])
            Xd = self.X[idx_d, :, :]
        else:
            Xd = np.zeros_like(Xh)

        # weekly branch
        if self.has_weekly:
            idx_w = np.array([t - self.W1 - s for s in range(self.Wh, 0, -1)])
            Xw = self.X[idx_w, :, :]
        else:
            Xw = np.zeros_like(Xh)

        y  = self.X[t+1, :, self.fy]                       # (N,)
        mask = ~np.isnan(y)

        Xh = np.nan_to_num(Xh, nan=0.0)
        Xd = np.nan_to_num(Xd, nan=0.0)
        Xw = np.nan_to_num(Xw, nan=0.0)

        # to torch: (F,T,N)
        Xh = torch.from_numpy(np.transpose(Xh, (2,0,1)))   # (F,Wh,N)
        Xd = torch.from_numpy(np.transpose(Xd, (2,0,1)))
        Xw = torch.from_numpy(np.transpose(Xw, (2,0,1)))
        y  = torch.from_numpy(np.nan_to_num(y, nan=0.0).astype(np.float32))
        m  = torch.from_numpy(mask.astype(np.bool_))
        return {"Xh": Xh, "Xd": Xd, "Xw": Xw, "y": y, "y_mask": m, "t": t}
