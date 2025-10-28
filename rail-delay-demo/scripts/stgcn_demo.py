import os, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============== Data utils ===============
def load_processed(data_dir):
    X = np.load(os.path.join(data_dir, "dataset.npy"))[...,0].astype(np.float32)  # (T,N)
    A = np.load(os.path.join(data_dir, "adj.npy")).astype(np.float32)             # (N,N)
    meta = json.load(open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8"))
    return X, A, meta

def split_train_val(X, ratio=0.7):
    T = X.shape[0]
    t = max(1, int(T*ratio))
    return X[:t], X[t:]

def normalize_adj(A):
    # A_hat = A + I ; D^(-1/2) A_hat D^(-1/2)
    N = A.shape[0]
    A_hat = A.copy()
    A_hat[np.arange(N), np.arange(N)] += 1.0
    d = A_hat.sum(axis=1)
    d_inv_sqrt = np.power(d + 1e-8, -0.5)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

class SeqDataset(Dataset):
    def __init__(self, X, window=4, horizon=1):
        self.X = X  # (T,N)
        self.w = window
        self.h = horizon
        self.T = X.shape[0]
        self.N = X.shape[1]
        self.samples = max(0, self.T - self.w - self.h + 1)
    def __len__(self):
        return self.samples
    def __getitem__(self, i):
        # input: (w, N), target: (N,)
        x = self.X[i:i+self.w]             # (w,N)
        y = self.X[i+self.w+self.h-1]      # (N,)
        x = x[None, ...]                   # (1,w,N) -> C=1
        return torch.from_numpy(x), torch.from_numpy(y)

# =============== Model ===============
class TemporalConv(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        # dùng padding "same": k lẻ => (k//2)
        pad = (k//2, 0)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(k,1), padding=pad)
        self.act  = nn.GELU()
        self.bn   = nn.BatchNorm2d(c_out)
    def forward(self, x):
        y = self.conv(x)   # (B,C,T,N), T giữ nguyên
        y = self.act(y)
        y = self.bn(y)
        return y

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, A_hat):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A_hat), requires_grad=False)  # (N,N)
        self.theta = nn.Conv2d(c_in, c_out, kernel_size=(1,1))
        self.act = nn.GELU()
        self.bn  = nn.BatchNorm2d(c_out)
    def forward(self, x):
        # x: (B, C, T, N)
        B,C,T,N = x.shape
        x_perm = x.permute(0,2,1,3).reshape(B*T, C, N)  # (B*T, C, N)
        Ax = torch.matmul(x_perm, self.A)               # (B*T, C, N)
        Ax = Ax.reshape(B, T, C, N).permute(0,2,1,3)    # (B, C, T, N)
        y = self.theta(Ax)
        y = self.act(y)
        y = self.bn(y)
        return y

class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A_hat, k=3):
        super().__init__()
        self.t1 = TemporalConv(c_in, c_out, k=k)
        self.g  = GraphConv(c_out, c_out, A_hat)
        self.t2 = TemporalConv(c_out, c_out, k=k)
    def forward(self, x):
        y = self.t1(x)
        y = self.g(y)
        y = self.t2(y)
        return y

class STGCN(nn.Module):
    def __init__(self, A_hat, hidden=32):
        super().__init__()
        self.block1 = STGCNBlock(1, hidden, A_hat)
        self.block2 = STGCNBlock(hidden, hidden, A_hat)
        self.head_t = nn.Conv2d(hidden, 1, kernel_size=(1,1))
        self.relu_out = nn.ReLU()
    def forward(self, x):
        # x: (B,1,T,N)
        y = self.block1(x)          # (B,H,T,N)
        y = self.block2(y)          # (B,H,T,N)
        y = self.head_t(y)          # (B,1,T,N)
        y = y.squeeze(1)            # (B,T,N)
        # Global average pooling theo trục T để dự báo 1 bước: (B,N)
        y = y.mean(dim=1)           # (B,N)
        y = self.relu_out(y)        # không âm
        return y

# =============== Train/Eval ===============
def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def run(data_dir, window=4, horizon=1, hidden=32, epochs=30, lr=1e-3, batch=16, train_ratio=0.7, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    X, A, meta = load_processed(data_dir)
    X_train, X_val = split_train_val(X, train_ratio)
    if X_train.shape[0] < window + horizon:
        raise SystemExit("Chuỗi train quá ngắn; giảm --window hoặc thu thêm dữ liệu.")

    A_hat = normalize_adj(A)
    ds_tr = SeqDataset(X_train, window=window, horizon=horizon)
    ds_va = SeqDataset(np.vstack([X_train[-window:], X_val]), window=window, horizon=horizon) if X_val.shape[0]>0 else SeqDataset(X_train, window=window, horizon=horizon)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False)

    device = torch.device("cpu")
    model = STGCN(A_hat=A_hat, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE

    best_va = math.inf
    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device).float()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()*xb.size(0)
        tr_loss /= max(1, len(ds_tr))

        model.eval(); va_loss = 0.0; va_mae = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device); yb = yb.to(device).float()
                pred = model(xb)
                l = loss_fn(pred, yb).item()
                va_loss += l*xb.size(0)
                va_mae  += torch.mean(torch.abs(pred - yb)).item()*xb.size(0)
        va_loss /= max(1, len(ds_va)); va_mae /= max(1, len(ds_va))
        if va_mae < best_va: best_va = va_mae
        if ep % max(1, epochs//5) == 0 or ep==1:
            print(f"Epoch {ep:02d}/{epochs} | train_MAE={tr_loss:.4f} | val_MAE={va_mae:.4f}")

    print("\n=== Kết quả (MAE phút) ===")
    print(f"Best val MAE: {best_va:.4f}")
    print(f"Window={window}, Horizon={horizon}, Hidden={hidden}, Epochs={epochs}, LR={lr}, Batch={batch}")
    return best_va

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r".\data\processed")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    run(args.data, args.window, args.horizon, args.hidden, args.epochs, args.lr, args.batch)

if __name__ == "__main__":
    main()
