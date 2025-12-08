import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data_modules import load_processed, ThreeBranchDataset, train_val_split_T
from models.rstgcn import RSTGCN

class SimpleLSTM(nn.Module):
    """Simple LSTM baseline"""
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        B, T, N, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*N, T, F)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.reshape(B, N)

class SimpleGCN(nn.Module):
    """Simple GCN without temporal branches"""
    def __init__(self, in_ch, hid, A):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32)
        self.register_buffer("A", (A > 0).float())
        self.gcn1 = nn.Linear(in_ch, hid)
        self.gcn2 = nn.Linear(hid, 1)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x = x[:, -1, :, :]
        B, N, F = x.shape
        
        x = self.gcn1(x)
        x = torch.bmm(x, self.A.unsqueeze(0).repeat(B, 1, 1))
        x = self.act(x)
        
        x = self.gcn2(x)
        x = torch.bmm(x, self.A.unsqueeze(0).repeat(B, 1, 1))
        return x.squeeze(-1)

class MeanBaseline:
    """Simple mean baseline"""
    def predict(self, y_train):
        return np.mean(y_train, axis=0, keepdims=True)

def train_baseline(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    """Train a baseline model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    best_val_mae = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if isinstance(model, SimpleLSTM):
                x = batch['Xh'].to(device)
                x = x.permute(0, 2, 3, 1)
            elif isinstance(model, SimpleGCN):
                x = batch['Xh'].to(device)
                x = x.permute(0, 2, 3, 1)
            else:
                x = batch['Xh'].to(device)
            
            y = batch['y'].to(device)
            m = batch['y_mask'].to(device).float()
            
            y_pred = model(x)
            loss = (criterion(y_pred, y) * m).sum() / (m.sum() + 1e-8)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_mae = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(model, SimpleLSTM):
                    x = batch['Xh'].to(device)
                    x = x.permute(0, 2, 3, 1)
                elif isinstance(model, SimpleGCN):
                    x = batch['Xh'].to(device)
                    x = x.permute(0, 2, 3, 1)
                else:
                    x = batch['Xh'].to(device)
                
                y = batch['y'].to(device)
                m = batch['y_mask'].to(device).float()
                
                y_pred = model(x)
                mae = ((y_pred - y).abs() * m).sum() / (m.sum() + 1e-8)
                val_mae += mae.item() * y.size(0)
                val_count += y.size(0)
        
        val_mae /= max(1, val_count)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(train_loader):.4f}, Val MAE={val_mae:.4f}")
    
    return best_val_mae

def evaluate_baseline(model, val_loader, device):
    """Evaluate a baseline model"""
    model.eval()
    preds, trues, masks = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(model, SimpleLSTM):
                x = batch['Xh'].to(device)
                x = x.permute(0, 2, 3, 1)
            elif isinstance(model, SimpleGCN):
                x = batch['Xh'].to(device)
                x = x.permute(0, 2, 3, 1)
            else:
                x = batch['Xh'].to(device)
            
            y = batch['y'].to(device)
            m = batch['y_mask'].to(device)
            
            y_pred = model(x)
            
            preds.append(y_pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            masks.append(m.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    masks = np.concatenate(masks, axis=0)
    
    mae = np.abs(preds - trues)[masks].mean()
    rmse = np.sqrt(((preds - trues) ** 2)[masks].mean())
    mape = (np.abs((preds - trues) / (trues + 1e-6)))[masks].mean() * 100
    
    return mae, rmse, mape

def main(a):
    seed_all(42)
    
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
    
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    if os.path.exists(a.rstgcn_ckpt):
        print("\n[1/4] Evaluating RSTGCN...")
        model = RSTGCN(in_ch=Fdim, hid=a.hidden, A=A, D=D, Freq=F).to(device)
        state = torch.load(a.rstgcn_ckpt, map_location=device)
        model.load_state_dict(state)
        mae, rmse, mape = evaluate_baseline(model, dl_va, device)
        results['RSTGCN'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    else:
        print(f"[WARN] RSTGCN checkpoint not found: {a.rstgcn_ckpt}")
    
    print("\n[2/4] Training Simple LSTM...")
    model_lstm = SimpleLSTM(input_dim=Fdim, hidden_dim=a.hidden).to(device)
    train_baseline(model_lstm, dl_tr, dl_va, device, epochs=a.epochs, lr=a.lr)
    mae, rmse, mape = evaluate_baseline(model_lstm, dl_va, device)
    results['Simple LSTM'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    print("\n[3/4] Training Simple GCN...")
    model_gcn = SimpleGCN(in_ch=Fdim, hid=a.hidden, A=A).to(device)
    train_baseline(model_gcn, dl_tr, dl_va, device, epochs=a.epochs, lr=a.lr)
    mae, rmse, mape = evaluate_baseline(model_gcn, dl_va, device)
    results['Simple GCN'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    print("\n[4/4] Evaluating Mean Baseline...")
    y_train = []
    for batch in dl_tr:
        y_train.append(batch['y'].numpy())
    y_train = np.concatenate(y_train, axis=0)
    mean_pred = np.mean(y_train, axis=0, keepdims=True)
    
    y_val = []
    m_val = []
    for batch in dl_va:
        y_val.append(batch['y'].numpy())
        m_val.append(batch['y_mask'].numpy())
    y_val = np.concatenate(y_val, axis=0)
    m_val = np.concatenate(m_val, axis=0)
    
    pred_mean = np.repeat(mean_pred, len(y_val), axis=0)
    mae = np.abs(pred_mean - y_val)[m_val].mean()
    rmse = np.sqrt(((pred_mean - y_val) ** 2)[m_val].mean())
    mape = (np.abs((pred_mean - y_val) / (y_val + 1e-6)))[m_val].mean() * 100
    results['Mean Baseline'] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    
    os.makedirs(a.outdir, exist_ok=True)
    df_results.to_csv(os.path.join(a.outdir, 'baseline_comparison.csv'))
    with open(os.path.join(a.outdir, 'baseline_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Results saved to: {a.outdir}")

def seed_all(s=42):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare RSTGCN with baseline models")
    p.add_argument("--data", default="data/processed")
    p.add_argument("--rstgcn-ckpt", default="runs/rstgcn_headway/rstgcn_best.pt")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--target", type=int, default=4)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--outdir", default="runs/baseline_comparison")
    main(p.parse_args())

