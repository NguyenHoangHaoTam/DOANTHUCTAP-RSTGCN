import torch
import torch.nn as nn
import torch.nn.functional as F

def minmax_norm(M, eps=1e-6):
    mmin = torch.min(M)
    mmax = torch.max(M)
    return (M - mmin) / (mmax - mmin + eps)

class SpatialAttentionDomain(nn.Module):
    """
    W = softmax( -λd * D_norm + λf * Freq_norm ) ⊙ A_mask  (+ self-preserve)
    """
    def __init__(self, A, D, Freq):
        super().__init__()
        A = torch.tensor(A, dtype=torch.float32)
        D = torch.tensor(D, dtype=torch.float32)
        Fq = torch.tensor(Freq, dtype=torch.float32)
        self.register_buffer("A", (A>0).float())
        self.register_buffer("D", minmax_norm(D))
        self.register_buffer("Fq", minmax_norm(Fq))
        self.lambda_d = nn.Parameter(torch.tensor(0.5))
        self.lambda_f = nn.Parameter(torch.tensor(0.5))

    def forward(self):
        lam_d = F.softplus(self.lambda_d)
        lam_f = F.softplus(self.lambda_f)
        S = (-lam_d)*self.D + lam_f*self.Fq
        mask = (self.A>0).float()
        S = S * mask + (-1e4)*(1-mask)
        W = torch.softmax(S, dim=-1)
        return W

class TemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(k,1), padding=(k//2,0))
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class GCNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.theta = nn.Linear(ch, ch, bias=False)
    def forward(self, x, W):
        B,C,T,N = x.shape
        x = x.permute(0,2,3,1)
        x = self.theta(x)
        x = x.permute(0,1,3,2)
        x = x.reshape(B*T, C, N)
        Wb = W.unsqueeze(0).repeat(B*T,1,1)
        x = torch.bmm(x, Wb.transpose(1,2))
        x = x.reshape(B,T,C,N).permute(0,2,1,3)
        return x

class Conv2DBlock(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=(k,k), padding=(k//2,k//2))
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Branch(nn.Module):
    def __init__(self, in_ch, hid, A, D, Freq):
        super().__init__()
        self.tconv = TemporalConv(in_ch, hid)
        self.satt  = SpatialAttentionDomain(A,D,Freq)
        self.gcn   = GCNBlock(hid)
        self.c2d   = Conv2DBlock(hid)
        self.head  = nn.Conv2d(hid, 1, kernel_size=1)
    def forward(self, x):
        x = self.tconv(x)
        W = self.satt()
        x = self.gcn(x, W)
        x = self.c2d(x)
        y = self.head(x)[:, :, -1, :]
        return y.squeeze(1)

class RSTGCN(nn.Module):
    def __init__(self, in_ch, hid, A, D, Freq):
        super().__init__()
        self.h = Branch(in_ch, hid, A, D, Freq)
        self.d = Branch(in_ch, hid, A, D, Freq)
        self.w = Branch(in_ch, hid, A, D, Freq)
        self.w_h = nn.Parameter(torch.tensor(1.0))
        self.w_d = nn.Parameter(torch.tensor(0.5))
        self.w_w = nn.Parameter(torch.tensor(0.5))
    def forward(self, Xh, Xd, Xw):
        yh, yd, yw = self.h(Xh), self.d(Xd), self.w(Xw)
        W_h, W_d, W_w = F.softplus(self.w_h), F.softplus(self.w_d), F.softplus(self.w_w)
        y = W_h*yh + W_d*yd + W_w*yw
        return F.relu(y)
