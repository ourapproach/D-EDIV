# Necessary imports
import os, random, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.exceptions import DataConversionWarning

import warnings
warnings.filterwarnings("ignore")

# Mount Google Drive / Local Drive
drive.mount('/content/drive') #Replace with your actual path

# Seed and Device 
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
assert torch.cuda.is_available(), "Please enable a GPU runtime."
device = torch.device('cuda')

# Determinism/perf knobs
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

# Datasets

NSL_TRAIN = '/content/drive/MyDrive/KDDTrain+.txt'  # Replace with your actual path
UNSW_TRAIN = '/content/drive/MyDrive/UNSW_NB15_training-set.csv'  # Replace with your actual path 
UNSW_TEST  = '/content/drive/MyDrive/UNSW_NB15_testing-set.csv'  # Replace with your actual path

# Model temporary storage at edge nodes

CACHE_DIR = '/content/drive/MyDrive/ediv_client_cache' # Replace with your actual path
os.makedirs(CACHE_DIR, exist_ok=True)

def save_client_cache(client_idx: int, round_idx: int, packet: dict, meta: dict = None):
    """
    Save a deploy-ready client model snapshot (pruned/slim).
      - packet['state']: slim state_dict
      - packet['idx']: kept indices dict {'b1','b2','b3'}
    Also maintain a 'latest' pointer (overwrite each round).
    """
    payload = {
        'round': int(round_idx),
        'state': {k: v.detach().cpu() for k, v in packet['state'].items()},
        'idx':   {k: v.detach().cpu() for k, v in packet['idx'].items()},
        'meta':  (meta or {})
    }
    path_round = os.path.join(CACHE_DIR, f'client{client_idx:02d}_round{round_idx:02d}.pt')
    torch.save(payload, path_round)
    path_latest = os.path.join(CACHE_DIR, f'client{client_idx:02d}_latest.pt')
    torch.save(payload, path_latest)
    return path_round, path_latest

def load_client_cache(client_idx: int, device='cuda'):
    """
    Load most recent cached pruned model for given client (if exists).
    """
    path_latest = os.path.join(CACHE_DIR, f'client{client_idx:02d}_latest.pt')
    if not os.path.exists(path_latest):
        return None
    payload = torch.load(path_latest, map_location=device)
    return payload

# Data: NSL-KDD features

def load_nsl_kdd(train_path):
    cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
            'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
            'root_shell','su_attempted','num_root','num_file_creations','num_shells',
            'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
            'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
            'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
            'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
            'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
            'label','difficulty']
    df = pd.read_csv(train_path, names=cols)
    df = df.drop(columns=[c for c in ['difficulty'] if c in df.columns])
    df['y'] = df['label'].astype(str).str.strip().str.lower().map(lambda v: 0 if v=='normal' else 1).astype(int)

    tr, te = train_test_split(df, test_size=0.1, random_state=SEED, stratify=df['y'])
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    for c in ['num_outbound_cmds','is_host_login']:
        tr = tr.drop(columns=[c], errors='ignore'); te = te.drop(columns=[c], errors='ignore')

    cats = ['protocol_type','service','flag']
    nums = [c for c in tr.columns if c not in (['label','y']+cats)]
    maj = tr['y'].value_counts().idxmax()
    tr_bal = pd.concat([
        resample(tr[tr['y']==maj], replace=False, n_samples=len(tr[tr['y']!=maj]), random_state=SEED),
        tr[tr['y']!=maj]
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Pre-processing (NSL-KDD)
    
    try: ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError: ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    Xtr_cat = ohe.fit_transform(tr_bal[cats]) if cats else np.zeros((len(tr_bal),0))
    Xte_cat = ohe.transform(te[cats]) if cats else np.zeros((len(te),0))
    scaler_ = StandardScaler()
    Xtr_num = scaler_.fit_transform(tr_bal[nums]); Xte_num = scaler_.transform(te[nums])
    Xtr_tab = np.concatenate([Xtr_num, Xtr_cat], 1); Xte_tab = np.concatenate([Xte_num, Xte_cat], 1)
    ytr = tr_bal['y'].values.astype(np.int64); yte = te['y'].values.astype(np.int64)

    TARGET = 256
    if Xtr_tab.shape[1] > TARGET:
        pca = PCA(n_components=TARGET, random_state=SEED)
        Xtrf = pca.fit_transform(Xtr_tab); Xtef = pca.transform(Xte_tab)
    else:
        pad = TARGET - Xtr_tab.shape[1]
        Xtrf = np.hstack([Xtr_tab, np.zeros((Xtr_tab.shape[0], pad))]) if pad>0 else Xtr_tab
        pad = TARGET - Xte_tab.shape[1]
        Xtef = np.hstack([Xte_tab, np.zeros((Xte_tab.shape[0], pad))]) if pad>0 else Xte_tab
    # 16x16 image tensors
    Xtr_img = Xtrf.reshape(-1,1,16,16).astype(np.float32)
    Xte_img = Xtef.reshape(-1,1,16,16).astype(np.float32)
    return Xtr_img, ytr, Xte_img, yte

# UNSW-NB15

def load_unsw_train_test(train_csv, test_csv):
    def load(csv_path):
        df = pd.read_csv(csv_path); df.columns = [c.strip() for c in df.columns]
        if 'label' in df.columns and pd.api.types.is_numeric_dtype(df['label']):
            y = df['label'].astype(int).clip(0,1)
        elif 'label' in df.columns:
            y = df['label'].astype(str).str.strip().str.lower().map(lambda v: 0 if v in ('0','normal') else 1).fillna(1).astype(int)
        elif 'attack_cat' in df.columns:
            y = df['attack_cat'].astype(str).str.strip().str.lower().map(lambda v: 0 if v=='normal' else 1).fillna(1).astype(int)
        else:
            raise ValueError("UNSW needs 'label' or 'attack_cat'")
        df['y'] = y
        cats = [c for c in ['proto','service','state'] if c in df.columns]
        drop_high = [c for c in ['srcip','dstip','id'] if c in df.columns]
        X_cat = df[cats] if cats else pd.DataFrame(index=df.index)
        drop_cols = set(['y']+cats+drop_high)
        for m in ['label','attack_cat']:
            if m in df.columns: drop_cols.add(m)
        X_num = df.drop(columns=list(drop_cols), errors='ignore')
        for col in X_num.columns:
            if not pd.api.types.is_numeric_dtype(X_num[col]):
                X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
        X_num = X_num.replace([np.inf,-np.inf], np.nan).fillna(0.0)
        return X_num, X_cat, y.values.astype(np.int64), cats
    
    # Pre-pocessing (UNSW-NB15)
    
    Xtr_num, Xtr_cat_raw, ytr, cats = load(train_csv)
    Xte_num, Xte_cat_raw, yte, _    = load(test_csv)
    try: ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError: ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    Xtr_cat = ohe.fit_transform(Xtr_cat_raw) if len(cats)>0 else np.zeros((len(ytr),0))
    Xte_cat = ohe.transform(Xte_cat_raw) if len(cats)>0 else np.zeros((len(yte),0))
    scaler_ = StandardScaler()
    Xtr_num = scaler_.fit_transform(Xtr_num); Xte_num = scaler_.transform(Xte_num)
    Xtr_tab = np.concatenate([Xtr_num, Xtr_cat], 1); Xte_tab = np.concatenate([Xte_num, Xte_cat], 1)

    TARGET = 256
    if Xtr_tab.shape[1] > TARGET:
        pca = PCA(n_components=TARGET, random_state=SEED)
        Xtrf = pca.fit_transform(Xtr_tab); Xtef = pca.transform(Xte_tab)
    else:
        pad = TARGET - Xtr_tab.shape[1]
        Xtrf = np.hstack([Xtr_tab, np.zeros((Xtr_tab.shape[0], pad))]) if pad>0 else Xtr_tab
        pad = TARGET - Xte_tab.shape[1]
        Xtef = np.hstack([Xte_tab, np.zeros((Xte_tab.shape[0], pad))]) if pad>0 else Xte_tab
    # 16x16 image tensors
    Xtr_img = Xtrf.reshape(-1,1,16,16).astype(np.float32)
    Xte_img = Xtef.reshape(-1,1,16,16).astype(np.float32)
    return Xtr_img, ytr, Xte_img, yte

# Data loaders

def to_loader(X, y, bs=64, shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=bs, shuffle=shuffle, drop_last=False, pin_memory=True, num_workers=2)

# Reference Model

class ECALayer(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k,padding=(k-1)//2,bias=False)
        self.sig = nn.Sigmoid()
    def forward(self,x):
        y = self.avg(x); y = y.squeeze(-1).transpose(1,2)
        y = self.conv(y); y = self.sig(y)
        y = y.transpose(1,2).unsqueeze(-1)
        return x * y.expand_as(x)

class ECA_CNN_12x12_Prune(nn.Module):
    """
    Name kept for continuity; now handles 16x16 inputs (two pools -> 8x8 -> 4x4).
    """
    def __init__(self, num_classes=2, last_channels=128):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ELU(), nn.BatchNorm2d(32),
            nn.Conv2d(32,32,3,padding=1), nn.ELU(), nn.BatchNorm2d(32),
        )
        self.eca1 = ECALayer(32); self.pool1 = nn.MaxPool2d(2,2)  # 16->8
        self.b2 = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1), nn.ELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64,64,3,padding=1), nn.ELU(), nn.BatchNorm2d(64),
        )
        self.eca2 = ECALayer(64); self.pool2 = nn.MaxPool2d(2,2)  # 8->4
        self.b3 = nn.Sequential(
            nn.Conv2d(64,last_channels,3,padding=1), nn.ELU(), nn.BatchNorm2d(last_channels),
            nn.Conv2d(last_channels,last_channels,3,padding=1), nn.ELU(), nn.BatchNorm2d(last_channels),
        )
        self.eca3 = ECALayer(last_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(last_channels, num_classes)
    def forward(self,x, mask=None):
        x = self.b1(x)
        if mask is not None and mask.get('b1') is not None:
            m1 = mask['b1']; m1 = m1.view(1,-1,1,1) if m1.dim()==2 else m1; x = x * m1
        x = self.eca1(x); x = self.pool1(x)
        x = self.b2(x)
        if mask is not None and mask.get('b2') is not None:
            m2 = mask['b2']; m2 = m2.view(1,-1,1,1) if m2.dim()==2 else m2; x = x * m2
        x = self.eca2(x); x = self.pool2(x)
        x = self.b3(x)
        if mask is not None and mask.get('b3') is not None:
            m3 = mask['b3']; m3 = m3.view(1,-1,1,1) if m3.dim()==2 else m3; x = x * m3
        x = self.eca3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

# Importance & Global (guidance-only) mask

@torch.no_grad()
def channel_importance_from_weights(model: ECA_CNN_12x12_Prune):
    return {
        'b1': model.b1[3].weight.abs().mean(dim=(1,2,3)),
        'b2': model.b2[3].weight.abs().mean(dim=(1,2,3)),
        'b3': model.b3[3].weight.abs().mean(dim=(1,2,3)),
    }

def topk_mask_from_importance(importance: torch.Tensor, keep_frac: float):
    C = importance.numel(); k = max(1, int(round(C * float(np.clip(keep_frac,0.0,1.0)))))
    idx = torch.topk(importance, k=k, dim=0).indices
    mask = torch.zeros(C, device=importance.device); mask[idx] = 1.0
    return mask

def masks_from_importances(imp_dict, keep_frac):
    return {b: topk_mask_from_importance(imp_dict[b], keep_frac) for b in ['b1','b2','b3']}

# Network adapter

def gumbel_sigmoid(logits, tau=1.5, training=True):
    # Differentiable stochastic gate; deterministic sigmoid when not training
    if training:
        u = torch.rand_like(logits); g = torch.log(u+1e-8) - torch.log(1-u+1e-8)
        return torch.sigmoid((logits + g) / tau)
    return torch.sigmoid(logits)

class AdapterMultiLayer(nn.Module):
    def __init__(self, ch1=32, ch2=64, ch3=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, ch1+ch2+ch3)
        )
        self.bias = nn.Parameter(torch.zeros(ch1+ch2+ch3))
    def forward(self, scalar):
        if scalar.dim()==0: scalar = scalar.view(1)
        h = self.net(scalar.view(-1,1))
        return h + self.bias.view(1,-1)

def split_logits(logits, ch1=32, ch2=64, ch3=128):
    return logits[:, :ch1], logits[:, ch1:ch1+ch2], logits[:, ch1+ch2:]

# Latency surrogate 

# (Updated for 16x16 input; after two 2x2 pools -> 8x8 and 4x4)
w1_raw, w2_raw, w3_raw = 32*16*16, 64*8*8, 128*4*4
w_sum = w1_raw + w2_raw + w3_raw
W1, W2, W3 = w1_raw/w_sum, w2_raw/w_sum, w3_raw/w_sum

class LinearLatencyCalibrator:
    def __init__(self, min_ms: float, max_ms: float):
        self.min_ms = float(min_ms); self.max_ms = float(max_ms)
    def pred_ms(self, m1, m2, m3):
        keep_eff = W1*m1.mean() + W2*m2.mean() + W3*m3.mean()
        return m1.new_tensor(self.min_ms) + (self.max_ms - self.min_ms) * keep_eff

# Eval/utility 

@torch.no_grad()
def evaluate_acc_unpruned(model, loader, title="Eval"):
    model.eval(); ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            try:
                logits = model(xb, mask=None)
            except TypeError:
                logits = model(xb)
        ps.extend(logits.argmax(1).cpu().tolist()); ys.extend(yb.numpy().tolist())
    acc = accuracy_score(ys, ps)
    print(f"[{title}] Acc: {acc:.4f}")
    return acc

@torch.no_grad()
def recalibrate_bn(model, loader, steps=200):
    """
    Runs a few forward passes to update BatchNorm stats.
    Works for models with and without 'mask'.
    """
    was_train = model.training
    model.train()
    it = iter(loader)

    def fwd(ex):
        try:
            return model(ex, mask=None)
        except TypeError:
            return model(ex)

    for _ in range(steps):
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(loader)
            xb, _ = next(it)
        xb = xb.to(next(model.parameters()).device, non_blocking=True)
        _ = fwd(xb)

    if not was_train:
        model.eval()

@torch.no_grad()
def measure_latency_ms(model, example_x, mask=None, iters=80, warmup=40):
    """
    Times a single-image forward pass (median ms). Works for models that
    accept 'mask' and those that don't.
    """
    model.eval()
    example_x = example_x[:1]

    def fwd(ex):
        try:
            return model(ex, mask=mask)
        except TypeError:
            return model(ex)

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        _ = fwd(example_x)
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start.record()
        _ = fwd(example_x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return float(np.median(times))

def random_keep_mask(frac=0.1):
    out={}
    for k,C in [('b1',32),('b2',64),('b3',128)]:
        m = torch.zeros(C, device=device)
        idx = torch.randperm(C, device=device)[:max(1,int(round(C*frac)))]
        m[idx]=1.0; out[k]=m.view(1,-1)
    return out

# Structured-pruned Slim & export 

class ECA_CNN_12x12_Slim(nn.Module):
    """
    Name kept for continuity; works with 16x16 inputs (pools to 8x8 then 4x4).
    """
    def __init__(self, C1, C2, C3, num_classes=2, k_eca=3):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ELU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, C1, 3, padding=1), nn.ELU(), nn.BatchNorm2d(C1),
        )
        self.eca1 = ECALayer(C1, k=k_eca); self.pool1 = nn.MaxPool2d(2,2)  # 16->8
        self.b2 = nn.Sequential(
            nn.Conv2d(C1, 64, 3, padding=1), nn.ELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, C2, 3, padding=1), nn.ELU(), nn.BatchNorm2d(C2),
        )
        self.eca2 = ECALayer(C2, k=k_eca); self.pool2 = nn.MaxPool2d(2,2)  # 8->4
        self.b3 = nn.Sequential(
            nn.Conv2d(C2, 128, 3, padding=1), nn.ELU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, C3, 3, padding=1), nn.ELU(), nn.BatchNorm2d(C3),
        )
        self.eca3 = ECALayer(C3, k=k_eca)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(C3, num_classes)
    def forward(self, x):
        x = self.b1(x); x=self.eca1(x); x=self.pool1(x)
        x = self.b2(x); x=self.eca2(x); x=self.pool2(x)
        x = self.b3(x); x=self.eca3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

def _copy_bn_subset(dst_bn: nn.BatchNorm2d, src_bn: nn.BatchNorm2d, idx_keep: torch.Tensor):
    idx = idx_keep.to(dtype=torch.long, device=src_bn.weight.device)
    dst_bn.weight.data.copy_(src_bn.weight.data[idx])
    dst_bn.bias.data.copy_(src_bn.bias.data[idx])
    dst_bn.running_mean.data.copy_(src_bn.running_mean.data[idx])
    dst_bn.running_var.data.copy_(src_bn.running_var.data[idx])
    if hasattr(dst_bn, 'num_batches_tracked') and hasattr(src_bn, 'num_batches_tracked'):
        dst_bn.num_batches_tracked.data.copy_(src_bn.num_batches_tracked.data)

@torch.no_grad()
def export_pruned_model(full_model: ECA_CNN_12x12_Prune, masks_01: dict, device='cuda'):
    idx1 = torch.nonzero(masks_01['b1'].to(full_model.b1[3].weight.device), as_tuple=False).view(-1)
    idx2 = torch.nonzero(masks_01['b2'].to(full_model.b2[3].weight.device), as_tuple=False).view(-1)
    idx3 = torch.nonzero(masks_01['b3'].to(full_model.b3[3].weight.device), as_tuple=False).view(-1)
    C1, C2, C3 = int(idx1.numel()), int(idx2.numel()), int(idx3.numel())
    assert C1>=1 and C2>=1 and C3>=1

    slim = ECA_CNN_12x12_Slim(C1,C2,C3).to(device)
    # Block 1
    slim.b1[0].weight.data.copy_(full_model.b1[0].weight.data)
    if full_model.b1[0].bias is not None: slim.b1[0].bias.data.copy_(full_model.b1[0].bias.data)
    slim.b1[2].load_state_dict(full_model.b1[2].state_dict())
    slim.b1[3].weight.data.copy_(full_model.b1[3].weight.data[idx1, :, :, :])
    if full_model.b1[3].bias is not None: slim.b1[3].bias.data.copy_(full_model.b1[3].bias.data[idx1])
    _copy_bn_subset(slim.b1[5], full_model.b1[5], idx1)
    # Block 2
    slim.b2[0].weight.data.copy_(full_model.b2[0].weight.data[:, idx1, :, :])
    if full_model.b2[0].bias is not None: slim.b2[0].bias.data.copy_(full_model.b2[0].bias.data)
    slim.b2[2].load_state_dict(full_model.b2[2].state_dict())
    slim.b2[3].weight.data.copy_(full_model.b2[3].weight.data[idx2, :, :, :])
    if full_model.b2[3].bias is not None: slim.b2[3].bias.data.copy_(full_model.b2[3].bias.data[idx2])
    _copy_bn_subset(slim.b2[5], full_model.b2[5], idx2)
    # Block 3
    slim.b3[0].weight.data.copy_(full_model.b3[0].weight.data[:, idx2, :, :])
    if full_model.b3[0].bias is not None: slim.b3[0].bias.data.copy_(full_model.b3[0].bias.data)
    slim.b3[2].load_state_dict(full_model.b3[2].state_dict())
    slim.b3[3].weight.data.copy_(full_model.b3[3].weight.data[idx3, :, :, :])
    if full_model.b3[3].bias is not None: slim.b3[3].bias.data.copy_(full_model.b3[3].bias.data[idx3])
    _copy_bn_subset(slim.b3[5], full_model.b3[5], idx3)
    # FC
    slim.fc.weight.data.copy_(full_model.fc.weight.data[:, idx3])
    slim.fc.bias.data.copy_(full_model.fc.bias.data)
    return slim, (idx1, idx2, idx3)

def finetune_pruned(model_slim, loader, epochs=6, lr=5e-4):
    model_slim.train()
    opt = torch.optim.Adam(model_slim.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = model_slim(xb); loss = crit(out, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * xb.size(0); n += xb.size(0)
        print(f"[Pruned finetune] ep {ep+1:02d} loss {tot/max(n,1):.4f}")

# Budget sampler (per client) 
class BudgetSampler:
    def __init__(self, min_ms, max_ms, mode='mixture', target_ms=None, band=0.15, beta_a=0.7, beta_b=1.5):
        self.min_ms=float(min_ms); self.max_ms=float(max_ms)
        self.mode=mode; self.target_ms=(float(target_ms) if target_ms is not None else None)
        self.band=float(band); self.beta_a=float(beta_a); self.beta_b=float(beta_b)
    def _clip(self, x): return float(np.clip(x, self.min_ms, self.max_ms))
    def sample_one(self):
        if self.mode=='uniform':
            r=np.random.rand(); return self.min_ms + r*(self.max_ms-self.min_ms)
        if self.mode=='around_target' and self.target_ms is not None:
            sigma=self.band*(self.max_ms-self.min_ms); return self._clip(np.random.normal(self.target_ms, sigma))
        if self.mode=='mixture' and self.target_ms is not None:
            if np.random.rand()<0.5:
                r=np.random.rand(); return self.min_ms + r*(self.max_ms-self.min_ms)
            sigma=self.band*(self.max_ms-self.min_ms); return self._clip(np.random.normal(self.target_ms, sigma))
        if self.mode in ('beta_skew_low','beta_skew_high'):
            a,b=(self.beta_a,self.beta_b) if self.mode=='beta_skew_low' else (self.beta_b,self.beta_a)
            z=np.random.beta(a,b); return self.min_ms + z*(self.max_ms-self.min_ms)
        r=np.random.rand(); return self.min_ms + r*(self.max_ms-self.min_ms)
    def sample(self, n): return [self.sample_one() for _ in range(n)]

# Top-k Hard Thresholding for Deployment

@torch.no_grad()
def topk_hard_threshold_from_logits(logits_all, keep_frac=0.5, allowed_dict=None):
    """
    Pick the top-k channels per block from adapter logits (1 x (32+64+128)).
    If allowed_dict is given, disallowed channels get -inf before top-k.
    """
    l1, l2, l3 = split_logits(logits_all, 32, 64, 128)

    if allowed_dict is not None:
        def mask_logits(logits, allow):
            a = allow.view(1,-1).float()
            # put very negative score on disallowed
            return logits + (a-1.0)*1e9
        l1 = mask_logits(l1, allowed_dict['b1'])
        l2 = mask_logits(l2, allowed_dict['b2'])
        l3 = mask_logits(l3, allowed_dict['b3'])

    def make_mask(l, k):
        k = max(1, min(k, l.size(1)))
        idx = torch.topk(l, k=k, dim=1).indices.squeeze(0)
        mask = torch.zeros_like(l)
        mask[:, idx] = 1.0
        return mask.squeeze(0)

    k1 = max(1, int(round(32 * keep_frac)))
    k2 = max(1, int(round(64 * keep_frac)))
    k3 = max(1, int(round(128 * keep_frac)))

    m1 = make_mask(l1, k1)
    m2 = make_mask(l2, k2)
    m3 = make_mask(l3, k3)
    return {'b1': m1, 'b2': m2, 'b3': m3}, (k1, k2, k3)

# Client update: Differentiable → Top-k 

def client_update_prune_then_train_sparse_adaptive(
    client_idx,
    client_loader,
    init_state_gpu,                  # full server weights
    global_mask_keep_dict_gpu,       # guidance-only global mask (0/1)
    adapter_state=None,              # client adapter (persisted or pretrained)
    *,
    target_ms=5.0,
    calibrator=None,
    budget_sampler=None,
    adapter_epochs=4,
    finetune_epochs=6,
    lr_adapter=2e-4,
    tau_start=4.0, tau_end=0.7,
    guidance_mode='hard',
    # lambda_task is fixed to 1.0
    lambda_task=1.0,
):
    assert calibrator is not None, "Need per-client calibrator."
    assert budget_sampler is not None, "Provide per-client BudgetSampler."

    # Phase A: adapter warmup
    
    full_model = ECA_CNN_12x12_Prune(num_classes=2, last_channels=128).to(device)
    full_model.load_state_dict(init_state_gpu, strict=True)
    for p in full_model.parameters():
        p.requires_grad = False

    adapter = AdapterMultiLayer(32, 64, 128).to(device)
    if adapter_state is not None:
        adapter.load_state_dict(adapter_state, strict=True)

    opt_adapt = torch.optim.Adam(adapter.parameters(), lr=lr_adapter)
    crit = nn.CrossEntropyLoss()

    # initialize adaptive multipliers
    
    lam_lat, lam_keep, lam_l0 = 0.6, 0.2, 1e-3
    eta_lat, eta_keep, eta_l0 = 0.05, 0.02, 0.001
    tol_lat, tol_keep = 0.05, 0.03  # tolerances

    for ep in range(adapter_epochs):
        full_model.train()
        adapter.train()
        tau = tau_end + (tau_start - tau_end) * max(0.0, (1 - ep / max(1, adapter_epochs - 1)))

        # dynamic target per epoch
        target_ms_ep = budget_sampler.sample_one()

        for xb, yb in client_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_adapt.zero_grad(set_to_none=True)

            tlat = torch.tensor([target_ms_ep], device=device).float()
            l1, l2, l3 = split_logits(adapter(tlat), 32, 64, 128)
            g1 = gumbel_sigmoid(l1, tau=tau, training=True)
            g2 = gumbel_sigmoid(l2, tau=tau, training=True)
            g3 = gumbel_sigmoid(l3, tau=tau, training=True)

            if guidance_mode == 'hard':
                m1 = g1 * global_mask_keep_dict_gpu['b1'].view(1, -1)
                m2 = g2 * global_mask_keep_dict_gpu['b2'].view(1, -1)
                m3 = g3 * global_mask_keep_dict_gpu['b3'].view(1, -1)
            else:
                m1, m2, m3 = g1, g2, g3
            masks = {'b1': m1, 'b2': m2, 'b3': m3}

            with torch.amp.autocast('cuda'):
                out = full_model(xb, mask=masks)
                loss_task = crit(out, yb)

                pred_ms = calibrator.pred_ms(m1, m2, m3)
                loss_lat = ((pred_ms - tlat) / tlat).pow(2)

                keep_tgt = ((tlat - calibrator.min_ms) /
                            max(1e-6, (calibrator.max_ms - calibrator.min_ms))).clamp(0, 1)
                loss_keep = ((m1.mean() - keep_tgt).pow(2) +
                             (m2.mean() - keep_tgt).pow(2) +
                             (m3.mean() - keep_tgt).pow(2)) / 3.0

                reg_l0 = (m1.mean() + m2.mean() + m3.mean()) / 3.0

                # constraint violations
                
                v_lat = max(0.0, abs((pred_ms - tlat) / tlat).mean().item() - tol_lat)
                v_keep = max(0.0, abs(((m1.mean()+m2.mean()+m3.mean())/3.0 - keep_tgt).item()) - tol_keep)
                v_l0 = max(0.0, (pred_ms.item() / tlat.item()) - 1.0)

                # update multipliers
                
                lam_lat = max(0.0, lam_lat + eta_lat * v_lat)
                lam_keep = max(0.0, lam_keep + eta_keep * v_keep)
                lam_l0 = max(0.0, lam_l0 + eta_l0 * v_l0)

                loss = (lambda_task * loss_task +
                        lam_lat * loss_lat +
                        lam_keep * loss_keep +
                        lam_l0 * reg_l0)

            loss.backward()
            opt_adapt.step()

    # final pruning at target_ms
    
    adapter.eval()
    tlat = torch.tensor([float(target_ms)], device=device).float()
    l1, l2, l3 = split_logits(adapter(tlat), 32, 64, 128)

    m1 = torch.sigmoid(l1).squeeze(0)
    m2 = torch.sigmoid(l2).squeeze(0)
    m3 = torch.sigmoid(l3).squeeze(0)

    masks_hard = {
        'b1': (m1 > 0.3).float(),
        'b2': (m2 > 0.3).float(),
        'b3': (m3 > 0.3).float()
    }

    slim, (idx1, idx2, idx3) = export_pruned_model(full_model, masks_hard, device=device)
    keep_frac = (idx1.numel() / 32 + idx2.numel() / 64 + idx3.numel() / 128) / 3.0
    print(f"  [Client {client_idx:02d}] Kept -> b1:{idx1.numel()} "
          f"b2:{idx2.numel()} b3:{idx3.numel()} | keep≈{keep_frac:.2f}")

    # finetune pruned
    
    finetune_pruned(slim, client_loader, epochs=finetune_epochs, lr=5e-4)

    xb_ex, _ = next(iter(client_loader))
    xb_ex = xb_ex[:1].to(device)
    ms_final = measure_latency_ms(slim, xb_ex, mask=None, iters=60, warmup=30)
    print(f"  [Client {client_idx:02d}] Pruned latency after FT: {ms_final:.2f} ms "
          f"(target {target_ms:.2f} ms)")

    packet = {"state": {k: v.detach().clone().to(device) for k, v in slim.state_dict().items()}}
    packet["idx"] = {"b1": idx1.detach().clone(),
                     "b2": idx2.detach().clone(),
                     "b3": idx3.detach().clone()}
    return packet, adapter.state_dict()



# def client_update_prune_then_train_sparse_adaptive(
#     client_idx,
#     client_loader,
#     init_state_gpu,                  # full server weights
#     global_mask_keep_dict_gpu,       # guidance-only global mask (0/1)
#     adapter_state=None,              # client adapter (persisted or pretrained)
#     *,
#     target_ms=5.0,
#     calibrator=None,
#     budget_sampler=None,
#     adapter_epochs=4,
#     finetune_epochs=6,
#     lr_adapter=2e-4,
#     tau_start=4.0, tau_end=0.7,
#     guidance_mode='hard',
#     lambda_task=1.0, lambda_lat=0.6, lambda_keep=0.2, lambda_l0=1e-3,
# ):
#     assert calibrator is not None, "Need per-client calibrator."
#     assert budget_sampler is not None, "Provide per-client BudgetSampler."

#     # Phase A: adapter warmup
#     full_model = ECA_CNN_12x12_Prune(num_classes=2, last_channels=128).to(device)
#     full_model.load_state_dict(init_state_gpu, strict=True)
#     for p in full_model.parameters():
#         p.requires_grad = False

#     adapter = AdapterMultiLayer(32, 64, 128).to(device)
#     if adapter_state is not None:
#         adapter.load_state_dict(adapter_state, strict=True)

#     opt_adapt = torch.optim.Adam(adapter.parameters(), lr=lr_adapter)
#     crit = nn.CrossEntropyLoss()

#     for ep in range(adapter_epochs):
#         full_model.train()
#         adapter.train()
#         tau = tau_end + (tau_start - tau_end) * max(0.0, (1 - ep / max(1, adapter_epochs - 1)))

#         # 🔥 NEW: dynamic target_ms each epoch
#         target_ms_ep = budget_sampler.sample_one()

#         for xb, yb in client_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             opt_adapt.zero_grad(set_to_none=True)

#             tlat = torch.tensor([target_ms_ep], device=device).float()
#             l1, l2, l3 = split_logits(adapter(tlat), 32, 64, 128)
#             g1 = gumbel_sigmoid(l1, tau=tau, training=True)
#             g2 = gumbel_sigmoid(l2, tau=tau, training=True)
#             g3 = gumbel_sigmoid(l3, tau=tau, training=True)

#             if guidance_mode == 'hard':
#                 m1 = g1 * global_mask_keep_dict_gpu['b1'].view(1, -1)
#                 m2 = g2 * global_mask_keep_dict_gpu['b2'].view(1, -1)
#                 m3 = g3 * global_mask_keep_dict_gpu['b3'].view(1, -1)
#             else:
#                 m1, m2, m3 = g1, g2, g3
#             masks = {'b1': m1, 'b2': m2, 'b3': m3}

#             with torch.amp.autocast('cuda'):
#                 out = full_model(xb, mask=masks)
#                 loss_task = crit(out, yb)
#                 pred_ms = calibrator.pred_ms(m1, m2, m3)
#                 loss_lat = ((pred_ms - tlat) / tlat).pow(2)

#                 keep_tgt = ((tlat - calibrator.min_ms) /
#                             max(1e-6, (calibrator.max_ms - calibrator.min_ms))).clamp(0, 1)
#                 loss_keep = ((m1.mean() - keep_tgt).pow(2) +
#                              (m2.mean() - keep_tgt).pow(2) +
#                              (m3.mean() - keep_tgt).pow(2)) / 3.0

#                 reg_l0 = (m1.mean() + m2.mean() + m3.mean()) / 3.0

#                 loss = (lambda_task * loss_task +
#                         lambda_lat * loss_lat +
#                         lambda_keep * loss_keep +
#                         lambda_l0 * reg_l0)

#             loss.backward()
#             opt_adapt.step()

#     # Phase B: final pruning at user target_ms
#     adapter.eval()
#     tlat = torch.tensor([float(target_ms)], device=device).float()
#     l1, l2, l3 = split_logits(adapter(tlat), 32, 64, 128)

#     # Convert logits → probabilities
#     m1 = torch.sigmoid(l1).squeeze(0)
#     m2 = torch.sigmoid(l2).squeeze(0)
#     m3 = torch.sigmoid(l3).squeeze(0)

#     # Apply threshold (e.g. >0.5 means keep) #0.4 seems okay
#     masks_hard = {
#         'b1': (m1 > 0.4).float(),
#         'b2': (m2 > 0.4).float(),
#         'b3': (m3 > 0.4).float()
#     }

#     slim, (idx1, idx2, idx3) = export_pruned_model(full_model, masks_hard, device=device)
#     keep_frac = (idx1.numel() / 32 + idx2.numel() / 64 + idx3.numel() / 128) / 3.0
#     print(f"  [Client {client_idx:02d}] Kept -> b1:{idx1.numel()} "
#           f"b2:{idx2.numel()} b3:{idx3.numel()} | keep≈{keep_frac:.2f}")

#     # Phase C: finetune pruned
#     finetune_pruned(slim, client_loader, epochs=finetune_epochs, lr=5e-4)

#     # Optional: measure realized latency
#     xb_ex, _ = next(iter(client_loader))
#     xb_ex = xb_ex[:1].to(device)
#     ms_final = measure_latency_ms(slim, xb_ex, mask=None, iters=60, warmup=30)
#     print(f"  [Client {client_idx:02d}] Pruned latency after FT: {ms_final:.2f} ms "
#           f"(target {target_ms:.2f} ms)")

#     packet = {"state": {k: v.detach().clone().to(device) for k, v in slim.state_dict().items()}}
#     packet["idx"] = {
#         "b1": idx1.detach().clone(),
#         "b2": idx2.detach().clone(),
#         "b3": idx3.detach().clone()
#     }
#     return packet, adapter.state_dict()



# Server sparse aggregation 

@torch.no_grad()
def server_channelwise_aggregate_sparse(server_model, client_packets, client_sizes=None):
    base = server_model.state_dict()
    device = next(server_model.parameters()).device
    agg, wt = {}, {}
    for k, t in base.items():
        if t.dtype.is_floating_point:
            agg[k] = torch.zeros_like(t, device=device)
            wt[k]  = torch.zeros_like(t, dtype=torch.float32, device=device)

    if client_sizes is None:
        w_per_client = [1.0 / max(1, len(client_packets))] * len(client_packets)
    else:
        total = float(sum(client_sizes)); w_per_client = [float(n)/total for n in client_sizes]

    def get_st(key, st): return st[key].to(device) if key in st else None
    def add_full(key, st, w):
        tk = get_st(key, st)
        if tk is None or key not in agg: return
        agg[key] += tk * w; wt[key]  += w
    def add_rows(key, rows_idx, st, w):
        tk = get_st(key, st)
        if tk is None or key not in agg: return
        agg[key][rows_idx] += tk * w; wt[key][rows_idx] += w
    def add_cols_conv(key, cols_idx, st, w):
        tk = get_st(key, st)
        if tk is None or key not in agg: return
        agg[key][:, cols_idx, :, :] += tk * w; wt[key][:, cols_idx, :, :] += w
    def add_cols_fc(key, cols_idx, st, w):
        tk = get_st(key, st)
        if tk is None or key not in agg: return
        agg[key][:, cols_idx] += tk * w; wt[key][:, cols_idx] += w

    for (packet, w) in zip(client_packets, w_per_client):
        st = packet["state"]; idx1 = packet["idx"]["b1"].to(device)
        idx2 = packet["idx"]["b2"].to(device); idx3 = packet["idx"]["b3"].to(device)

        for key in ["b1.0.weight","b1.0.bias",
                    "b1.2.weight","b1.2.bias","b1.2.running_mean","b1.2.running_var",
                    "b2.2.weight","b2.2.bias","b2.2.running_mean","b2.2.running_var",
                    "b3.2.weight","b3.2.bias","b3.2.running_mean","b3.2.running_var",
                    "eca1.conv.weight","eca2.conv.weight","eca3.conv.weight",
                    "fc.bias"]:
            add_full(key, st, w)

        add_rows("b1.3.weight", idx1, st, w); add_rows("b1.3.bias", idx1, st, w)
        for suf in ["weight","bias","running_mean","running_var"]:
            add_rows(f"b1.5.{suf}", idx1, st, w)

        add_cols_conv("b2.0.weight", idx1, st, w); add_full("b2.0.bias", st, w)

        add_rows("b2.3.weight", idx2, st, w); add_rows("b2.3.bias", idx2, st, w)
        for suf in ["weight","bias","running_mean","running_var"]:
            add_rows(f"b2.5.{suf}", idx2, st, w)

        add_cols_conv("b3.0.weight", idx2, st, w); add_full("b3.0.bias", st, w)

        add_rows("b3.3.weight", idx3, st, w); add_rows("b3.3.bias", idx3, st, w)
        for suf in ["weight","bias","running_mean","running_var"]:
            add_rows(f"b3.5.{suf}", idx3, st, w)

        add_cols_fc("fc.weight", idx3, st, w)

    new_state = {}
    for k, base_t in base.items():
        if k in agg:
            ws = wt[k]; out = base_t.clone(); mask = ws > 0
            out[mask] = (agg[k][mask] / ws[mask]).to(out.dtype); new_state[k] = out
        else:
            new_state[k] = base_t
    server_model.load_state_dict(new_state, strict=False)

# Server pretrain on NSL-KDD 
Xtr_img, ytr, Xte_img, yte = load_nsl_kdd(NSL_TRAIN)
train_loader_nsl = to_loader(Xtr_img, ytr, bs=64, shuffle=True)
test_loader_nsl  = to_loader(Xte_img, yte, bs=256, shuffle=False)

server_model = ECA_CNN_12x12_Prune(num_classes=2, last_channels=128).to(device)
opt = torch.optim.Adam(server_model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
EPOCHS_PRE = 10
for ep in range(1, EPOCHS_PRE+1):
    server_model.train(); tot, n = 0.0, 0
    for xb, yb in train_loader_nsl:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = server_model(xb, mask=None); loss = crit(out, yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item()*xb.size(0); n += xb.size(0)
    if ep % 2 == 0: print(f"[Server pretrain] Epoch {ep:02d} Loss {tot/max(n,1):.4f}")
print("Server NSL-KDD 10% test (no pruning):")
_ = evaluate_acc_unpruned(server_model, test_loader_nsl, title="Server NSL test")

GLOBAL_KEEP_FRAC = 0.85  # guidance fraction
with torch.no_grad():
    imp0 = channel_importance_from_weights(server_model)
    global_mask = masks_from_importances(imp0, GLOBAL_KEEP_FRAC)
global_mask_gpu = {k: v.detach().clone().to(device) for k,v in global_mask.items()}
print("Global mask keep fractions (guidance only):",
      {k: float(global_mask[k].mean().item()) for k in ['b1','b2','b3']})

# UNSW as FL task 

X_unsw_tr, y_unsw_tr, X_unsw_te, y_unsw_te = load_unsw_train_test(UNSW_TRAIN, UNSW_TEST)
unsw_te_loader = to_loader(X_unsw_te, y_unsw_te, bs=256, shuffle=False)
unsw_tr_loader_full = to_loader(X_unsw_tr, y_unsw_tr, bs=256, shuffle=True)

NUM_CLIENTS = 20
idx = np.arange(len(X_unsw_tr)); np.random.shuffle(idx)
shards = np.array_split(idx, NUM_CLIENTS)
client_loaders = [to_loader(X_unsw_tr[s], y_unsw_tr[s], bs=64, shuffle=True) for s in shards]
client_sizes = [len(s) for s in shards]

# PRETRAIN ADAPTER (Server) 

def pretrain_adapter_on_server(full_model, train_loader, global_mask_keep_dict_gpu,
                               pre_epochs=60, lr=2e-4, tau_start=4.0, tau_end=0.7,
                               guidance_mode='hard',
                               min_ms=None, max_ms=None, budget_band=0.15):
    """
    Train a global adapter (backbone frozen) on a representative dataset before FL.
    Returns state_dict to initialize clients.
    """
    # freeze backbone
    
    for p in full_model.parameters(): p.requires_grad = False
    full_model.eval()

    # Calibrate latency range with a sample
    
    xb_ex, _ = next(iter(train_loader)); xb_ex = xb_ex.to(device)
    if max_ms is None:
        max_ms = measure_latency_ms(full_model, xb_ex, mask=None, iters=60, warmup=30)
    if min_ms is None:
        min_ms = measure_latency_ms(full_model, xb_ex, mask=random_keep_mask(0.05), iters=60, warmup=30)
        if (not np.isfinite(min_ms)) or (min_ms >= 0.98*max_ms): min_ms = 0.6*max_ms
    print(f"[Adapter Pretrain] calib min≈{min_ms:.2f} ms, max≈{max_ms:.2f} ms")

    # Set up adapter, optimizer, losses
    
    adapter = AdapterMultiLayer(32,64,128).to(device)
    opt_adapt = torch.optim.Adam(adapter.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # Budget sampler centered in the middle
    target_ms = 0.5*(min_ms + max_ms)
    budget_sampler = BudgetSampler(min_ms, max_ms, mode='mixture', target_ms=target_ms, band=budget_band)
    calibrator = LinearLatencyCalibrator(min_ms=min_ms, max_ms=max_ms)

    for ep in range(1, pre_epochs+1):
        tot_loss = 0.0; n_ex = 0
        tau = tau_end + (tau_start - tau_end) * max(0.0, (1 - (ep-1) / max(1, pre_epochs-1)))
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt_adapt.zero_grad(set_to_none=True)

            # sample multiple latency targets per batch 
            targets = budget_sampler.sample(n=2)
            loss_sum = 0.0
            for ms in targets:
                tlat = torch.tensor([ms], device=device).float()
                l1,l2,l3 = split_logits(adapter(tlat), 32,64,128)
                g1 = gumbel_sigmoid(l1, tau=tau, training=True)
                g2 = gumbel_sigmoid(l2, tau=tau, training=True)
                g3 = gumbel_sigmoid(l3, tau=tau, training=True)

                if guidance_mode=='hard':
                    m1 = g1 * global_mask_keep_dict_gpu['b1'].view(1,-1)
                    m2 = g2 * global_mask_keep_dict_gpu['b2'].view(1,-1)
                    m3 = g3 * global_mask_keep_dict_gpu['b3'].view(1,-1)
                else:
                    m1,m2,m3 = g1,g2,g3
                masks = {'b1':m1,'b2':m2,'b3':m3}

                with torch.amp.autocast('cuda'):
                    out = full_model(xb, mask=masks)
                    loss_task = crit(out, yb)

                    pred_ms = calibrator.pred_ms(m1, m2, m3)
                    loss_lat = ((pred_ms - tlat) / tlat).pow(2)

                    keep_tgt = ((tlat - calibrator.min_ms) /
                                max(1e-6, (calibrator.max_ms - calibrator.min_ms))).clamp(0,1)
                    loss_keep = ((m1.mean()-keep_tgt).pow(2) +
                                 (m2.mean()-keep_tgt).pow(2) +
                                 (m3.mean()-keep_tgt).pow(2)) / 3.0

                    reg_l0 = (m1.mean()+m2.mean()+m3.mean())/3.0

                    loss = loss_task + 0.6*loss_lat + 0.2*loss_keep + 1e-3*reg_l0
                loss_sum = loss_sum + loss
            loss_sum.backward(); opt_adapt.step()
            tot_loss += float(loss_sum.item()); n_ex += xb.size(0)
        if ep % 10 == 0 or ep == 1:
            print(f"[Adapter Pretrain] ep {ep:03d}/{pre_epochs} loss_sum {tot_loss/max(1,n_ex):.6f}")

    # Return pretrained adapter state
    
    return adapter.state_dict(), (min_ms, max_ms)

# Run the pretraining

pretrained_adapter_state, (pre_min_ms, pre_max_ms) = pretrain_adapter_on_server(
    full_model=server_model,
    train_loader=unsw_tr_loader_full,
    global_mask_keep_dict_gpu=global_mask_gpu,
    pre_epochs=60,            # you can adjust (e.g., 60-100 for stability)
    lr=2e-4,
    tau_start=4.0, tau_end=0.7,
    guidance_mode='hard',
    min_ms=None, max_ms=None,
    budget_band=0.15
)

# Per-client adapters (persisted) 
client_adapter_states = [pretrained_adapter_state for _ in range(NUM_CLIENTS)]

# Per-client latency calibration 
server_model.eval()
client_calibrators = []
for i, loader in enumerate(client_loaders):
    xb_ex, _ = next(iter(loader)); xb_ex = xb_ex.to(device)
    max_ms_i = measure_latency_ms(server_model, xb_ex, mask=None, iters=60, warmup=30)
    min_ms_i = measure_latency_ms(server_model, xb_ex, mask=random_keep_mask(0.05), iters=60, warmup=30)
    if (not np.isfinite(min_ms_i)) or (min_ms_i >= 0.98*max_ms_i): min_ms_i = 0.6*max_ms_i
    client_calibrators.append(LinearLatencyCalibrator(min_ms=min_ms_i, max_ms=max_ms_i))
    print(f"[Calib client {i:02d}] min≈{min_ms_i:.2f} ms, max≈{max_ms_i:.2f} ms")

# Per-client targets and budget samplers
TARGET_MS_PER_CLIENT = []
budget_samplers = []

# Introduce heterogeneity: each client gets a random keep fraction between 0.2 and 0.8

keep_fracs = np.random.uniform(0.2, 0.8, size=NUM_CLIENTS)

for i in range(NUM_CLIENTS):
    calib = client_calibrators[i]
    frac = keep_fracs[i]
    tgt = calib.min_ms + frac * (calib.max_ms - calib.min_ms)

    TARGET_MS_PER_CLIENT.append(tgt)
    budget_samplers.append(
        BudgetSampler(calib.min_ms, calib.max_ms,
                      mode='mixture', target_ms=tgt, band=0.15)
    )

    print(f"[Client {i:02d}] Hetero keep≈{frac:.2f} "
          f"Target {tgt:.2f} ms (range {calib.min_ms:.2f}-{calib.max_ms:.2f} ms)")


print("\n[Before FL] Server on UNSW test (unpruned):")
_ = evaluate_acc_unpruned(server_model, unsw_te_loader, title="UNSW test (pre-FL)")

server_state_gpu = {k: v.detach().clone().to(device) for k,v in server_model.state_dict().items()}

# Deployment: reconstruct slim model from cache 
@torch.no_grad()
def build_slim_from_cache_payload(payload, device='cuda'):
    """
    Recreate a slim CNN from cache payload for deployment at edge servers.
    """
    idx = payload['idx']
    C1, C2, C3 = int(idx['b1'].numel()), int(idx['b2'].numel()), int(idx['b3'].numel())
    model = ECA_CNN_12x12_Slim(C1, C2, C3).to(device)
    model.load_state_dict(payload['state'], strict=True)
    model.eval()
    return model

# FL loop (with per-round caching) 

ROUNDS = 20
ADAPTER_EPOCHS = 6      # reduced (light fine-tuning since we start from pretrained)
FINETUNE_EPOCHS = 20 ###25
LR_ADAPT = 2e-4
TAU_START, TAU_END = 4.0, 0.7
GUIDANCE_MODE = 'hard'

for rnd in range(1, ROUNDS+1):
    print(f"\n=== FL Round {rnd} ===")
    client_packets = []
    for i, loader in enumerate(client_loaders):
        packet, adapter_state = client_update_prune_then_train_sparse_adaptive(
            client_idx=i,
            client_loader=loader,
            init_state_gpu=server_state_gpu,
            global_mask_keep_dict_gpu=global_mask_gpu,
            adapter_state=client_adapter_states[i],
            target_ms=float(TARGET_MS_PER_CLIENT[i]),
            calibrator=client_calibrators[i],
            budget_sampler=budget_samplers[i],
            adapter_epochs=ADAPTER_EPOCHS,
            finetune_epochs=FINETUNE_EPOCHS,
            lr_adapter=LR_ADAPT,
            tau_start=TAU_START, tau_end=TAU_END,
            guidance_mode=GUIDANCE_MODE,
            lambda_task=1.0, #, lambda_lat=0.6, lambda_keep=0.2, lambda_l0=1e-3,
      )


        client_packets.append(packet); client_adapter_states[i] = adapter_state
        calib = client_calibrators[i]
        print(f" Client {i:02d} finished (range ~{calib.min_ms:.2f}-{calib.max_ms:.2f} ms, target {TARGET_MS_PER_CLIENT[i]:.2f} ms)")

        # Cache the client's deployable slim model; replace previous cache
        _path_round, _path_latest = save_client_cache(
            client_idx=i,
            round_idx=rnd,
            packet=packet,
            meta={'target_ms': float(TARGET_MS_PER_CLIENT[i]), 'round': int(rnd)}
        )
        print(f"   ↳ Cached deploy model for client {i:02d} at {_path_latest}")

    # Sparse channel-wise aggregation
    
    server_channelwise_aggregate_sparse(server_model, client_packets, client_sizes=client_sizes)
    
    # BN recal with full train
    
    recalibrate_bn(server_model, unsw_tr_loader_full, steps=200)
    
    # Eval
    
    _ = evaluate_acc_unpruned(server_model, unsw_te_loader, title=f"UNSW test (after round {rnd})")
    
    # Refresh guidance mask & server snapshot
   
    global_mask = masks_from_importances(channel_importance_from_weights(server_model), keep_frac=GLOBAL_KEEP_FRAC)
    global_mask_gpu = {k: v.detach().clone().to(device) for k,v in global_mask.items()}
    server_state_gpu = {k: v.detach().clone().to(device) for k,v in server_model.state_dict().items()}

print("\n[Final] Server on UNSW test (unpruned):")
_ = evaluate_acc_unpruned(server_model, unsw_te_loader, title="UNSW test (final)")

# Example: load cached model at deployment (unchanged, client 0) 

payload0 = load_client_cache(client_idx=0, device=device)
if payload0 is not None:
    slim0 = build_slim_from_cache_payload(payload0, device=device)
    # Use a representative example from client 0 shard if available
    if len(client_loaders) > 0:
        xb_ex, _ = next(iter(client_loaders[0]))
        ms0 = measure_latency_ms(slim0, xb_ex.to(device), mask=None, iters=40, warmup=20)
        print(f"[Client 00 cached] deploy latency ≈ {ms0:.2f} ms")
