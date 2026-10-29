# %% [markdown]
# # 1. Glucose forecaster (GRU) with XGBoost baseline
# Put per-patient CGM files (CSV/XLSX) in `/content/cgm`, e.g. ShanghaiT1DM / ShanghaiT2DM (Zhao et al., 2023).
# Each file needs a date-time column and a CGM column (name containing "cgm" or "glucose").
# Output: `glucose_gru.pt` -> copy into `backend/weights/`.

# %%
#!pip install -q torch pandas numpy scikit-learn xgboost openpyxl

# %%
import glob, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR, OUT_DIR = "/content/cgm", Path("/content/weights")
OUT_DIR.mkdir(exist_ok=True)
WINDOW, HORIZONS = 24, (6, 12)          # keep identical to backend/app/modules/glucose.py
dev = "cuda" if torch.cuda.is_available() else "cpu"

def make_features(g, minutes_of_day):   # keep identical to backend
    g = np.asarray(g, dtype=np.float32)
    d = np.diff(g, prepend=g[0])
    ang = 2 * np.pi * np.asarray(minutes_of_day, dtype=np.float32) / 1440
    return np.stack([g / 400, (g - g[-1]) / 100, d / 20, np.sin(ang), np.cos(ang)], axis=1)

class GlucoseGRU(nn.Module):
    def __init__(self, n_in=5, hidden=64, layers=2, n_out=len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_out))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])

# %% [markdown]
# ## Load, clean, resample
# 5-min grid, sensor range 40-400 mg/dL, gaps up to 30 min interpolated, longer gaps left empty (windows touching them are dropped).

# %%
def load_patient(path):
    df = pd.read_excel(path) if path.endswith(("xlsx", "xls")) else pd.read_csv(path)
    tcol = next(c for c in df.columns if "date" in c.lower() or "time" in c.lower())
    gcol = next(c for c in df.columns if "cgm" in c.lower() or "glucose" in c.lower())
    s = pd.Series(pd.to_numeric(df[gcol], errors="coerce").values, index=pd.to_datetime(df[tcol], errors="coerce"))
    s = s[s.index.notna()]
    s = s[~s.index.duplicated()].sort_index()
    if s.median() < 30: s = s * 18.016                       # mmol/L -> mg/dL
    s = s.where((s >= 40) & (s <= 400)).resample("5min").mean()
    na = s.isna()
    run = na.groupby((~na).cumsum()).transform("sum")        # length of each NaN run
    out = s.interpolate(limit_area="inside")
    out[na & (run > 6)] = np.nan                             # keep long gaps empty
    return out

def windows(s, stride=2):
    g, mins = s.values.astype(np.float32), (s.index.hour * 60 + s.index.minute).values
    X, Y, raw = [], [], []
    for i in range(0, len(g) - WINDOW - HORIZONS[-1] + 1, stride):
        w, fut = g[i:i + WINDOW], np.array([g[i + WINDOW - 1 + h] for h in HORIZONS])
        if np.isnan(w).any() or np.isnan(fut).any(): continue
        X.append(make_features(w, mins[i:i + WINDOW])); Y.append((fut - w[-1]) / 100); raw.append((w[-1], *fut))
    return X, Y, raw

files = sorted(glob.glob(f"{DATA_DIR}/*.csv") + glob.glob(f"{DATA_DIR}/*.xlsx"))
rng = np.random.default_rng(0); rng.shuffle(files)
n = len(files); cut1, cut2 = int(.70 * n), int(.85 * n)      # patient-wise 70/15/15 split (no leakage)
splits = {"train": files[:cut1], "val": files[cut1:cut2], "test": files[cut2:]}
data = {}
for k, fl in splits.items():
    X, Y, R = [], [], []
    for f in fl:
        x, y, r = windows(load_patient(f)); X += x; Y += y; R += r
    data[k] = (np.array(X, np.float32), np.array(Y, np.float32), np.array(R, np.float32))
    print(k, len(fl), "patients,", len(X), "windows")

# %% [markdown]
# ## Train GRU

# %%
def loader(k, shuffle): return DataLoader(TensorDataset(torch.from_numpy(data[k][0]), torch.from_numpy(data[k][1])), batch_size=256, shuffle=shuffle)
model = GlucoseGRU().to(dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
loss_fn = nn.HuberLoss()
best, bad = 1e9, 0
for ep in range(40):
    model.train()
    for xb, yb in loader("train", True):
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = np.mean([nn.functional.mse_loss(model(xb.to(dev)), yb.to(dev)).item() for xb, yb in loader("val", False)])
    sched.step(v); print(f"epoch {ep:02d} val_mse {v:.5f}")
    if v < best: best, bad = v, 0; torch.save(model.state_dict(), OUT_DIR / "glucose_gru.pt")
    else:
        bad += 1
        if bad >= 6: break
model.load_state_dict(torch.load(OUT_DIR / "glucose_gru.pt")); model.eval()

# %% [markdown]
# ## Evaluate on the held-out patients: RMSE, MAE, Clarke Error Grid, persistence + XGBoost baselines

# %%
def clarke_zone(ref, pred):
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref): return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180): return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182): return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref): return "D"
    return "B"

Xt, Yt, Rt = data["test"]
with torch.no_grad(): P = model(torch.from_numpy(Xt).to(dev)).cpu().numpy() * 100
last = Rt[:, 0]
true = Rt[:, 1:]                                  # absolute future values (30, 60 min)
gru = last[:, None] + P
pers = np.repeat(last[:, None], 2, axis=1)

import xgboost as xgb
flat = lambda X: X.reshape(len(X), -1)
xg = np.stack([xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=.08, tree_method="hist")
               .fit(flat(data["train"][0]), data["train"][1][:, j]).predict(flat(Xt)) * 100 + last for j in range(2)], axis=1)

rmse = lambda p, t: float(np.sqrt(np.mean((p - t) ** 2)))
for j, h in enumerate((30, 60)):
    r = {m: rmse(p[:, j], true[:, j]) for m, p in (("GRU", gru), ("XGBoost", xg), ("Persistence", pers))}
    zones = [clarke_zone(a, b) for a, b in zip(true[:, j], gru[:, j])]
    ab = 100 * np.mean([z in "AB" for z in zones])
    print(f"+{h} min  RMSE {r}  GRU MAE {np.mean(np.abs(gru[:, j] - true[:, j])):.1f}  "
          f"RMSE gain vs persistence {100 * (1 - r['GRU'] / r['Persistence']):.1f}%  Clarke A+B {ab:.1f}%")

# %%
#from google.colab import files; files.download(str(OUT_DIR / "glucose_gru.pt"))
