# %% [markdown]
# # 2. Chest X-ray classifier (DenseNet121, NORMAL vs PNEUMONIA)
# Dataset: Kermany et al. "Chest X-Ray Images (Pneumonia)" from Kaggle. Upload your `kaggle.json` when asked.
# Output: `xray_densenet121.pt` and `xray_meta.json` -> copy both into `backend/weights/`.

# %%
#!pip install -q kaggle torch torchvision opencv-python-headless scikit-learn
#!mkdir -p ~/.kaggle
#from google.colab import files; files.upload()          # choose kaggle.json
#!cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
#!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /content --unzip

# %%
import glob, json, re, cv2, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix

OUT = Path("/content/weights"); OUT.mkdir(exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]; STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

# %% [markdown]
# ## Re-split 80/10/10, stratified AND grouped by patient
# The Kaggle train/val/test folders are re-pooled. Image filenames carry a patient id (`person123_...`, `IM-0001-...`),
# so images of one patient never land in different splits.

# %%
items = []
for p in glob.glob("/content/chest_xray/**/*.jp*g", recursive=True):
    if "__MACOSX" in p: continue
    label = 1 if "PNEUMONIA" in p.upper().split("/") else 0 if "NORMAL" in p.upper().split("/") else None
    if label is None: continue
    m = re.search(r"person(\d+)", p) or re.search(r"IM-(\d+)", p)
    items.append((p, label, f"{label}-{m.group(1) if m else Path(p).stem}"))
paths, y, groups = zip(*items); y, groups = np.array(y), np.array(groups)
folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=0).split(paths, y, groups))
idx = {"test": folds[0][1], "val": folds[1][1], "train": np.concatenate([f[1] for f in folds[2:]])}
split = {k: [(paths[i], int(y[i])) for i in v] for k, v in idx.items()}
print({k: (len(v), int(sum(l for _, l in v))) for k, v in split.items()}, "(images, pneumonia)")

class DS(Dataset):
    def __init__(self, items, train): self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, lab = self.items[i]
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.imread(p, 0))      # same as backend preprocess()
        if self.train:                                                                           # mild augmentation
            h, w = g.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-7, 7), np.random.uniform(0.9, 1.1))
            g = np.clip(cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.float32) * np.random.uniform(.9, 1.1), 0, 255).astype(np.uint8)
        g = cv2.resize(g, (224, 224)).astype(np.float32) / 255
        return torch.from_numpy((np.stack([g, g, g]) - MEAN) / STD).float(), lab

dl = {k: DataLoader(DS(v, k == "train"), batch_size=32, shuffle=k == "train", num_workers=2) for k, v in split.items()}

# %% [markdown]
# ## Fine-tune DenseNet121 (ImageNet weights, class-weighted loss)

# %%
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
model.classifier = nn.Linear(1024, 2)
model.to(dev)
cnt = np.bincount([l for _, l in split["train"]])
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(y) / (2 * cnt), dtype=torch.float32).to(dev))
opt = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

def predict(loader):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            P += torch.softmax(model(xb.to(dev)), 1)[:, 1].cpu().tolist(); Y += yb.tolist()
    return np.array(P), np.array(Y)

best = 0
for ep in range(8):
    model.train()
    for xb, yb in dl["train"]:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        with torch.autocast(dev, enabled=dev == "cuda"): loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    pv, yv = predict(dl["val"]); auc = roc_auc_score(yv, pv); print(f"epoch {ep} val ROC-AUC {auc:.4f}")
    if auc > best: best = auc; torch.save(model.state_dict(), OUT / "xray_densenet121.pt")
model.load_state_dict(torch.load(OUT / "xray_densenet121.pt"))

# %% [markdown]
# ## Pick the decision threshold on validation (smallest threshold change that gives >= 90% pneumonia recall), then test

# %%
pv, yv = predict(dl["val"])
fpr, tpr, thr = roc_curve(yv, pv)
thr_ok = thr[np.argmax(tpr >= 0.90)] if (tpr >= 0.90).any() else 0.5
threshold = float(min(0.5, thr_ok))
json.dump({"threshold": threshold}, open(OUT / "xray_meta.json", "w"))
pt, yt = predict(dl["test"]); pred = (pt >= threshold).astype(int)
pr, rc, f1, _ = precision_recall_fscore_support(yt, pred, average="binary")
print(f"threshold {threshold:.3f}  acc {accuracy_score(yt, pred):.3f}  precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}  ROC-AUC {roc_auc_score(yt, pt):.3f}")
print(confusion_matrix(yt, pred))

# %%
#from google.colab import files; [files.download(str(OUT / n)) for n in ("xray_densenet121.pt", "xray_meta.json")]
