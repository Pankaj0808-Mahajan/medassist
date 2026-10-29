# %% [markdown]
# # 3. Lab risk classifier (XGBoost on NHANES 2017-2018) - optional module
# Predicts the probability of diagnosed diabetes from age + a few lab values. Missing values are allowed (XGBoost handles NaN),
# so a partial blood report still works. This is a screening aid, not a diagnosis.
# Output: `lab_risk_xgb.json` -> copy into `backend/weights/`.

# %%
#!pip install -q xgboost scikit-learn pandas requests

# %%
import io, numpy as np, pandas as pd, requests, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

CDC_URLS = [  # CDC moved the files; the first pattern is current, the second is the old one
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{n}.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/{n}.XPT",
]

def xpt(name):
    for url in CDC_URLS:
        r = requests.get(url.format(n=name), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        if r.ok and r.content.startswith(b"HEADER RECORD"):      # a real SAS XPORT file, not an HTML error page
            return pd.read_sas(io.BytesIO(r.content), format="xport").set_index("SEQN")
    raise RuntimeError(f"Could not download {name}.xpt from CDC. Download it manually from "
                       "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx and upload it to /content.")

df = xpt("DEMO_J")[["RIDAGEYR"]].join([xpt("GHB_J")[["LBXGH"]], xpt("TCHOL_J")[["LBXTC"]], xpt("HDL_J")[["LBDHDD"]],
                                       xpt("BIOPRO_J")[["LBXSCR", "LBXSATSI"]], xpt("DIQ_J")[["DIQ010"]]])
df = df[(df.RIDAGEYR >= 18) & df.DIQ010.isin([1, 2])]
y = (df.DIQ010 == 1).astype(int).values
# same order as RISK_FEATURES in backend/app/modules/labs.py: age, HbA1c, total cholesterol, HDL, creatinine, ALT
X = df[["RIDAGEYR", "LBXGH", "LBXTC", "LBDHDD", "LBXSCR", "LBXSATSI"]].values.astype(np.float32)
X[(X < 0)] = np.nan                                                    # remove impossible values
print(X.shape, "adults;", y.mean().round(3), "diagnosed diabetes")

# %%
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=0)
rng = np.random.default_rng(0)
Xaug = Xtr.copy(); mask = rng.random(Xaug.shape) < 0.2; mask[:, 0] = False; Xaug[mask] = np.nan   # robust to partial reports
clf = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                        scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), eval_metric="auc", early_stopping_rounds=30)
clf.fit(np.vstack([Xtr, Xaug]), np.concatenate([ytr, ytr]), eval_set=[(Xva, yva)], verbose=False)
p = clf.predict_proba(Xte)[:, 1]
pr, rc, f1, _ = precision_recall_fscore_support(yte, p >= 0.5, average="binary")
print(f"test ROC-AUC {roc_auc_score(yte, p):.3f}  precision {pr:.3f}  recall {rc:.3f}  F1 {f1:.3f}")
clf.save_model("/content/lab_risk_xgb.json")

# %%
#from google.colab import files; files.download("/content/lab_risk_xgb.json")
