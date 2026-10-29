"""Glucose forecaster: GRU predicting glucose 30 and 60 min ahead from the last 2 h (5-min grid).

make_features / WINDOW / HORIZONS must stay identical to colab/01_glucose_gru.py.
"""
from datetime import datetime, timedelta
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..config import WEIGHTS

WINDOW = 24          # 2 h of history at 5-min spacing
HORIZONS = (6, 12)   # +30 min, +60 min, in 5-min steps


class GlucoseGRU(nn.Module):
    def __init__(self, n_in=5, hidden=64, layers=2, n_out=len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, n_out))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


def make_features(g, minutes_of_day) -> np.ndarray:
    """(WINDOW,) mg/dL + minute-of-day -> (WINDOW, 5): level, offset from latest, rate of change, time of day."""
    g = np.asarray(g, dtype=np.float32)
    d = np.diff(g, prepend=g[0])
    ang = 2 * np.pi * np.asarray(minutes_of_day, dtype=np.float32) / 1440
    return np.stack([g / 400, (g - g[-1]) / 100, d / 20, np.sin(ang), np.cos(ang)], axis=1)


class GlucoseForecaster:
    def __init__(self):
        self.model: Optional[GlucoseGRU] = None
        path = WEIGHTS / "glucose_gru.pt"
        if path.exists():
            m = GlucoseGRU()
            m.load_state_dict(torch.load(path, map_location="cpu"))
            self.model = m.eval()

    def predict(self, readings: Sequence[float], last_time: Optional[datetime] = None) -> dict:
        if len(readings) < WINDOW:
            raise ValueError(f"Need at least {WINDOW} glucose readings (2 hours at 5-minute spacing); got {len(readings)}.")
        g = np.clip(np.asarray(readings[-WINDOW:], dtype=np.float32), 40, 400)  # sensor range
        end = last_time or datetime.now()
        mins = [(t.hour * 60 + t.minute) for t in (end - timedelta(minutes=5 * (WINDOW - 1 - i)) for i in range(WINDOW))]
        if self.model is not None:
            with torch.no_grad():
                delta = self.model(torch.from_numpy(make_features(g, mins))[None]).numpy()[0] * 100
            preds, name = g[-1] + delta, "GRU (trained on Colab)"
        else:  # persistence baseline so the app still runs before weights are added
            preds, name = np.array([g[-1]] * len(HORIZONS)), "persistence baseline (no glucose_gru.pt found)"
        preds = np.clip(preds, 40, 400)
        return {
            "current": float(g[-1]),
            "forecast": {"30": float(preds[0]), "60": float(preds[1])},
            "rate_per_5min": float(g[-1] - g[-2]),
            "history": [float(x) for x in g],
            "model": name,
        }
