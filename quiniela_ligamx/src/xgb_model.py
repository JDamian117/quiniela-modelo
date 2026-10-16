"""XGBoost multiclase (1/X/2) con validación walk-forward, calibración por temperatura y métricas probabilísticas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from xgboost import XGBClassifier

from .features import FEATS

PARAMS = dict(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=8, reg_lambda=8.0, objective="multi:softprob", eval_metric="mlogloss",
              tree_method="hist", n_jobs=2, random_state=7)


def _weights(dates: pd.Series, half_life_days: float = 540) -> np.ndarray:
    age = (dates.max() - dates).dt.days.to_numpy(float)
    return 0.5 ** (age / half_life_days)


def _fit(df: pd.DataFrame) -> XGBClassifier:
    m = XGBClassifier(**PARAMS)
    m.fit(df[FEATS], df["y"], sample_weight=_weights(df["date"]))
    return m


def walk_forward_oof(df: pd.DataFrame, n_splits: int = 5, min_train: int = 350) -> tuple[np.ndarray, np.ndarray]:
    """Folds expansivos en el tiempo (entrena con pasado, predice futuro). Devuelve (índices OOF, probs OOF)."""
    n = len(df)
    start = min(min_train, n // 2)
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    idx, P = [], []
    for k in range(n_splits):
        tr, te = df.iloc[:edges[k]], df.iloc[edges[k]:edges[k + 1]]
        m = _fit(tr)
        P.append(m.predict_proba(te[FEATS]))
        idx.append(te.index.to_numpy())
    return np.concatenate(idx), np.vstack(P)


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    """Escala T>0 que minimiza log-loss de softmax(log p / T) en OOF (T>1 suaviza sobreconfianza)."""
    logp = np.log(np.clip(P, 1e-9, 1))
    f = lambda T: log_loss(softmax(logp / T, axis=1), y)  # noqa: E731
    return float(minimize_scalar(f, bounds=(0.5, 2.5), method="bounded").x)


def calibrate(P: np.ndarray, T: float) -> np.ndarray:
    return softmax(np.log(np.clip(P, 1e-9, 1)) / T, axis=1)


# ---- Métricas -----------------------------------------------------------------------------------
def log_loss(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean())


def rps(P: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score (resultados ordenados 1 > X > 2); menor es mejor."""
    Y = np.eye(3)[y]
    return float(((np.cumsum(P, 1)[:, :2] - np.cumsum(Y, 1)[:, :2]) ** 2).sum(1).mean() / 2)


def ece_by_class(P: np.ndarray, y: np.ndarray, bins: int = 10) -> np.ndarray:
    """ECE uno-contra-resto por clase [1, X, 2] (spec §5); menor = mejor calibrado."""
    Y, edges = np.eye(3)[y], np.linspace(0, 1, bins + 1)
    out = np.zeros(3)
    for c in range(3):
        b = np.clip(np.digitize(P[:, c], edges) - 1, 0, bins - 1)
        for k in range(bins):
            m = b == k
            if m.any():
                out[c] += m.mean() * abs(P[m, c].mean() - Y[m, c].mean())
    return out


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    return float(ece_by_class(P, y, bins).mean())


def reliability_table(P: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Curva de fiabilidad (10 bins) por clase: prob. media predicha vs frecuencia observada."""
    Y, edges, rows = np.eye(3)[y], np.linspace(0, 1, bins + 1), []
    for c, name in enumerate(("1", "X", "2")):
        b = np.clip(np.digitize(P[:, c], edges) - 1, 0, bins - 1)
        for k in range(bins):
            m = b == k
            if m.any():
                rows.append({"clase": name, "bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}", "n": int(m.sum()),
                             "p_pred": float(P[m, c].mean()), "freq_obs": float(Y[m, c].mean())})
    return pd.DataFrame(rows)


def score(P: np.ndarray, y: np.ndarray) -> dict:
    Y = np.eye(3)[y]
    return {"logloss": log_loss(P, y), "rps": rps(P, y), "brier": float(((P - Y) ** 2).sum(1).mean()),
            "acc": float((P.argmax(1) == y).mean()), "ece": ece(P, y)}
