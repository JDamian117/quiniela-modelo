"""Ensamble de modelos (pool lineal con pesos óptimos en OOF) y ajuste acotado por señales contextuales."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

from .config import SIGNAL_CAP
from .xgb_model import log_loss


def fit_weights(components: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Pesos en el simplex que minimizan log-loss del pool lineal sobre predicciones fuera de muestra."""
    k = len(components)
    f = lambda z: log_loss(sum(w * P for w, P in zip(softmax(z), components)), y)  # noqa: E731
    z = minimize(f, np.zeros(k), method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-6}).x
    return softmax(z)


def blend(components: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    P = sum(wi * Pi for wi, Pi in zip(w, components))
    return P / P.sum(axis=-1, keepdims=True)


def apply_signals(p: np.ndarray, inj_h: float, inj_a: float, sent_h: float, sent_a: float,
                  b_inj: float = 0.10, b_sent: float = 0.05, cap: float = SIGNAL_CAP) -> np.ndarray:
    """Desplaza log-odds local/visitante según lesiones (índice de importancia) y sentimiento ([-1, 1]).

    Lesiones y noticias no existen en el histórico de entrenamiento, así que NO entran como features de XGBoost:
    se aplican como prior bayesiano acotado (|shift| <= cap en log-odds ~ <= 5 pts de probabilidad).
    """
    shift = float(np.clip(b_inj * (inj_a - inj_h) + b_sent * (sent_h - sent_a), -cap, cap))
    q = np.array(p, float)
    q[0] *= np.exp(shift / 2)
    q[2] *= np.exp(-shift / 2)
    return q / q.sum()


def holdout_check(components: list[np.ndarray], names: list[str], y: np.ndarray, frac: float = 0.6) -> dict:
    """Chequeo honesto (spec §5): pesos ajustados en el primer `frac` de las OOF y evaluados en el 40% posterior,
    comparando contra cada componente. Mide el optimismo de ajustar pesos y evaluar sobre las mismas filas."""
    k = int(len(y) * frac)
    w = fit_weights([c[:k] for c in components], y[:k])
    res = {"n_holdout": int(len(y) - k), "pesos_ajuste": dict(zip(names, np.round(w, 3).tolist())),
           "ensamble": log_loss(blend([c[k:] for c in components], w), y[k:])}
    res.update({n: log_loss(c[k:], y[k:]) for n, c in zip(names, components)})
    best = min(names, key=lambda n: res[n])
    res["mejor_componente"], res["ensamble_le_mejor"] = best, bool(res["ensamble"] <= res[best] + 1e-9)
    return res


def confidence(p: np.ndarray) -> str:
    """Confianza calculada en código (no depende del LLM): margen entre la 1ª y 2ª probabilidad."""
    top = np.sort(p)[::-1]
    m = top[0] - top[1]
    return "alta" if m >= 0.15 else "media" if m >= 0.07 else "baja"
