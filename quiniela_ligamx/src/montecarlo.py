"""Monte Carlo: marcadores simulados con incertidumbre paramétrica + distribución de aciertos de la quiniela."""
from __future__ import annotations

import numpy as np

from .config import MC_PARAM_SIGMA
from .ratings import score_matrix


def simulate_match(lam: float, mu: float, rho: float, n: int = 100_000, sigma: float = MC_PARAM_SIGMA,
                   seed: int | None = None, max_goals: int = 9) -> dict:
    """Simula n partidos. Cada simulación perturba (lam, mu) con ruido lognormal para reflejar que las
    tasas de gol son estimadas, no conocidas (ensancha colas y sube ligeramente P(empate) de forma realista).
    Se muestrea el marcador exacto de la matriz Dixon-Coles correspondiente (agrupada en 25 niveles de ruido)."""
    rng = np.random.default_rng(seed)
    levels = 25
    z = rng.standard_normal(levels) if sigma > 0 else np.zeros(levels)
    zh, za = z, rng.standard_normal(levels) if sigma > 0 else np.zeros(levels)
    per = n // levels
    hg_all, ag_all = [], []
    side = max_goals + 1
    for i in range(levels):
        P = score_matrix(lam * np.exp(sigma * zh[i] - sigma ** 2 / 2), mu * np.exp(sigma * za[i] - sigma ** 2 / 2), rho, max_goals)
        k = rng.choice(side * side, size=per, p=P.ravel())
        hg_all.append(k // side)
        ag_all.append(k % side)
    hg, ag = np.concatenate(hg_all), np.concatenate(ag_all)
    probs = np.array([(hg > ag).mean(), (hg == ag).mean(), (hg < ag).mean()])
    scores, counts = np.unique(hg * 100 + ag, return_counts=True)
    top = np.argsort(-counts)[:3]
    return {
        "probs": probs,
        "exp_goals": (float(hg.mean()), float(ag.mean())),
        "p_over25": float(((hg + ag) > 2.5).mean()),
        "p_btts": float(((hg > 0) & (ag > 0)).mean()),
        "top_scores": [(f"{scores[j] // 100}-{scores[j] % 100}", float(counts[j] / len(hg))) for j in top],
    }


def simulate_slate(P: np.ndarray, picks: np.ndarray, n: int = 200_000, seed: int | None = 0) -> dict:
    """P: (partidos x 3) probabilidades finales; picks: índice 0/1/2 elegido por partido.
    Devuelve E[aciertos], distribución de aciertos y P(>=k)."""
    rng = np.random.default_rng(seed)
    cum = P.cumsum(axis=1)
    u = rng.random((n, P.shape[0]))
    outcome = (u[:, :, None] > cum[None, :, :2]).sum(axis=2)  # 0/1/2 simulado
    hits = (outcome == picks[None, :]).sum(axis=1)
    dist = np.bincount(hits, minlength=P.shape[0] + 1) / n
    return {"expected_hits": float(hits.mean()), "dist": dist,
            "p_ge": {k: float(dist[k:].sum()) for k in range(1, P.shape[0] + 1)}}
