"""Historia sintética tipo Liga MX para probar el pipeline sin red (NO son resultados reales)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CURRENT_TEAMS
from .ratings import score_matrix, wdl


def make_history(n_tournaments: int = 10, seed: int = 11, end: str = "2026-10-12") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [t for t in CURRENT_TEAMS if t != "Atlante"] + ["Mazatlán"]
    atk = {t: rng.normal(0, 0.22) for t in teams + ["Atlante"]}
    dfn = {t: rng.normal(0, 0.22) for t in teams + ["Atlante"]}
    rows, mu, h = [], 0.18, 0.28
    last_start = pd.Timestamp(end) - pd.Timedelta(weeks=11)
    for tn in range(n_tournaments):
        start = last_start - pd.Timedelta(weeks=26 * (n_tournaments - 1 - tn))
        tt = [("Atlante" if t == "Mazatlán" else t) for t in teams] if tn == n_tournaments - 1 else teams
        for t in tt:  # deriva lenta de nivel entre torneos
            atk[t] += rng.normal(0, 0.05)
            dfn[t] += rng.normal(0, 0.05)
        for wk in range(17 if tn < n_tournaments - 1 else 11):
            order = rng.permutation(tt)
            for i in range(0, 18, 2):
                a, b = order[i], order[i + 1]
                home, away = (a, b) if rng.random() < 0.5 else (b, a)
                lam, mua = np.exp(mu + h + atk[home] - dfn[away]), np.exp(mu + atk[away] - dfn[home])
                hg, ag = rng.poisson(lam), rng.poisson(mua)
                # momios sintéticos ~ probabilidades verdaderas + ruido + margen 6%
                pr = wdl(score_matrix(lam, mua, 0.0)) * np.exp(rng.normal(0, 0.05, 3))
                pr /= pr.sum()
                odds = 1 / (pr * 1.06)
                rows.append((start + pd.Timedelta(weeks=wk) + pd.Timedelta(days=int(rng.integers(0, 3))),
                             home, away, hg, ag, *odds))
    df = pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag", "odds_h", "odds_d", "odds_a"])
    return df.sort_values("date", kind="stable").reset_index(drop=True)
