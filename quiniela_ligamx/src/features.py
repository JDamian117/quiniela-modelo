"""Motor de features sin fuga de datos: para cada partido solo usa información ANTERIOR a su fecha.

Mismo código para entrenamiento (features -> update, partido a partido) e inferencia (features sobre fixtures futuros).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .markov import FormMarkov
from .ratings import Elo, DixonColes

FEATS = [
    "elo_h", "elo_a", "elo_diff", "elo_p",
    "ppg5_h", "ppg5_a", "gf5_h", "ga5_h", "gf5_a", "ga5_a", "hppg_h", "appg_a", "rest_h", "rest_a",
    "mk_pH", "mk_pD", "mk_pA",
    "dc_pH", "dc_pD", "dc_pA", "dc_lam_h", "dc_lam_a",
]
MIN_DC_MATCHES = 120


def _mean(d: deque, default: float) -> float:
    return float(np.mean(d)) if len(d) else default


class FeatureEngine:
    def __init__(self, refit_days: int = 28, markov_order: int = 1):
        self.elo, self.mk = Elo(), FormMarkov(order=markov_order)
        self.refit_days = refit_days
        self.dc: DixonColes | None = None
        self.last_fit: pd.Timestamp | None = None
        self.rows: list[dict] = []
        self.t = defaultdict(lambda: {k: deque(maxlen=n) for k, n in
                                      (("pts", 5), ("gf", 5), ("ga", 5), ("hpts", 8), ("apts", 8))} | {"last": None})

    # ---- Dixon-Coles con refit periódico (solo datos previos a `date`) -------------------------
    def refit_dc(self, date: pd.Timestamp, force: bool = False) -> None:
        if len(self.rows) < MIN_DC_MATCHES:
            return
        if force or self.dc is None or (date - self.last_fit).days >= self.refit_days:
            self.dc = DixonColes().fit(pd.DataFrame(self.rows), ref_date=date)
            self.last_fit = date

    def features(self, home: str, away: str, date: pd.Timestamp) -> dict:
        self.refit_dc(date)
        th, ta = self.t[home], self.t[away]
        eh, ea = self.elo.r[home], self.elo.r[away]
        f = {
            "elo_h": eh, "elo_a": ea, "elo_diff": eh - ea, "elo_p": self.elo.expected(home, away),
            "ppg5_h": _mean(th["pts"], 1.35), "ppg5_a": _mean(ta["pts"], 1.35),
            "gf5_h": _mean(th["gf"], 1.3), "ga5_h": _mean(th["ga"], 1.3),
            "gf5_a": _mean(ta["gf"], 1.3), "ga5_a": _mean(ta["ga"], 1.3),
            "hppg_h": _mean(th["hpts"], 1.6), "appg_a": _mean(ta["apts"], 1.0),
            "rest_h": min((date - th["last"]).days, 21) if th["last"] is not None else 7,
            "rest_a": min((date - ta["last"]).days, 21) if ta["last"] is not None else 7,
        }
        f["mk_pH"], f["mk_pD"], f["mk_pA"] = self.mk.probs(home, away)
        if self.dc is not None:
            f["dc_pH"], f["dc_pD"], f["dc_pA"] = self.dc.probs(home, away)
            f["dc_lam_h"], f["dc_lam_a"] = self.dc.lambdas(home, away)
        else:
            for k in ("dc_pH", "dc_pD", "dc_pA", "dc_lam_h", "dc_lam_a"):
                f[k] = np.nan
        return f

    def update(self, r) -> None:
        h, a, hg, ag, d = r["home"], r["away"], int(r["hg"]), int(r["ag"]), r["date"]
        self.elo.update(h, a, hg, ag)
        self.mk.update(h, a, hg, ag)
        ph, pa = (3 if hg > ag else 1 if hg == ag else 0), (3 if ag > hg else 1 if hg == ag else 0)
        for team, pts, gf, ga, key in ((h, ph, hg, ag, "hpts"), (a, pa, ag, hg, "apts")):
            t = self.t[team]
            t["pts"].append(pts); t["gf"].append(gf); t["ga"].append(ga); t[key].append(pts); t["last"] = d
        self.rows.append({"date": d, "home": h, "away": a, "hg": hg, "ag": ag})


def build_training_frame(hist: pd.DataFrame, warmup: int = 150, refit_days: int = 28, verbose: bool = True):
    """Recorre la historia en orden temporal. Devuelve (df con features + y, engine en estado final).

    y: 0 = gana local, 1 = empate, 2 = gana visitante. Las primeras `warmup` filas solo calientan Elo/Markov.
    """
    eng = FeatureEngine(refit_days=refit_days)
    out = []
    n = len(hist)
    for i, r in enumerate(hist.itertuples(index=False)):
        row = r._asdict()
        if i >= warmup:
            f = eng.features(row["home"], row["away"], row["date"])
            f.update(date=row["date"], home=row["home"], away=row["away"],
                     y=0 if row["hg"] > row["ag"] else 1 if row["hg"] == row["ag"] else 2,
                     mkt_pH=row.get("mkt_pH", np.nan), mkt_pD=row.get("mkt_pD", np.nan), mkt_pA=row.get("mkt_pA", np.nan))
            out.append(f)
        eng.update(row)
        if verbose and i and i % 400 == 0:
            print(f"  features: {i}/{n}")
    df = pd.DataFrame(out).dropna(subset=["dc_pH"]).reset_index(drop=True)
    return df, eng
