"""Modelos estadísticos base: Elo incremental y Dixon-Coles (Poisson bivariado con decaimiento temporal)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .config import DC_HALF_LIFE_DAYS, DC_RIDGE, ELO_HOME_ADV, ELO_K


class Elo:
    def __init__(self, k: float = ELO_K, home_adv: float = ELO_HOME_ADV, base: float = 1500.0):
        self.k, self.home_adv = k, home_adv
        self.r: dict[str, float] = defaultdict(lambda: base)

    def expected(self, home: str, away: str) -> float:
        return 1.0 / (1.0 + 10 ** (-(self.r[home] + self.home_adv - self.r[away]) / 400.0))

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mult = np.log(abs(hg - ag) + 1.0) + 1.0  # prima por margen de gol
        d = self.k * mult * (s - self.expected(home, away))
        self.r[home] += d
        self.r[away] -= d


def poisson_pmf(lam: float, n: int) -> np.ndarray:
    k = np.arange(n + 1)
    return np.exp(k * np.log(lam) - lam - gammaln(k + 1))


def score_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = 9) -> np.ndarray:
    """P[i, j] = P(local anota i, visitante anota j) con corrección Dixon-Coles para marcadores bajos."""
    P = np.outer(poisson_pmf(lam, max_goals), poisson_pmf(mu, max_goals))
    P[0, 0] *= max(1 - lam * mu * rho, 1e-6)
    P[0, 1] *= max(1 + lam * rho, 1e-6)
    P[1, 0] *= max(1 + mu * rho, 1e-6)
    P[1, 1] *= max(1 - rho, 1e-6)
    return P / P.sum()


def wdl(P: np.ndarray) -> np.ndarray:
    """Matriz de marcadores -> [P(local), P(empate), P(visitante)]."""
    return np.array([np.tril(P, -1).sum(), np.trace(P), np.triu(P, 1).sum()])


class DixonColes:
    """lambda_local = exp(mu + h + atk_local - def_visita); lambda_visita = exp(mu + atk_visita - def_local).

    Ridge sobre atk/def: equipos con pocos partidos (ej. Atlante recién ascendido) regresan al promedio de la liga.
    """

    def __init__(self, half_life_days: float = DC_HALF_LIFE_DAYS, ridge: float = DC_RIDGE, maxiter: int = 150):
        self.xi = np.log(2) / half_life_days
        self.ridge, self.maxiter = ridge, maxiter
        self.fitted = False

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> "DixonColes":
        ref_date = ref_date or df["date"].max()
        self.teams = sorted(set(df["home"]) | set(df["away"]))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi, ai = df["home"].map(self.idx).to_numpy(), df["away"].map(self.idx).to_numpy()
        hg, ag = df["hg"].to_numpy(float), df["ag"].to_numpy(float)
        age = (ref_date - df["date"]).dt.days.clip(lower=0).to_numpy(float)
        w = np.exp(-self.xi * age)
        lg_h, lg_a = gammaln(hg + 1), gammaln(ag + 1)
        m00, m01, m10, m11 = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1), (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)

        def nll(p):
            atk, dfn, mu, h, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]
            lam = np.exp(mu + h + atk[hi] - dfn[ai])
            mu_a = np.exp(mu + atk[ai] - dfn[hi])
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu_a[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu_a[m10] * rho
            tau[m11] = 1 - rho
            ll = hg * np.log(lam) - lam - lg_h + ag * np.log(mu_a) - mu_a - lg_a + np.log(np.clip(tau, 1e-6, None))
            return -(w * ll).sum() + self.ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))

        x0 = np.concatenate([np.zeros(2 * n), [0.2, 0.25, -0.05]])
        bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": self.maxiter})
        p = res.x
        self.atk, self.dfn = p[:n], p[n:2 * n]
        self.mu, self.h, self.rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        self.fitted = True
        return self

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Equipo desconocido -> atk=def=0 (promedio de liga)."""
        ih, ia = self.idx.get(home), self.idx.get(away)
        ah, dh = (self.atk[ih], self.dfn[ih]) if ih is not None else (0.0, 0.0)
        aa, da = (self.atk[ia], self.dfn[ia]) if ia is not None else (0.0, 0.0)
        return float(np.exp(self.mu + self.h + ah - da)), float(np.exp(self.mu + aa - dh))

    def matrix(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.lambdas(home, away)
        return score_matrix(lam, mu, self.rho)

    def probs(self, home: str, away: str) -> np.ndarray:
        return wdl(self.matrix(home, away))
