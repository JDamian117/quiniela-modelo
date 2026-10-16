"""Cadena de Markov de forma: estado = últimos `order` resultados (G/E/P) del equipo.

P(siguiente resultado | estado, sede) se estima por equipo con suavizado de Dirichlet hacia la matriz global,
así que equipos con pocos datos heredan el comportamiento promedio de la liga.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

W, D, L = 0, 1, 2


def _res(gf: int, ga: int) -> int:
    return W if gf > ga else D if gf == ga else L


class FormMarkov:
    def __init__(self, order: int = 1, alpha: float = 6.0):
        self.order, self.alpha = order, alpha
        self.team_cnt = defaultdict(lambda: np.zeros(3))   # (equipo, sede, estado) -> conteos W/D/L
        self.glob_cnt = defaultdict(lambda: np.zeros(3))   # (sede, estado) -> conteos
        self.hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=order))

    def state(self, team: str) -> tuple:
        h = list(self.hist[team])
        return tuple([-1] * (self.order - len(h)) + h)  # -1 = sin historial

    def _dist(self, team: str, venue: str) -> np.ndarray:
        s = self.state(team)
        g = self.glob_cnt[(venue, s)] + np.array([0.45, 0.27, 0.28]) * 3.0  # prior de liga
        g = g / g.sum()
        c = self.team_cnt[(team, venue, s)]
        return (c + self.alpha * g) / (c.sum() + self.alpha)

    def probs(self, home: str, away: str) -> np.ndarray:
        """[P(local), P(empate), P(visitante)] promediando la cadena del local (en casa) y la del visitante (fuera)."""
        ph, pa = self._dist(home, "H"), self._dist(away, "A")
        p = np.array([(ph[W] + pa[L]) / 2, (ph[D] + pa[D]) / 2, (ph[L] + pa[W]) / 2])
        return p / p.sum()

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        for team, venue, gf, ga in ((home, "H", hg, ag), (away, "A", ag, hg)):
            r, s = _res(gf, ga), self.state(team)
            self.team_cnt[(team, venue, s)][r] += 1
            self.glob_cnt[(venue, s)][r] += 1
            self.hist[team].append(r)

    def transition_matrix(self, team: str) -> np.ndarray:
        """Matriz 3x3 (orden 1) agregada por sede, para explicabilidad: filas = último resultado G/E/P."""
        M = np.zeros((3, 3))
        for (t, _v, s), c in self.team_cnt.items():
            if t == team and self.order == 1 and s[0] >= 0:
                M[s[0]] += c
        return M / np.clip(M.sum(axis=1, keepdims=True), 1, None)
