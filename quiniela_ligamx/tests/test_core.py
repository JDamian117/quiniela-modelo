import numpy as np
import pandas as pd

from src.agent import LLM, apply_agent_adjustment, decide
from src.config import AGENT_MAX_DELTA, canon
from src.demo_data import make_history
from src.ensemble import apply_signals
from src.features import FeatureEngine
from src.markov import FormMarkov
from src.montecarlo import simulate_match, simulate_slate
from src.ratings import DixonColes, score_matrix, wdl


def test_canon_aliases():
    assert canon("Club America") == "América" and canon("Guadalajara Chivas") == "Chivas"
    assert canon("Atletico San Luis") == "San Luis" and canon("Juarez") == "Juárez" and canon("Leon") == "León"


def test_score_matrix_and_wdl_sum_to_one():
    P = score_matrix(1.6, 1.1, rho=-0.08)
    assert abs(P.sum() - 1) < 1e-9 and abs(wdl(P).sum() - 1) < 1e-9
    assert wdl(score_matrix(2.0, 0.8))[0] > wdl(score_matrix(0.8, 2.0))[0]


def test_dixon_coles_recovers_strength_order():
    h = make_history(n_tournaments=6)
    dc = DixonColes().fit(h)
    p = dc.probs("América", "Necaxa")
    assert abs(p.sum() - 1) < 1e-9 and dc.h > 0  # localía positiva
    assert dc.probs("Equipo Nuevo", "Equipo Nuevo 2").argmax() == 0  # desconocidos -> promedio + localía


def test_montecarlo_matches_analytic_when_sigma_zero():
    sim = simulate_match(1.5, 1.0, -0.05, n=250_000, sigma=0.0, seed=1)
    assert np.abs(sim["probs"] - wdl(score_matrix(1.5, 1.0, -0.05))).max() < 0.01


def test_markov_probs_valid_and_learn():
    mk = FormMarkov()
    for _ in range(30):
        mk.update("A", "B", 2, 0)
    p = mk.probs("A", "B")
    assert abs(p.sum() - 1) < 1e-9 and p[0] > p[2]


def test_no_leakage_in_features():
    h = make_history(n_tournaments=3)
    i = 200
    full = FeatureEngine(); part = FeatureEngine()
    for k, r in enumerate(h.head(i + 1).to_dict("records")):
        if k == i:
            f_full = full.features(r["home"], r["away"], r["date"])
        full.update(r) if k < i else None
    for r in h.head(i).to_dict("records"):
        part.update(r)
    r = h.iloc[i]
    f_part = part.features(r["home"], r["away"], r["date"])
    for k in ("elo_h", "elo_a", "ppg5_h", "gf5_a", "rest_h", "mk_pH", "mk_pD"):
        assert abs(f_full[k] - f_part[k]) < 1e-12


class Stub(LLM):
    def __init__(self, out):
        super().__init__("hf"); self.out = out
    def chat(self, system, user, max_tokens=700):
        return self.out


def test_agent_guardrails_clip_and_argmax_by_code():
    base = np.array([0.40, 0.30, 0.30])
    out = Stub('texto {"ajuste": {"H": -0.9, "D": 0.0, "A": 0.9}, "pick": "2", "razonamiento": ["x"]} fin')
    d = decide(out, {"partido": "t"}, base)
    assert abs(d["final"].sum() - 1) < 1e-9
    assert np.abs(d["final"] - base).max() <= AGENT_MAX_DELTA + 1e-9
    assert d["pick"] == "1X2"[int(d["final"].argmax())] and d["agent_used"]


def test_agent_invalid_json_falls_back_to_base():
    base = np.array([0.50, 0.25, 0.25])
    d = decide(Stub("no es json"), {}, base)
    assert not d["agent_used"] and np.allclose(d["final"], base) and d["pick"] == "1"
    assert np.allclose(apply_agent_adjustment(base, {"H": "x"}), base)


def test_signals_direction_and_cap():
    p = np.array([0.4, 0.3, 0.3])
    assert apply_signals(p, 2.0, 0.0, 0, 0)[0] < p[0] and apply_signals(p, 0, 0, 0.5, -0.5)[0] > p[0]
    assert abs(apply_signals(p, 0, 3.0, 0, 0)[0] - p[0]) < 0.07  # tope de log-odds


def test_slate_simulation():
    P = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    s = simulate_slate(P, np.array([0, 1]), n=1000)
    assert s["expected_hits"] == 2.0 and s["p_ge"][2] == 1.0


# ---- Criterios de aceptación del prompt mejorado (§3.4, §5, §6) ----------------------------------
import json
import subprocess
import sys
from pathlib import Path

from src import provenance as prov
from src.ensemble import confidence, holdout_check
from src.xgb_model import ece_by_class, reliability_table

ROOT = Path(__file__).resolve().parents[1]


def test_failed_source_is_logged_not_hidden(monkeypatch):
    import src.data as data
    prov.clear()
    monkeypatch.setattr(data, "_download", lambda refresh: (_ for _ in ()).throw(OSError("sin red")))
    try:
        data.load_history()
        assert False, "debió fallar explícitamente"
    except RuntimeError as e:
        assert "offline-csv" in str(e)
    assert any(r["estado"] == "fallo" and "football-data" in r["fuente"] for r in prov.get())
    assert "FALLO" in prov.summary()


def test_ece_per_class_and_reliability_bins():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([4, 3, 3], 4000)
    y = np.array([rng.choice(3, p=p) for p in P])
    assert ece_by_class(P, y).shape == (3,) and ece_by_class(P, y).max() < 0.04  # calibrado -> ECE bajo
    assert ece_by_class(P[:, ::-1], y).mean() > ece_by_class(P, y).mean()        # descalibrado -> peor
    rt = reliability_table(P, y)
    assert set(rt["clase"]) == {"1", "X", "2"} and rt["n"].sum() == 3 * len(y)


def test_confidence_and_holdout_check():
    assert confidence(np.array([0.6, 0.2, 0.2])) == "alta" and confidence(np.array([0.36, 0.33, 0.31])) == "baja"
    rng = np.random.default_rng(1)
    P = rng.dirichlet([5, 3, 3], 600)
    y = np.array([rng.choice(3, p=p) for p in P])
    hc = holdout_check([P, P[:, ::-1], np.full_like(P, 1 / 3)], ["bueno", "malo", "unif"], y)
    assert hc["mejor_componente"] == "bueno" and isinstance(hc["ensamble_le_mejor"], bool)


def test_end_to_end_demo_output_contract():
    r = subprocess.run([sys.executable, "run_quiniela.py", "--demo", "--llm", "none", "--no-news", "--sims", "20000"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    rep = json.loads((ROOT / "outputs" / "DEMO_quiniela_j12_detalle.json").read_text(encoding="utf-8"))
    assert rep["aviso"] and rep["fuentes"] and rep["convergencia_montecarlo_max_abs"] < 0.01
    assert len(rep["partidos"]) == 9
    m = rep["partidos"][0]
    for k in ("modelos", "senales", "razonamiento", "riesgos", "agente_altero_ensamble", "confianza", "pick"):
        assert k in m
    assert abs(sum(m["final"]) - 1) < 1e-3
    atl = next(x for x in rep["partidos"] if "Atlante" in x["partido"])
    assert any("Atlante" in a for a in atl["alertas"])  # bandera de muestra pequeña / cold start
