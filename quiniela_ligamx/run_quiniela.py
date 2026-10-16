#!/usr/bin/env python
"""Pipeline completo: datos -> features -> Dixon-Coles/Monte Carlo + Markov + XGBoost -> ensamble -> señales -> agente LLM.

Uso:
  python run_quiniela.py --demo --llm none --no-news          # prueba sin red con datos sintéticos
  python run_quiniela.py                                        # datos reales + noticias + agente HF (requiere HF_TOKEN)
  python run_quiniela.py --offline-csv MEX.csv --llm local      # CSV local + LLM local en CPU
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import provenance as prov
from src.agent import LLM, decide, extract_injuries
from src.config import OUT, load_fixtures
from src.data import implied_probs, load_history
from src.ensemble import apply_signals, blend, confidence, fit_weights, holdout_check
from src.features import FEATS, build_training_frame
from src.montecarlo import simulate_match, simulate_slate
from src.signals import Sentiment, injury_index, team_signals
from src.ratings import score_matrix, wdl
from src.xgb_model import _fit, calibrate, ece_by_class, fit_temperature, reliability_table, score, walk_forward_oof

LABEL = {"1": "Local", "X": "Empate", "2": "Visitante"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="historia sintética (sin red)")
    ap.add_argument("--offline-csv", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignora caché y vuelve a descargar")
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--llm", choices=["hf", "local", "none"], default="hf")
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--years", type=float, default=6.0)
    args = ap.parse_args()
    prov.clear()
    if args.demo:
        prov.record("--demo", "aviso", "datos SINTÉTICOS: métricas y picks NO son reales")
    if args.no_news:
        prov.record("noticias/sentimiento", "aviso", "desactivados (--no-news): sentimiento neutro")
    if args.llm == "none":
        prov.record("agente LLM", "aviso", "desactivado (--llm none): decisión = ensamble tras señales")

    # 1) Datos ------------------------------------------------------------------------------------
    if args.demo:
        from src.demo_data import make_history
        hist = make_history()
        hist[["mkt_pH", "mkt_pD", "mkt_pA"]] = implied_probs(hist["odds_h"], hist["odds_d"], hist["odds_a"])
    else:
        hist = load_history(args.offline_csv, args.refresh, args.years)
    fx = load_fixtures(args.fixtures) if args.fixtures else load_fixtures()
    print(f"[datos] {len(hist)} partidos ({hist.date.min().date()} -> {hist.date.max().date()}), {len(fx)} fixtures")

    # 2) Features sin fuga + XGBoost walk-forward ---------------------------------------------------
    df, eng = build_training_frame(hist)
    idx, P_xgb = walk_forward_oof(df)
    y = df.loc[idx, "y"].to_numpy()
    T = fit_temperature(P_xgb, y)
    P_xgb_cal = calibrate(P_xgb, T)
    model = _fit(df)
    sub = df.loc[idx]
    comps_oof = [sub[["dc_pH", "dc_pD", "dc_pA"]].to_numpy(), sub[["mk_pH", "mk_pD", "mk_pA"]].to_numpy(), P_xgb_cal]
    names = ["dc_mc", "markov", "xgboost"]
    w = fit_weights(comps_oof, y)
    mkt_mask = sub[["mkt_pH", "mkt_pD", "mkt_pA"]].notna().all(axis=1).to_numpy()
    w_mkt = None
    if mkt_mask.sum() >= 150:
        mk = sub[["mkt_pH", "mkt_pD", "mkt_pA"]].to_numpy()
        w_mkt = fit_weights([c[mkt_mask] for c in comps_oof] + [mk[mkt_mask]], y[mkt_mask])

    clim = np.tile(np.bincount(df["y"].iloc[:idx.min()], minlength=3) / idx.min(), (len(y), 1))
    table = {"climatología": score(clim, y), "markov": score(comps_oof[1], y), "dixon-coles": score(comps_oof[0], y),
             "xgboost(cal)": score(P_xgb_cal, y), "ENSAMBLE": score(blend(comps_oof, w), y)}
    if w_mkt is not None:
        table["mercado (subset)"] = score(mk[mkt_mask], y[mkt_mask])
    print(f"\n[validación walk-forward] n={len(y)}  T={T:.2f}  pesos {dict(zip(names, np.round(w, 3).tolist()))}")
    print(pd.DataFrame(table).T.round(4).to_string())
    P_ens_oof = blend(comps_oof, w)
    print("ECE por clase [1, X, 2] del ensamble:", ece_by_class(P_ens_oof, y).round(4).tolist())
    reliability_table(P_ens_oof, y).to_csv(OUT / f"{'DEMO_' if args.demo else ''}reliability.csv", index=False)
    hc = holdout_check(comps_oof, names, y)
    print(f"[holdout 60/40] log-loss ensamble={hc['ensamble']:.4f} | mejor componente ({hc['mejor_componente']})={hc[hc['mejor_componente']]:.4f}"
          f" -> {'OK: ensamble <= mejor componente' if hc['ensamble_le_mejor'] else 'ensamble NO supera al mejor componente'}")
    if not hc["ensamble_le_mejor"]:
        why = ("el componente dominante ya captura casi toda la señal y los pesos ajustados en la 1ª parte no generalizan; "
               "con muestra pequeña el ruido de estimar pesos supera la ganancia de diversificar")
        print("  explicación:", why)
        prov.record("ensamble", "aviso", f"no supera al mejor componente en holdout: {why}")

    # 3) Señales + agente ---------------------------------------------------------------------------
    llm = LLM(args.llm)
    sent = None if args.no_news else Sentiment()
    eng.refit_dc(fx["date"].min(), force=True)  # una sola vez, con TODA la historia disponible

    last_year = hist[hist["date"] >= hist["date"].max() - pd.Timedelta(days=365)]
    n_recent = pd.concat([last_year["home"], last_year["away"]]).value_counts()
    rows, detail = [], []
    mc_gap = 0.0
    sig_cache: dict[str, dict] = {}

    def signals(team: str) -> dict:
        if team not in sig_cache:
            s = team_signals(team, sent, news=not args.no_news)
            if not s["injuries"] and s["injury_headlines"]:
                s["injuries"] = extract_injuries(llm, team, s["injury_headlines"])
                s["injury_source"] = "llm" if s["injuries"] else "none"
            sig_cache[team] = s
        return sig_cache[team]

    for i, r in fx.iterrows():
        f = eng.features(r.home, r.away, r.date)
        p_xgb = calibrate(model.predict_proba(pd.DataFrame([f])[FEATS]), T)[0]
        lam, mu = eng.dc.lambdas(r.home, r.away)
        sim = simulate_match(lam, mu, eng.dc.rho, n=args.sims, seed=100 + i)
        p_mk = eng.mk.probs(r.home, r.away)
        mc_chk = simulate_match(lam, mu, eng.dc.rho, n=100_000, sigma=0.0, seed=7)["probs"]  # convergencia vs analítico
        mc_gap = max(mc_gap, float(np.abs(mc_chk - wdl(score_matrix(lam, mu, eng.dc.rho))).max()))
        comps = [sim["probs"], p_mk, p_xgb]
        weights = w
        if w_mkt is not None and not np.isnan(r[["odds_h", "odds_d", "odds_a"]].to_numpy(float)).any():
            comps.append(implied_probs(*r[["odds_h", "odds_d", "odds_a"]].to_numpy(float).reshape(3, 1))[0])
            weights = w_mkt
        p_ens = blend(comps, weights)

        sh, sa = signals(r.home), signals(r.away)
        p_sig = apply_signals(p_ens, injury_index(sh["injuries"]), injury_index(sa["injuries"]), sh["sentiment"], sa["sentiment"])
        dossier = {
            "partido": f"{r.home} vs {r.away}", "fecha": str(r.date.date()),
            "modelos": {"dixon_coles_montecarlo": sim["probs"].round(3).tolist(), "markov": p_mk.round(3).tolist(),
                        "xgboost": p_xgb.round(3).tolist(), "mercado": comps[3].round(3).tolist() if len(comps) > 3 else None},
            "formato_probs": "[local, empate, visitante]", "ensamble": p_ens.round(3).tolist(), "tras_senales": p_sig.round(3).tolist(),
            "goles_esperados": [round(g, 2) for g in sim["exp_goals"]], "marcadores_top": sim["top_scores"],
            "p_over_2_5": round(sim["p_over25"], 3), "p_ambos_anotan": round(sim["p_btts"], 3),
            "senales": {"local": {k: sh[k] for k in ("sentiment", "injuries", "injury_headlines", "headlines")},
                        "visitante": {k: sa[k] for k in ("sentiment", "injuries", "injury_headlines", "headlines")}},
            "alertas": [f"{t}: solo {int(n_recent.get(t, 0))} partidos en 12 meses en el histórico (muestra pequeña, incertidumbre alta)"
                        for t in (r.home, r.away) if n_recent.get(t, 0) < 20],
        }
        d = decide(llm, dossier, p_sig)
        fin = d["final"]
        altero = bool(np.abs(fin - p_sig).max() > 1e-6)
        rows.append({"partido": f"{r.home} vs {r.away}", "fecha": f"{r.date.date()} {r.time if isinstance(r.time, str) else ''}".strip(),
                     "P(1)": fin[0], "P(X)": fin[1], "P(2)": fin[2], "pick": d["pick"], "confianza": confidence(fin),
                     "ens_1": p_ens[0], "ens_X": p_ens[1], "ens_2": p_ens[2], "marcador_mas_probable": sim["top_scores"][0][0],
                     "agente": d["agent_used"], "agente_altero": altero})
        detail.append({**dossier, "final": fin.round(4).tolist(), "pick": d["pick"], "llm_pick": d["llm_pick"],
                       "confianza": confidence(fin), "confianza_llm": d["confidence"],
                       "agente_altero_ensamble": altero, "delta_agente": (fin - p_sig).round(4).tolist(),
                       "razonamiento": d["reasoning"], "riesgos": d["risks"]})

    res = pd.DataFrame(rows)
    picks = res["pick"].map({"1": 0, "X": 1, "2": 2}).to_numpy()
    slate = simulate_slate(res[["P(1)", "P(X)", "P(2)"]].to_numpy(), picks)

    # 4) Salida -------------------------------------------------------------------------------------
    show = res.copy()
    for c in ("P(1)", "P(X)", "P(2)"):
        show[c] = (show[c] * 100).round(1)
    print("\n=== QUINIELA LIGA MX — JORNADA 12 ===")
    print(show[["partido", "fecha", "P(1)", "P(X)", "P(2)", "pick", "confianza", "marcador_mas_probable"]].to_string(index=False))
    print(f"\nAciertos esperados: {slate['expected_hits']:.2f}/9 | "
          + " ".join(f"P(>={k})={slate['p_ge'][k]:.2f}" for k in (5, 6, 7)))
    tag = "DEMO_" if args.demo else ""
    res.to_csv(OUT / f"{tag}quiniela_j12.csv", index=False)
    report = {"aviso": "DATOS SINTÉTICOS (--demo): no son predicciones reales" if args.demo else None,
              "fuentes": prov.get(), "validacion_walk_forward": {k: {m: round(v, 5) for m, v in d_.items()} for k, d_ in table.items()},
              "pesos_ensamble": dict(zip(names, np.round(w, 4).tolist())), "temperatura_xgb": round(T, 3),
              "ece_por_clase_ensamble": ece_by_class(P_ens_oof, y).round(4).tolist(), "holdout_60_40": hc,
              "convergencia_montecarlo_max_abs": round(mc_gap, 5),
              "aciertos_esperados": round(slate["expected_hits"], 3), "p_ge": {str(k): round(v, 4) for k, v in slate["p_ge"].items()},
              "partidos": detail}
    (OUT / f"{tag}quiniela_j12_detalle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[Monte Carlo] convergencia: max |MC(σ=0) - analítico| = {mc_gap:.4f} ({'OK' if mc_gap < 0.01 else 'REVISAR'})")
    print(prov.summary())
    print(f"\nGuardado en {OUT}/{tag}quiniela_j12.csv y {tag}quiniela_j12_detalle.json")
    if args.demo:
        print("AVISO: modo --demo usa datos SINTÉTICOS; estos picks NO son predicciones reales.")


if __name__ == "__main__":
    main()
