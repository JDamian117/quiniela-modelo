"""Carga de resultados históricos y actuales de Liga MX (football-data.co.uk) con caché y overrides manuales."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import provenance as prov
from .config import CACHE, DATA, FOOTBALL_DATA_URL, canon


def implied_probs(oh, od, oa) -> np.ndarray:
    """Momios decimales -> probabilidades sin margen (normalización proporcional)."""
    inv = 1.0 / np.column_stack([oh, od, oa]).astype(float)
    return inv / inv.sum(axis=1, keepdims=True)


def _download(refresh: bool) -> str:
    cache = CACHE / "MEX.csv"
    if cache.exists() and not refresh and time.time() - cache.stat().st_mtime < 12 * 3600:
        return cache.read_text(encoding="utf-8", errors="ignore")
    r = requests.get(FOOTBALL_DATA_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    cache.write_bytes(r.content)
    return r.content.decode("utf-8", errors="ignore")


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Acepta el formato de football-data (Home/Away/HG/AG/AvgH...) o uno propio (date,home,away,hg,ag)."""
    cols = {c.lower(): c for c in raw.columns}
    g = lambda *names: next((cols[n] for n in names if n in cols), None)  # noqa: E731
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[g("date")], dayfirst=True, errors="coerce"),
        "home": raw[g("home")].map(canon),
        "away": raw[g("away")].map(canon),
        "hg": pd.to_numeric(raw[g("hg")], errors="coerce"),
        "ag": pd.to_numeric(raw[g("ag")], errors="coerce"),
    })
    oh, od, oa = g("avgh", "psch", "ph"), g("avgd", "pscd", "pd"), g("avga", "psca", "pa")
    if oh and od and oa:
        for k, c in zip(("odds_h", "odds_d", "odds_a"), (oh, od, oa)):
            out[k] = pd.to_numeric(raw[c], errors="coerce")
    else:
        out["odds_h"] = out["odds_d"] = out["odds_a"] = np.nan
    return out.dropna(subset=["date", "hg", "ag"])


def load_history(offline_csv: str | Path | None = None, refresh: bool = False, years: float = 6.0) -> pd.DataFrame:
    """Devuelve partidos ordenados por fecha: date, home, away, hg, ag, odds_*, mkt_p{H,D,A}.

    Prioridad: CSV local (--offline-csv) > caché < 12 h > descarga. Luego añade data/manual_results.csv
    (para resultados muy recientes que la fuente aún no publica).
    """
    if offline_csv:
        raw, src = pd.read_csv(offline_csv), f"CSV local {offline_csv}"
    else:
        src = FOOTBALL_DATA_URL
        try:
            raw = pd.read_csv(io.StringIO(_download(refresh)))
        except Exception as e:  # sin red / bloqueo
            prov.record(FOOTBALL_DATA_URL, "fallo", f"descarga fallida: {type(e).__name__}")
            raise RuntimeError(
                f"No pude descargar {FOOTBALL_DATA_URL} ({e}). Descárgalo manualmente y usa --offline-csv ruta.csv"
            ) from e
    if "Country" in raw.columns:  # el archivo 'new' mezcla ligas; nos quedamos con México
        raw = raw[raw["Country"].astype(str).str.contains("Mexico", case=False, na=False)]
    df = standardize(raw)

    manual = DATA / "manual_results.csv"
    if manual.exists() and manual.stat().st_size > 40:
        m = standardize(pd.read_csv(manual))
        df = pd.concat([df, m], ignore_index=True).drop_duplicates(["date", "home", "away"], keep="last")
        prov.record("data/manual_results.csv", "ok", "resultados manuales añadidos", n=len(m))

    df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=int(365 * years))]
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    P = implied_probs(df["odds_h"], df["odds_d"], df["odds_a"])
    df[["mkt_pH", "mkt_pD", "mkt_pA"]] = P
    span = (df["date"].max() - df["date"].min()).days / 365.25
    prov.record(src, "ok", f"{len(df)} partidos {df['date'].min().date()} -> {df['date'].max().date()}", n=len(df))
    if span < 5:
        prov.record(src, "aviso", f"histórico de {span:.1f} años (< 5 recomendados en spec §3.1)")
    if df[["odds_h", "odds_d", "odds_a"]].isna().all(axis=None):
        prov.record(src, "neutro", "sin momios históricos: componente de mercado desactivado")
    return df
