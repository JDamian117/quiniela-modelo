"""Señales contextuales: noticias recientes (Google News RSS), lesiones y análisis de sentimiento (modelo HF)."""
from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests

from . import provenance as prov
from .config import DATA, HF_SENTIMENT_MODEL, NEWS_QUERY

INJ_RE = re.compile(
    r"\b(lesi[oó]n|lesionad[oa]s?|baja[s]?|desgarre?|fractura|operad[oa]|esguince|molestia|suspendid[oa]|"
    r"expulsad[oa]|sanci[oó]n|descarta[dn]?|duda[s]?)\b", re.I)


def fetch_news(team: str, days: int = 10, limit: int = 20) -> list[dict]:
    """Titulares recientes del equipo. Devuelve [{title, source, age_days}]; lista vacía si falla la red."""
    q = f'{NEWS_QUERY.get(team, team)} Liga MX when:{days}d'
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=es-419&gl=MX&ceid=MX:es-419"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        items = ET.fromstring(r.content).findall(".//item")
    except Exception as e:
        prov.record(f"Google News RSS · {team}", "fallo", f"{type(e).__name__}; sentimiento neutro (0.0) y sin titulares")
        return []
    now, out = datetime.now(timezone.utc), []
    for it in items[:limit]:
        try:
            age = (now - parsedate_to_datetime(it.findtext("pubDate"))).total_seconds() / 86400
        except Exception:
            age = days / 2
        out.append({"title": (it.findtext("title") or "").strip(), "source": (it.findtext("source") or ""), "age_days": max(age, 0.0)})
    prov.record(f"Google News RSS · {team}", "ok" if out else "neutro", f"{len(out)} titulares ≤{days} d" if out else "0 titulares; sentimiento neutro", n=len(out))
    return out


class Sentiment:
    """Pipeline de clasificación de sentimiento (lazy). Score por texto = P(pos) - P(neg) en [-1, 1]."""

    def __init__(self, model: str = HF_SENTIMENT_MODEL):
        self.model, self.pipe = model, None

    def _load(self):
        from transformers import pipeline  # import perezoso: solo si se usan noticias
        self.pipe = pipeline("text-classification", model=self.model, top_k=None, truncation=True, max_length=128)

    def score(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        if self.pipe is None:
            self._load()
        out = []
        for res in self.pipe(texts):
            p = {d["label"].lower(): d["score"] for d in res}
            pos = next((v for k, v in p.items() if "pos" in k or k.endswith("2")), 0.0)
            neg = next((v for k, v in p.items() if "neg" in k or k.endswith("0")), 0.0)
            out.append(pos - neg)
        return np.array(out)


def aggregate_sentiment(scores: np.ndarray, ages: np.ndarray, tau_days: float = 5.0, shrink_n: float = 3.0) -> float:
    """Media ponderada por recencia, encogida hacia 0 cuando hay pocos titulares (evita señales ruidosas)."""
    if len(scores) == 0:
        return 0.0
    w = np.exp(-np.asarray(ages) / tau_days)
    return float((w * scores).sum() / w.sum() * len(scores) / (len(scores) + shrink_n))


def manual_injuries(team: str) -> list[dict]:
    """data/injuries.csv (team,player,status,importance 0-1) -> la vía MÁS confiable; tiene prioridad sobre titulares."""
    p = DATA / "injuries.csv"
    if not p.exists() or p.stat().st_size < 30:
        return []
    df = pd.read_csv(p)
    return df[df["team"].astype(str).str.lower() == team.lower()].to_dict("records")


def injury_index(injuries: list[dict]) -> float:
    return float(min(sum(float(i.get("importance", 0.3)) for i in injuries), 3.0))


def team_signals(team: str, sentiment: Sentiment | None, news: bool = True) -> dict:
    heads = fetch_news(team) if news else []
    scores = np.array([])
    if sentiment and heads:
        try:
            scores = sentiment.score([h["title"] for h in heads])
        except Exception as e:  # modelo HF no descargable / sin red
            prov.record(f"Sentimiento HF · {sentiment.model}", "fallo", f"{type(e).__name__}; sentimiento neutro (0.0)")
    s = aggregate_sentiment(scores, np.array([h["age_days"] for h in heads]))
    flagged = [h["title"] for h in heads if INJ_RE.search(h["title"])]
    inj = manual_injuries(team)
    prov.record(f"Lesiones · {team}", "ok" if inj else "neutro",
                "data/injuries.csv" if inj else "sin bajas registradas (índice 0); confirmar a mano", n=len(inj))
    return {"team": team, "sentiment": s, "n_headlines": len(heads), "headlines": [h["title"] for h in heads[:8]],
            "injury_headlines": flagged[:6], "injuries": inj, "injury_source": "manual" if inj else "none"}
