"""Configuración central: rutas, modelos de Hugging Face, normalización de equipos y fixtures."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CACHE = ROOT / ".cache"
for _d in (DATA, OUT, CACHE):
    _d.mkdir(exist_ok=True)

# ---- Fuentes de datos -------------------------------------------------------
# CSV público con resultados históricos de Liga MX (+ momios Pinnacle/promedio).
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/MEX.csv"

# ---- Modelos open source de Hugging Face -----------------------------------
HF_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # multilingüe, corre en CPU
# Alternativa específica en español: "pysentimiento/robertuito-sentiment-analysis"
HF_AGENT_MODEL = "Qwen/Qwen2.5-72B-Instruct"   # vía HF Inference Providers (requiere HF_TOKEN)
HF_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # fallback local (CPU, menos preciso)

# ---- Hiperparámetros con justificación --------------------------------------
DC_HALF_LIFE_DAYS = 365          # decaimiento temporal de Dixon-Coles
DC_RIDGE = 0.03                  # regulariza equipos con pocos partidos (p. ej. Atlante)
ELO_K, ELO_HOME_ADV = 20.0, 60.0
MC_PARAM_SIGMA = 0.10            # incertidumbre epistémica sobre lambdas (lognormal)
SIGNAL_CAP = 0.20                # máx. desplazamiento de log-odds por lesiones/sentimiento
AGENT_MAX_DELTA = 0.06           # máx. ajuste (prob. absoluta) que el agente puede aplicar por resultado

CURRENT_TEAMS = [
    "Necaxa", "Atlas", "Atlante", "Pumas", "Tijuana", "Puebla", "Santos", "Querétaro", "Chivas",
    "Tigres", "León", "América", "Toluca", "San Luis", "Cruz Azul", "Juárez", "Monterrey", "Pachuca",
]
NEWS_QUERY = {
    "Necaxa": "Club Necaxa", "Atlas": "Atlas FC", "Atlante": "Atlante FC", "Pumas": "Pumas UNAM",
    "Tijuana": "Xolos Tijuana", "Puebla": "Club Puebla", "Santos": "Santos Laguna",
    "Querétaro": "Gallos Querétaro", "Chivas": "Chivas Guadalajara", "Tigres": "Tigres UANL",
    "León": "Club León", "América": "Club América", "Toluca": "Toluca FC", "San Luis": "Atlético San Luis",
    "Cruz Azul": "Cruz Azul", "Juárez": "FC Juárez", "Monterrey": "Rayados Monterrey", "Pachuca": "Pachuca Tuzos",
}

_ALIASES = [  # (subcadena sin acentos y en minúsculas, nombre canónico); orden = prioridad
    ("atlante", "Atlante"), ("atlas", "Atlas"), ("san luis", "San Luis"), ("chivas", "Chivas"),
    ("guadalajara", "Chivas"), ("tigres", "Tigres"), ("pumas", "Pumas"), ("monterrey", "Monterrey"),
    ("rayados", "Monterrey"), ("cruz azul", "Cruz Azul"), ("juarez", "Juárez"), ("queretaro", "Querétaro"),
    ("leon", "León"), ("tijuana", "Tijuana"), ("xolos", "Tijuana"), ("santos", "Santos"),
    ("necaxa", "Necaxa"), ("puebla", "Puebla"), ("toluca", "Toluca"), ("pachuca", "Pachuca"),
    ("america", "América"), ("mazatlan", "Mazatlán"),
]


def _strip(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn").lower().strip()


def canon(name: str) -> str:
    """Normaliza cualquier variante de nombre de equipo al nombre canónico."""
    s = _strip(name)
    for key, val in _ALIASES:
        if key in s:
            return val
    return str(name).strip()


def load_fixtures(path: Path | None = None) -> pd.DataFrame:
    """Fixtures de la quiniela. Columnas: date,time,home,away[,odds_h,odds_d,odds_a] (momios decimales opcionales)."""
    df = pd.read_csv(path or DATA / "fixtures_j12.csv", parse_dates=["date"])
    df["home"], df["away"] = df["home"].map(canon), df["away"].map(canon)
    for c in ("odds_h", "odds_d", "odds_a"):
        if c not in df:
            df[c] = float("nan")
    return df
