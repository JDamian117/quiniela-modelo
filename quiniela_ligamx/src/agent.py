"""Agente de decisión. El LLM (open source, Hugging Face) NO inventa probabilidades: lee evidencia textual y propone
ajustes pequeños sobre el ensamble; el código valida, acota, renormaliza y calcula el pick (argmax)."""
from __future__ import annotations

import json
import os
import re

import numpy as np

from . import provenance as prov
from .config import AGENT_MAX_DELTA, HF_AGENT_MODEL, HF_LOCAL_MODEL

SYSTEM_DECISION = """Eres un analista cuantitativo de fútbol mexicano (Liga MX) que decide una quiniela 1/X/2.
Recibes un DOSSIER JSON con probabilidades de varios modelos (Dixon-Coles+Monte Carlo, cadena de Markov, XGBoost,
mercado si existe), el ensamble y señales (lesiones, sentimiento, titulares). Razona PASO A PASO y de forma breve:
1. Concordancia: ¿los modelos coinciden en el favorito? Si difieren >0.12 en algún resultado, di cuál es más fiable y por qué.
2. Señales: ¿hay evidencia CONCRETA (jugador, estatus, fuente en titular) de bajas/rotaciones/crisis que el modelo no vea?
   Sentimiento sin hechos NO justifica ajuste. Rumores o titulares ambiguos se ignoran.
3. Contexto: equipo recién ascendido/poca muestra (incertidumbre alta), cansancio, localía.
4. Ajuste: propón deltas de probabilidad absoluta por resultado (H=local, D=empate, A=visitante), suma ≈ 0 y
   |delta| <= %MAXD%. Si no hay evidencia concreta, devuelve ceros. Nunca justifiques un ajuste con tu intuición.
Responde SOLO un JSON válido:
{"razonamiento": ["paso 1...", "paso 2..."], "ajuste": {"H": 0.0, "D": 0.0, "A": 0.0},
 "pick": "1"|"X"|"2", "confianza": "alta"|"media"|"baja", "riesgos": ["..."]}"""

SYSTEM_INJURY = """Extrae bajas de jugadores SOLO de futbol a partir de titulares. Ignora 'baja' en otros sentidos
(precios, temperatura, directivos). Responde SOLO JSON: {"bajas": [{"jugador": str, "estatus":
"lesionado"|"suspendido"|"duda", "importancia": 0.1-1.0, "titular_idx": int}]}. importancia: 1.0 figura/portero titular,
0.5 titular, 0.2 rotación. Si no hay evidencia explícita devuelve {"bajas": []}."""


class LLM:
    """Cliente mínimo. mode='hf' -> HF Inference Providers (HF_TOKEN); 'local' -> transformers en CPU; 'none' -> sin LLM."""

    def __init__(self, mode: str = "hf", model: str | None = None):
        self.mode, self.pipe, self.client = mode, None, None
        self.model = model or (HF_LOCAL_MODEL if mode == "local" else HF_AGENT_MODEL)

    @property
    def enabled(self) -> bool:
        return self.mode in ("hf", "local")

    def chat(self, system: str, user: str, max_tokens: int = 700) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.mode == "hf":
            if self.client is None:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(model=self.model, token=os.getenv("HF_TOKEN"))
            r = self.client.chat_completion(messages=msgs, max_tokens=max_tokens, temperature=0.2)
            return r.choices[0].message.content
        if self.pipe is None:
            from transformers import pipeline
            self.pipe = pipeline("text-generation", model=self.model, device_map="auto")
        out = self.pipe(msgs, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def ask_json(llm: LLM, system: str, user: str, retries: int = 1) -> dict | None:
    for _ in range(retries + 1):
        try:
            d = parse_json(llm.chat(system, user))
        except Exception as e:
            prov.record(f"LLM {llm.mode}:{llm.model}", "fallo", f"{type(e).__name__}; se usa fallback determinista (ensamble)")
            return None
        if d is not None:
            return d
        user += "\n\nTu respuesta anterior no fue JSON válido. Devuelve SOLO el JSON."
    return None


def extract_injuries(llm: LLM, team: str, headlines: list[str]) -> list[dict]:
    """Etapa 1 (LLM como lector de texto): titulares -> bajas estructuradas. Se descuenta 30% por ser inferencia."""
    if not (llm.enabled and headlines):
        return []
    numbered = "\n".join(f"[{i}] {h}" for i, h in enumerate(headlines))
    d = ask_json(llm, SYSTEM_INJURY, f"Equipo: {team}\nTitulares:\n{numbered}") or {}
    out = []
    for b in d.get("bajas", []):
        try:
            out.append({"team": team, "player": str(b["jugador"]), "status": b.get("estatus", "duda"),
                        "importance": float(np.clip(float(b.get("importancia", 0.3)), 0.1, 1.0)) * 0.7,
                        "source": "llm"})
        except (KeyError, ValueError, TypeError):
            continue
    return out


def apply_agent_adjustment(base: np.ndarray, delta: dict | None, max_delta: float = AGENT_MAX_DELTA) -> np.ndarray:
    """Guardrails: acota cada delta, fuerza suma cero, mantiene probabilidades >= 1% y renormaliza."""
    if not delta:
        return base
    try:
        d = np.array([float(delta.get(k, 0.0)) for k in "HDA"])
    except (TypeError, ValueError):
        return base
    d = np.clip(d, -max_delta, max_delta)
    d -= d.mean()
    return (lambda q: q / q.sum())(np.clip(base + d, 0.01, None))


def decide(llm: LLM, dossier: dict, base: np.ndarray) -> dict:
    """Etapa 2: decisión final. Fallback determinista (sin LLM o respuesta inválida) = argmax del ensamble."""
    resp = ask_json(llm, SYSTEM_DECISION.replace("%MAXD%", str(AGENT_MAX_DELTA)),
                    json.dumps(dossier, ensure_ascii=False, default=float)) if llm.enabled else None
    final = apply_agent_adjustment(base, resp.get("ajuste") if resp else None)
    pick = "1X2"[int(final.argmax())]  # pick lo calcula el código: maximiza aciertos esperados
    return {
        "final": final, "pick": pick,
        "llm_pick": (resp or {}).get("pick"), "confidence": (resp or {}).get("confianza", "n/a"),
        "reasoning": (resp or {}).get("razonamiento", []), "risks": (resp or {}).get("riesgos", []),
        "agent_used": resp is not None,
    }
