# Quiniela Liga MX — Jornada 12 (Apertura 2026)

Pipeline probabilístico 1/X/2: **Dixon-Coles + Monte Carlo + cadena de Markov + XGBoost → ensamble → señales (lesiones, sentimiento) → agente LLM con guardrails**.

## Arquitectura

```
datos (football-data CSV + manual_results.csv)
  └─ FeatureEngine (sin fuga: solo info anterior a cada partido)
       ├─ Elo incremental ─────────────┐
       ├─ Dixon-Coles (decay 365 d) ───┼─► XGBoost multiclase (walk-forward, calibración por temperatura)
       ├─ Markov de forma (G/E/P) ─────┤
       └─ forma 5p, localía, descanso ─┘
Monte Carlo: 200k marcadores/partido con incertidumbre en λ (lognormal σ=0.10)
Ensamble: pool lineal, pesos óptimos por log-loss en OOF (con/sin momios)
Señales: lesiones (CSV manual > LLM sobre titulares) + sentimiento HF  → shift log-odds acotado (±0.20)
Agente (HF): lee dossier → propone deltas (|Δ|≤0.06, suma 0) → el CÓDIGO valida, renormaliza y hace argmax
```

## Uso

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx                      # solo para --llm hf
python -m pytest -q                         # 14 tests (incluye contrato de salida end-to-end)
python run_quiniela.py --demo --llm none --no-news   # prueba sin red (datos SINTÉTICOS)
python run_quiniela.py                      # real: datos + noticias + agente HF
python run_quiniela.py --offline-csv MEX.csv --llm local --no-news
```

Salidas: `outputs/quiniela_j12.csv` (probabilidades, pick, confianza, `agente_altero`), `outputs/quiniela_j12_detalle.json` (fuentes, métricas, ECE por clase, holdout, convergencia MC, dossier y razonamiento por partido) y `outputs/reliability.csv` (curva de fiabilidad, 10 bins). En `--demo` los archivos llevan prefijo `DEMO_`.

## Supuestos y riesgos de fuga (spec §7.1)

- Fuente histórica: CSV football-data (`/new/MEX.csv`) o CSV local. Cada fuente externa se registra con fecha de consulta y estado en `outputs/quiniela_j12_detalle.json → fuentes`; un fallo de red **no se oculta**: queda como `fallo`/`neutro` y el resumen se imprime al final.
- Fuga controlada: Dixon-Coles se reajusta con datos previos; Elo/Markov/forma se actualizan **después** de predecir.
- Optimismo residual: temperatura XGB y pesos del ensamble se ajustan sobre las mismas OOF que se evalúan → se reporta un **holdout 60/40** (pesos en el primer 60%, evaluación en el 40% posterior) y si el ensamble no supera al mejor componente se explica.
- Reglas propias (el prompt no las fija): confianza por margen top1−top2 (alta ≥ 0.15, media ≥ 0.07, baja < 0.07); alerta de muestra pequeña si un equipo tiene < 20 partidos en 12 meses.

## Archivos de entrada que tú controlas

| Archivo | Para qué |
|---|---|
| `data/fixtures_j12.csv` | Partidos. Opcional `odds_h,odds_d,odds_a` (momios decimales) → activa el componente de mercado |
| `data/injuries.csv` | `team,player,status,importance(0-1)`. **Vía más confiable** de lesiones; tiene prioridad sobre el LLM |
| `data/manual_results.csv` | `date,home,away,hg,ag` — resultados recientes que la fuente aún no publica |

Importancia: 1.0 figura/portero titular · 0.5 titular · 0.2 rotación.

## Decisiones de diseño

- **Sin fuga**: cada fila usa solo partidos anteriores; Dixon-Coles se reajusta cada 28 días con datos previos.
- **Lesiones/sentimiento no son features de XGBoost**: no existen en el histórico → se aplican como prior acotado (≤ ~5 pts).
- **El LLM no inventa números**: solo propone deltas; `apply_agent_adjustment` los acota; el pick = argmax calculado en código. Si el LLM falla → fallback determinista (ensamble).
- **Pick = argmax por partido** maximiza aciertos esperados (suma lineal). El empate rara vez es argmax; solo sale cuando P(X) supera a ambos.
- **Atlante** (regresó a Liga MX en 2026): poca muestra; ridge de Dixon-Coles + prior de liga en Markov lo regresan al promedio y el dossier lo marca como alerta.
- **Métricas**: RPS, log-loss, Brier, accuracy vs climatología / Markov / Dixon-Coles / mercado, en validación walk-forward.

## Hardware / modelos

Sentimiento `cardiffnlp/twitter-xlm-roberta-base-sentiment` corre en CPU. El agente por defecto usa `Qwen/Qwen2.5-72B-Instruct` vía API de Hugging Face (`config.HF_AGENT_MODEL`; verifica que tu proveedor lo sirva o cambia a otro Instruct). `--llm local` usa `Qwen2.5-1.5B-Instruct` en CPU: funciona, pero razona peor.

## Limitaciones honestas

- Liga MX tiene alta entropía: aun un buen modelo ronda ~50% de acierto 1X2 (≈ 4–5 de 9). No hay garantía de ganar.
- `FOOTBALL_DATA_URL` (`/new/MEX.csv`) no pudo verificarse desde el entorno de desarrollo; si falla la descarga, usa `--offline-csv` con cualquier CSV con `date,home,away,hg,ag`.
- Probado de punta a punta solo con datos sintéticos + tests unitarios; las métricas reales dependerán de tu histórico.
- Titulares con "baja/duda" son ruidosos; el LLM los filtra pero conviene confirmar bajas clave a mano en `data/injuries.csv`.
