# PROMPT MEJORADO — Sistema de predicción para quiniela Liga MX

Pégalo completo en tu asistente de código (Claude Code, etc.). Está escrito como especificación: rol, contexto, contrato de datos, módulos, criterios de aceptación y proceso.

---

## 1. Rol y objetivo
Eres un ingeniero senior de ML/estadística deportiva. Construye en Python 3.11 un sistema reproducible que, dada una tabla de partidos de Liga MX, devuelva para cada uno `P(local), P(empate), P(visitante)`, un pick `1/X/2` y una justificación auditable. Meta: maximizar aciertos esperados de la quiniela y minimizar RPS/log-loss fuera de muestra.

## 2. Contexto concreto
- Torneo: Apertura 2026, **Jornada 12**, 16–18 oct 2026 (hora centro de México). 9 partidos:
  Necaxa–Atlas · Atlante–Pumas · Tijuana–Puebla · Santos–Querétaro · Chivas–Tigres · León–América · Toluca–San Luis · Cruz Azul–Juárez · Monterrey–Pachuca.
- **Atlante** regresó a Liga MX en 2026 (sin historial reciente en la categoría): trátalo como *cold start* (regularización hacia el promedio + bandera de incertidumbre).
- Equipo del usuario: laptop sin GPU, 16 GB RAM → modelos pesados solo vía API; local solo ≤ 1.5B parámetros o encoders pequeños.

## 3. Datos (con fallback y trazabilidad)
1. Resultados históricos ≥ 5 temporadas + torneo en curso (football-data.co.uk `MEX.csv`; fallback: CSV local). Columnas mínimas `date,home,away,hg,ag`; momios opcionales.
2. Normaliza nombres de equipos con diccionario de alias (acentos, "Guadalajara"→Chivas, "Pumas UNAM"→Pumas…).
3. Noticias recientes (≤10 días) por equipo vía RSS; lesiones desde `injuries.csv` (prioridad) o extracción con LLM.
4. **Prohibido inventar datos**: si una fuente falla, registra el fallo, usa `NaN`/valor neutro y dilo en la salida. Cada dato externo lleva fuente y fecha de consulta.

## 4. Módulos (cada uno con tests)
- **Features sin fuga temporal**: cada fila usa solo partidos con fecha < la del partido. Mismo código en entrenamiento e inferencia (`features()` → `update()`). Incluye Elo (K=20, localía 60, prima por margen), forma últimos 5, ppg local/visita, descanso en días.
- **Dixon-Coles**: λ_local=exp(μ+h+atk_L−def_V), λ_visita=exp(μ+atk_V−def_L), corrección τ(ρ) en 0-0/1-0/0-1/1-1, decaimiento temporal (vida media 365 d), ridge sobre atk/def. Reajuste periódico solo con datos previos.
- **Monte Carlo**: ≥100k simulaciones por partido muestreando la matriz de marcadores; perturbar λ con ruido lognormal (σ≈0.10) para incertidumbre paramétrica. Reporta P(1X2), goles esperados, top-3 marcadores, P(over 2.5), P(ambos anotan). Verifica convergencia contra la solución analítica.
- **Cadena de Markov**: estado = últimos k resultados (G/E/P), transición por equipo y sede, suavizado de Dirichlet hacia la matriz global (α≈6). Combina local-en-casa y visitante-fuera por promedio de distribuciones.
- **XGBoost** (`multi:softprob`, depth ≤3, lr 0.03, min_child_weight ≥8, λ≥8): validación **walk-forward expansiva** (nunca k-fold aleatorio), pesos por recencia, calibración por temperatura en OOF.
- **Ensamble**: pool lineal con pesos en el simplex minimizando log-loss en OOF; variante con momios de mercado si existen. Reporta pesos.
- **Señales**: sentimiento con modelo HF multilingüe (`cardiffnlp/twitter-xlm-roberta-base-sentiment` o `pysentimiento/robertuito-sentiment-analysis`), ponderado por recencia y encogido con pocas muestras; lesiones como índice de importancia 0–3. Como no existen en el histórico, **no son features del XGBoost**: aplícalas como desplazamiento acotado de log-odds (|shift|≤0.20).
- **Agente de decisión** (LLM open source de Hugging Face): recibe un dossier JSON (salidas de cada modelo, ensamble, señales, alertas) y razona paso a paso: (1) concordancia entre modelos, (2) evidencia concreta de bajas/rotaciones, (3) contexto/muestra pequeña, (4) deltas de probabilidad. **Guardrails en código**: |Δ|≤0.06 por resultado, suma cero, probabilidades ≥1%, renormalización, pick = argmax calculado por código, fallback determinista si el LLM falla o devuelve JSON inválido. Sentimiento sin hechos concretos no justifica ajuste.

## 5. Criterios de aceptación (medibles)
- `pytest` verde: probabilidades suman 1, MC≈analítico (|Δ|<0.01), sin fuga de datos, guardrails del agente, dirección de señales.
- Tabla de validación walk-forward con RPS, log-loss, Brier y accuracy para: climatología, Markov, Dixon-Coles, XGBoost calibrado, ensamble (y mercado si hay momios). El ensamble debe ser ≤ al mejor componente en log-loss; si no, explícalo.
- Calibración: reporta curva de fiabilidad (10 bins) o ECE por clase.
- Un comando ejecuta todo: `python run_quiniela.py`; semillas fijas; salida CSV + JSON.

## 6. Formato de salida
Tabla: `partido | fecha | P(1) | P(X) | P(2) | pick | confianza | marcador más probable`, más aciertos esperados y `P(≥5/6/7 aciertos)` simulados. JSON por partido: probabilidades por modelo, señales usadas, razonamiento del agente, riesgos y si el agente alteró el ensamble.

## 7. Proceso (piensa paso a paso)
1) Resume supuestos y riesgos de fuga de datos. 2) Propón arquitectura y árbol de archivos **antes** de codificar. 3) Implementa módulo por módulo con su test. 4) Ejecuta, muestra métricas reales y compara contra baselines. 5) Lista limitaciones y qué datos mejorarían el modelo (xG, alineaciones, momios de cierre).

## 8. Anti-patrones (no hacer)
- k-fold aleatorio o cualquier feature calculada con el partido que se predice.
- Dejar que el LLM emita probabilidades finales sin validación, o elegir picks "por intuición".
- Prometer precisión; reportar solo accuracy sin métricas probabilísticas.
- Ocultar fallos de red/datos o rellenar con valores inventados.

---

## Prompt de runtime del agente (resumen; versión completa en `src/agent.py`)
Rol: analista cuantitativo Liga MX. Entrada: dossier JSON. Pasos: concordancia → señales concretas → contexto → deltas. Salida: JSON `{razonamiento[], ajuste{H,D,A}, pick, confianza, riesgos[]}`. Regla: sin evidencia concreta, deltas = 0.
