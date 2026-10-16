"""Interfaz Streamlit para el motor original de quiniela_ligamx.
No altera los módulos de modelado: ejecuta run_quiniela.py como subproceso.
"""
from __future__ import annotations

import json
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="Quiniela Liga MX · IA", page_icon="⚽", layout="wide")
st.title("⚽ Quiniela Liga MX — análisis con IA")
st.caption("Interfaz Streamlit para el pipeline original. Las predicciones son probabilísticas, no garantías.")

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Fuente de datos", ["Demo sintética (rápida y sin internet)", "Datos reales"], index=0)
    sims = st.select_slider("Simulaciones Monte Carlo por partido", options=[2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], value=5_000,
                            help="Menos simulaciones suelen terminar más rápido; 10,000 es un punto de partida práctico.")
    if mode.startswith("Demo"):
        llm_choice = "none"
        no_news = True
        st.info("La demo usa datos sintéticos y desactiva noticias/LLM para evitar red y llamadas externas.")
    else:
        llm_choice = st.selectbox("Agente LLM", ["none", "hf", "local"], index=0,
                                  help="HF requiere HF_TOKEN. local descarga/ejecuta un modelo y puede ser lento en CPU.")
        no_news = st.checkbox("Desactivar noticias/sentimiento (más rápido)", value=True)
        years = st.slider("Años de histórico", min_value=2, max_value=12, value=6)

@st.cache_data(show_spinner=False, ttl=3600)
def execute_pipeline(demo: bool, n_sims: int, llm: str, disable_news: bool, years: float = 6.0):
    cmd = [sys.executable, "run_quiniela.py", "--sims", str(n_sims), "--llm", llm]
    if demo:
        cmd += ["--demo"]
    else:
        cmd += ["--years", str(years)]
    if disable_news:
        cmd += ["--no-news"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    output_tag = "DEMO_" if demo else ""
    csv_path = OUT / f"{output_tag}quiniela_j12.csv"
    json_path = OUT / f"{output_tag}quiniela_j12_detalle.json"
    csv_text = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
    report_text = json_path.read_text(encoding="utf-8") if json_path.exists() else "{}"
    return proc.returncode, proc.stdout, proc.stderr, csv_text, report_text, cmd

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if st.button("🚀 Ejecutar quiniela", type="primary", use_container_width=True):
    is_demo = mode.startswith("Demo")
    selected_years = 6.0 if is_demo else years
    with st.spinner("Ejecutando modelos y simulaciones… la primera ejecución puede tardar más"):
        try:
            st.session_state.last_result = execute_pipeline(is_demo, sims, llm_choice, no_news, selected_years)
        except subprocess.TimeoutExpired:
            st.error("El proceso superó los 30 minutos. Prueba con menos simulaciones o desactiva noticias/LLM.")
        except Exception as exc:
            st.error(f"No se pudo ejecutar el pipeline: {exc}")

result = st.session_state.last_result
if result:
    code, stdout, stderr, csv_text, report_text, cmd = result
    if code != 0:
        st.error(f"El programa terminó con código {code}.")
        if stderr:
            st.code(stderr[-12000:])
    elif csv_text:
        res = pd.read_csv(io.StringIO(csv_text))
        report = json.loads(report_text or "{}")
        if "--demo" in cmd:
            st.warning("MODO DEMO: los datos y picks son sintéticos. No representan partidos ni predicciones reales.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Partidos procesados", len(res))
        m2.metric("Simulaciones por partido", f"{sims:,}")
        m3.metric("Aciertos esperados", f"{report.get('aciertos_esperados', float('nan')):.2f}/9")
        st.subheader("Pronósticos")
        view = res.copy()
        for col in ["P(1)", "P(X)", "P(2)"]:
            if col in view:
                view[col] = (view[col] * 100).round(1).astype(str) + "%"
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.subheader("Validación del modelo")
        validation = report.get("validacion_walk_forward", {})
        if validation:
            st.dataframe(pd.DataFrame(validation).T, use_container_width=True)
        with st.expander("Registro de ejecución"):
            st.code(" ".join(cmd))
            st.text(stdout[-16000:] if stdout else "Sin salida estándar.")
            if stderr:
                st.text_area("Advertencias / errores", stderr[-8000:], height=180)
        st.download_button("Descargar pronósticos CSV", data=csv_text, file_name="quiniela_j12.csv", mime="text/csv")
        st.download_button("Descargar informe JSON", data=report_text, file_name="quiniela_j12_detalle.json", mime="application/json")
    else:
        st.warning("El programa terminó, pero no se encontró el CSV de salida. Revisa el registro.")
        st.text(stdout[-12000:])
else:
    st.markdown("### Cómo usarla")
    st.markdown("1. Elige **Demo sintética** para comprobar la interfaz sin servicios externos.\n2. Ajusta el número de simulaciones.\n3. Pulsa **Ejecutar quiniela** y consulta/descarga los resultados.")
    st.caption("La interfaz mejora la facilidad de uso y evita repetir ejecuciones iguales gracias a caché. Streamlit por sí solo no acelera los modelos; bajar las simulaciones sí reduce el tiempo de Monte Carlo.")
