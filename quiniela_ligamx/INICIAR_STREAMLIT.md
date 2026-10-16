# Iniciar interfaz Streamlit (Windows)

1. Extrae el ZIP completo.
2. Abre la carpeta `quiniela_ligamx`.
3. Haz doble clic en `INICIAR_STREAMLIT_WINDOWS.bat`.
4. La primera vez instalará dependencias; después abrirá Streamlit en el navegador.

También puedes hacerlo manualmente desde PowerShell en la carpeta del proyecto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

La opción Demo es la recomendada para la primera ejecución: usa datos sintéticos, desactiva noticias/LLM y evita llamadas de red. El control de simulaciones permite reducir el tiempo de Monte Carlo. La interfaz no cambia los módulos originales del modelo; agrega `app.py` como capa visual.

Para datos reales, el pipeline necesita conexión a internet para descargar el histórico. Si activas el agente de Hugging Face, configura `HF_TOKEN` como variable de entorno. El modo local puede consumir bastante RAM y ser lento en CPU.
