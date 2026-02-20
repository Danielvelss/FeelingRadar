# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import time
import logging
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from pipeline_steps import (
    load_sentiment_model,
    run_paso_1,
    run_paso_2,
    load_topic_model,
    run_paso_3,
    run_paso_4,
    PREDEFINED_TOPICS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# LOAD CONFIG — st.secrets (Streamlit Cloud) with .env fallback (local)
# =============================================================================

def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

USE_LOCAL_MODEL = get_secret("USE_LOCAL_MODEL", "false").lower() == "true"
LOCAL_MODEL_PATH = get_secret("LOCAL_MODEL_PATH", "")
HF_TOKEN = get_secret("HF_TOKEN", "")
HF_MODEL_ID = get_secret("HF_MODEL_ID", "ejerez003/robertuito-guatemala-v2.0")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
S3_BUCKET = get_secret("AWS_S3_BUCKET", "")
S3_PREFIX = get_secret("AWS_S3_PREFIX", "pipeline-output/")
AWS_ACCESS_KEY = get_secret("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = get_secret("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = get_secret("AWS_REGION", "us-east-1")

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="DDI Social Listening Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM STYLES
# =============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #b8d4f0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; color: #2d3748; }
    .metric-card p { margin: 0.2rem 0 0 0; color: #718096; font-size: 0.85rem; }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-normal { background: #c6f6d5; color: #22543d; }
    .badge-attention { background: #fefcbf; color: #744210; }
    .badge-escalating { background: #fed7aa; color: #7b341e; }
    .badge-crisis { background: #fed7d7; color: #822727; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2332 0%, #243447 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown label,
    div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: #e2e8f0; }

    .preview-header {
        background: #edf2f7;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0;
        font-weight: 600;
        color: #2d3748;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #2d6a9f 0%, #3182ce 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INIT
# =============================================================================

def init_state():
    defaults = {
        "df_original": None,
        "df_paso_1": None,
        "df_paso_2": None,
        "df_paso_3": None,
        "df_paso_4": None,
        "metrics_paso_2": None,
        "topic_stats": None,
        "thread_analyses": None,
        "escalation_summary": None,
        "crisis_conversations": None,
        "sentiment_model": None,
        "topic_model": None,
        "pipeline_started": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# =============================================================================
# HELPERS
# =============================================================================

def df_to_csv_bytes(df):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding='utf-8')
    return buffer.getvalue()


def render_metric_cards(metrics_list):
    cols = st.columns(len(metrics_list))
    for col, (value, label) in zip(cols, metrics_list):
        col.markdown(f"""
        <div class="metric-card">
            <h3>{value}</h3>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def upload_to_s3(file_bytes, bucket, key, aws_access_key, aws_secret_key, aws_region):
    import boto3
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    s3.put_object(Bucket=bucket, Key=key, Body=file_bytes, ContentType='text/csv')
    return f"s3://{bucket}/{key}"


def get_phase():
    if st.session_state.df_paso_4 is not None:
        return "complete"
    if st.session_state.pipeline_started:
        return "running"
    if st.session_state.df_original is not None:
        return "ready"
    return "upload"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## DDI Pipeline")
    st.markdown("---")

    phase = get_phase()
    if phase == "complete":
        st.success("Pipeline completado")
    elif phase == "running":
        st.info("Pipeline en ejecucion...")
    elif phase == "ready":
        st.info("Archivo cargado — listo para ejecutar")
    else:
        st.info("Esperando archivo...")

    st.markdown("---")
    if st.button("Reiniciar Pipeline", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>DDI Social Listening Pipeline</h1>
    <p>Analisis de sentimiento, ground truth, topic detection y conversation clustering</p>
</div>
""", unsafe_allow_html=True)

phase = get_phase()


# =============================================================================
# PROGRESS INDICATOR
# =============================================================================

steps_names = ["Cargar Archivo", "Sentimiento", "Ground Truth", "Topics", "Clustering"]
step_icons = ["📂", "🧠", "✅", "🏷️", "🔗"]
step_data_keys = [None, "df_paso_1", "df_paso_2", "df_paso_3", "df_paso_4"]

cols = st.columns(len(steps_names))
for i, (col, name, icon) in enumerate(zip(cols, steps_names, step_icons)):
    if phase == "complete":
        status = "done"
    elif phase == "upload":
        status = "active" if i == 0 else "pending"
    elif phase == "ready":
        status = "done" if i == 0 else "pending"
    else:
        if i == 0:
            status = "done"
        elif step_data_keys[i] and st.session_state.get(step_data_keys[i]) is not None:
            status = "done"
        elif i == 1 or (step_data_keys[i - 1] and st.session_state.get(step_data_keys[i - 1]) is not None):
            status = "active"
        else:
            status = "pending"

    if status == "done":
        col.markdown(f"<div style='text-align:center;padding:0.5rem;background:#c6f6d5;border-radius:8px;color:#22543d;'><b>{icon} {name}</b><br><small>Completado</small></div>", unsafe_allow_html=True)
    elif status == "active":
        col.markdown(f"<div style='text-align:center;padding:0.5rem;background:#bee3f8;border-radius:8px;border:2px solid #3182ce;color:#2a4365;'><b>{icon} {name}</b><br><small>En proceso</small></div>", unsafe_allow_html=True)
    else:
        col.markdown(f"<div style='text-align:center;padding:0.5rem;background:#edf2f7;border-radius:8px;color:#4a5568;'><b>{icon} {name}</b><br><small style='color:#a0aec0;'>Pendiente</small></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# PHASE: UPLOAD
# =============================================================================

if phase == "upload":
    st.markdown("### 📂 Cargar archivo de menciones")
    st.info("Sube el archivo CSV o Excel con las menciones de Brandwatch. Debe contener la columna **Comentario**.")

    uploaded = st.file_uploader(
        "Selecciona tu archivo",
        type=["csv", "xlsx", "xls"],
        help="DDI_Brandwatch_PoC_enriched - MENTIONS_STREAM.csv"
    )

    if uploaded:
        try:
            if uploaded.name.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded)
            else:
                try:
                    df = pd.read_csv(uploaded, sep=',')
                except UnicodeDecodeError:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, sep=',', encoding='latin-1')

            if 'Comentario' not in df.columns:
                st.error(f"La columna **Comentario** no fue encontrada. Columnas disponibles: {', '.join(df.columns.tolist())}")
            else:
                st.session_state.df_original = df
                st.rerun()

        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")


# =============================================================================
# PHASE: READY — File loaded, verify & launch
# =============================================================================

elif phase == "ready":
    df = st.session_state.df_original

    empty_count = (df['Comentario'].fillna("").astype(str).str.strip() == "").sum()
    render_metric_cards([
        (f"{len(df)}", "Menciones cargadas"),
        (f"{len(df.columns)}", "Columnas"),
        (f"{len(df) - empty_count}", "Con texto"),
        (f"{empty_count}", "Sin texto"),
    ])

    with st.expander("Vista previa del archivo cargado", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=350)

    st.markdown("---")
    st.markdown("### Verificacion de configuracion")

    issues = []
    can_sentiment = (USE_LOCAL_MODEL and LOCAL_MODEL_PATH) or (not USE_LOCAL_MODEL and HF_TOKEN)
    if not can_sentiment:
        issues.append("**Paso 1 (Sentimiento):** Configura `HF_TOKEN` o `LOCAL_MODEL_PATH` en `.env`")
    if not OPENAI_API_KEY:
        issues.append("**Paso 2 (Ground Truth):** Configura `OPENAI_API_KEY` en `.env`")

    if issues:
        for issue in issues:
            st.warning(issue)
        st.error("Corrige la configuracion antes de ejecutar el pipeline.")
    else:
        st.success("Configuracion valida. Todos los pasos pueden ejecutarse.")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Ejecutar Pipeline Completo", type="primary", use_container_width=True):
                st.session_state.pipeline_started = True
                st.rerun()


# =============================================================================
# PHASE: RUNNING — Execute all 4 steps automatically
# =============================================================================

elif phase == "running":
    df = st.session_state.df_original

    st.markdown("### Ejecutando Pipeline...")
    overall = st.progress(0, text="Iniciando pipeline...")

    error_occurred = False

    # -----------------------------------------------------------------
    # STEP 1: Sentimiento
    # -----------------------------------------------------------------
    if not error_occurred:
        overall.progress(0.0, text="Paso 1/4: Analisis de Sentimiento...")

        if st.session_state.df_paso_1 is None:
            with st.status("Paso 1: Analisis de Sentimiento con RoBERTuito V2.0", expanded=True) as status:
                try:
                    st.write("Cargando modelo RoBERTuito V2.0...")
                    if st.session_state.sentiment_model is None:
                        st.session_state.sentiment_model = load_sentiment_model(
                            USE_LOCAL_MODEL, LOCAL_MODEL_PATH, HF_TOKEN, HF_MODEL_ID
                        )
                    st.write("Modelo cargado. Procesando menciones...")

                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress_1(current_val, total_val):
                        pct = current_val / total_val
                        progress_bar.progress(pct)
                        progress_text.text(f"Procesando mencion {current_val}/{total_val}")

                    result = run_paso_1(df, st.session_state.sentiment_model, update_progress_1)
                    st.session_state.df_paso_1 = result
                    status.update(label="Paso 1: Sentimiento completado", state="complete")
                except Exception as e:
                    status.update(label="Error en Paso 1", state="error")
                    st.error(f"Error: {e}")
                    error_occurred = True
        else:
            st.success("Paso 1: Sentimiento — ya completado")

    # -----------------------------------------------------------------
    # STEP 2: Ground Truth
    # -----------------------------------------------------------------
    if not error_occurred:
        overall.progress(0.25, text="Paso 2/4: Ground Truth...")

        if st.session_state.df_paso_2 is None:
            df1 = st.session_state.df_paso_1
            with st.status("Paso 2: Ground Truth con OpenAI GPT-3.5", expanded=True) as status:
                try:
                    st.write("Conectando con OpenAI API...")

                    progress_bar_2 = st.progress(0)
                    progress_text_2 = st.empty()

                    def update_progress_2(current_val, total_val):
                        pct = current_val / total_val
                        progress_bar_2.progress(pct)
                        progress_text_2.text(f"Validando mencion {current_val}/{total_val}")

                    result_df, result_metrics = run_paso_2(df1, OPENAI_API_KEY, update_progress_2)
                    st.session_state.df_paso_2 = result_df
                    st.session_state.metrics_paso_2 = result_metrics
                    status.update(label="Paso 2: Ground Truth completado", state="complete")
                except Exception as e:
                    status.update(label="Error en Paso 2", state="error")
                    st.error(f"Error: {e}")
                    error_occurred = True
        else:
            st.success("Paso 2: Ground Truth — ya completado")

    # -----------------------------------------------------------------
    # STEP 3: Topic Detection
    # -----------------------------------------------------------------
    if not error_occurred:
        overall.progress(0.50, text="Paso 3/4: Topic Detection...")

        if st.session_state.df_paso_3 is None:
            df2 = st.session_state.df_paso_2
            with st.status("Paso 3: Topic Detection Hibrido", expanded=True) as status:
                try:
                    st.write("Cargando clasificador Zero-Shot (facebook/bart-large-mnli)...")
                    if st.session_state.topic_model is None:
                        st.session_state.topic_model = load_topic_model()
                    st.write("Modelo cargado. Clasificando menciones...")

                    progress_bar_3 = st.progress(0)
                    progress_text_3 = st.empty()

                    def update_progress_3(current_val, total_val):
                        pct = current_val / total_val
                        progress_bar_3.progress(pct)
                        progress_text_3.text(f"Clasificando mencion {current_val}/{total_val}")

                    result_df, result_stats = run_paso_3(
                        df2, st.session_state.topic_model,
                        openai_api_key=OPENAI_API_KEY if OPENAI_API_KEY else None,
                        progress_callback=update_progress_3
                    )
                    st.session_state.df_paso_3 = result_df
                    st.session_state.topic_stats = result_stats
                    status.update(label="Paso 3: Topics completado", state="complete")
                except Exception as e:
                    status.update(label="Error en Paso 3", state="error")
                    st.error(f"Error: {e}")
                    error_occurred = True
        else:
            st.success("Paso 3: Topics — ya completado")

    # -----------------------------------------------------------------
    # STEP 4: Clustering + S3
    # -----------------------------------------------------------------
    if not error_occurred:
        overall.progress(0.75, text="Paso 4/4: Clustering + S3...")

        if st.session_state.df_paso_4 is None:
            df3 = st.session_state.df_paso_3
            with st.status("Paso 4: Conversation Clustering + S3", expanded=True) as status:
                try:
                    st.write("Analizando threads y detectando crisis...")

                    progress_bar_4 = st.progress(0)
                    progress_text_4 = st.empty()

                    def update_progress_4(current_val, total_val):
                        pct = current_val / total_val
                        progress_bar_4.progress(pct)
                        progress_text_4.text(f"Analizando thread {current_val}/{total_val}")

                    result_df, analyses, esc_summary, crisis_convs = run_paso_4(df3, update_progress_4)
                    st.session_state.df_paso_4 = result_df
                    st.session_state.thread_analyses = analyses
                    st.session_state.escalation_summary = esc_summary
                    st.session_state.crisis_conversations = crisis_convs

                    can_upload = all([S3_BUCKET, AWS_ACCESS_KEY, AWS_SECRET_KEY])
                    if can_upload:
                        st.write("Subiendo archivo final a S3...")
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            s3_key = f"{S3_PREFIX.rstrip('/')}/{timestamp}_04_CONVERSATION_THREADS_CLUSTERED.csv"
                            csv_bytes = df_to_csv_bytes(result_df)
                            s3_uri = upload_to_s3(csv_bytes, S3_BUCKET, s3_key, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION)
                            st.write(f"Archivo subido a {s3_uri}")
                        except Exception as e:
                            st.warning(f"Clustering completado pero hubo un error subiendo a S3: {e}")

                    status.update(label="Paso 4: Clustering completado", state="complete")
                except Exception as e:
                    status.update(label="Error en Paso 4", state="error")
                    st.error(f"Error: {e}")
                    error_occurred = True
        else:
            st.success("Paso 4: Clustering — ya completado")

    # -----------------------------------------------------------------
    # FINAL: rerun to dashboard or show error
    # -----------------------------------------------------------------
    if not error_occurred:
        overall.progress(1.0, text="Pipeline completado exitosamente")
        st.session_state.pipeline_started = False
        time.sleep(1)
        st.rerun()
    else:
        overall.progress(0, text="Pipeline detenido por error")
        st.markdown("---")
        st.warning("El pipeline se detuvo por un error. Los pasos ya completados se conservan. Corrige el problema y haz clic en reintentar.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reintentar desde el paso fallido", type="primary", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("Reiniciar todo", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


# =============================================================================
# PHASE: COMPLETE — Results dashboard
# =============================================================================

elif phase == "complete":
    st.markdown("""
    <div style="text-align:center; padding:1rem; background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); border-radius:12px; margin-bottom:1rem;">
        <h2 style="color:#22543d; margin:0;">Pipeline Completado</h2>
        <p style="color:#276749; margin:0.3rem 0 0 0;">Todos los pasos han sido ejecutados exitosamente.</p>
    </div>
    """, unsafe_allow_html=True)

    df4 = st.session_state.df_paso_4
    esc = st.session_state.escalation_summary

    render_metric_cards([
        (f"{len(st.session_state.df_original)}", "Menciones procesadas"),
        (f"{esc['normal']}", "Normal"),
        (f"{esc['attention']}", "Atencion"),
        (f"{esc['escalating']}", "Escalando"),
        (f"{esc['crisis']}", "Crisis"),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sentimiento", "Ground Truth", "Topics", "Clustering", "Descargas"
    ])

    # ---- Tab 1: Sentimiento ----
    with tab1:
        df1 = st.session_state.df_paso_1
        dist = df1['sentiment_v2'].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=dist.values, names=dist.index,
                title="Distribucion de Sentimiento",
                color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4,
            )
            fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(
                df1, x='sentiment_v2_score', nbins=30,
                title="Distribucion de Confianza",
                color_discrete_sequence=['#3182ce'],
            )
            fig2.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Vista previa — Sentimiento", expanded=False):
            st.dataframe(df1.head(50), use_container_width=True, height=400)

    # ---- Tab 2: Ground Truth ----
    with tab2:
        df2 = st.session_state.df_paso_2
        metrics = st.session_state.metrics_paso_2

        if metrics and 'metrics' in metrics:
            col1, col2 = st.columns(2)
            with col1:
                if 'distribution' in metrics:
                    dist_v2 = metrics['distribution'].get('v2_0', {})
                    dist_gt = metrics['distribution'].get('ground_truth', {})
                    labels_all = sorted(set(list(dist_v2.keys()) + list(dist_gt.keys())))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='RoBERTuito V2', x=labels_all,
                        y=[dist_v2.get(l, 0) for l in labels_all], marker_color='#3182ce',
                    ))
                    fig.add_trace(go.Bar(
                        name='Ground Truth', x=labels_all,
                        y=[dist_gt.get(l, 0) for l in labels_all], marker_color='#38a169',
                    ))
                    fig.update_layout(barmode='group', title="V2 vs Ground Truth", margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    fig_cm = px.imshow(
                        cm['matrix'], x=cm['labels'], y=cm['labels'],
                        labels=dict(x="Prediccion V2", y="Ground Truth", color="Cantidad"),
                        color_continuous_scale="Blues", title="Matriz de Confusion",
                    )
                    fig_cm.update_layout(margin=dict(t=40, b=20))
                    st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("Vista previa — Ground Truth", expanded=False):
            st.dataframe(df2.head(50), use_container_width=True, height=400)

    # ---- Tab 3: Topics ----
    with tab3:
        df3 = st.session_state.df_paso_3
        stats = st.session_state.topic_stats

        col1, col2 = st.columns(2)
        with col1:
            if stats and 'topic_distribution' in stats:
                top_topics = dict(sorted(
                    stats['topic_distribution'].items(), key=lambda x: x[1], reverse=True
                )[:15])
                fig = px.bar(
                    x=list(top_topics.values()), y=list(top_topics.keys()), orientation='h',
                    title="Top 15 Temas Detectados", color_discrete_sequence=['#2d6a9f'],
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=40, b=20, l=150))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if stats and 'method_distribution' in stats:
                md = stats['method_distribution']
                fig2 = px.pie(
                    values=list(md.values()), names=list(md.keys()),
                    title="Metodos Utilizados",
                    hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig2.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        if stats and stats.get('emergent_themes'):
            st.info(f"**Temas emergentes descubiertos:** {', '.join(stats['emergent_themes'])}")

        with st.expander("Vista previa — Topics", expanded=False):
            st.dataframe(df3.head(50), use_container_width=True, height=400)

    # ---- Tab 4: Clustering ----
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            esc_data = {k: v for k, v in esc.items()}
            fig = px.pie(
                values=list(esc_data.values()), names=list(esc_data.keys()),
                title="Niveles de Escalation",
                color=list(esc_data.keys()),
                color_discrete_map={
                    "normal": "#48bb78", "attention": "#ecc94b",
                    "escalating": "#ed8936", "crisis": "#fc8181",
                },
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            crisis = st.session_state.crisis_conversations
            if crisis:
                crisis_df = pd.DataFrame([{
                    "Thread": c['thread_id'][:30],
                    "Tema": c['primary_topic'],
                    "Menciones": c['mention_count'],
                    "Risk Score": c['crisis_risk_score'],
                    "% Negativo": f"{c['sentiment_analysis']['negative_ratio']*100:.0f}%",
                } for c in crisis])
                st.markdown("**Conversaciones en Crisis:**")
                st.dataframe(crisis_df, use_container_width=True)
            else:
                st.info("No se detectaron conversaciones en crisis.")

        with st.expander("Vista previa — Clustering", expanded=False):
            st.dataframe(df4.head(50), use_container_width=True, height=400)

    # ---- Tab 5: Descargas ----
    with tab5:
        st.markdown("### Descargar resultados por paso")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Paso 1: Sentimiento",
                df_to_csv_bytes(st.session_state.df_paso_1),
                "01_MENCIONES_CON_SENTIMIENTO_V2.csv",
                "text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Paso 3: Topics",
                df_to_csv_bytes(st.session_state.df_paso_3),
                "03_MENCIONES_CON_TOPICS.csv",
                "text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Paso 2: Ground Truth",
                df_to_csv_bytes(st.session_state.df_paso_2),
                "02_GROUND_TRUTH_CON_METRICAS.csv",
                "text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Paso 4: Clustering (Final)",
                df_to_csv_bytes(st.session_state.df_paso_4),
                "04_CONVERSATION_THREADS_CLUSTERED.csv",
                "text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.download_button(
            "Descargar Resultado Final Completo",
            df_to_csv_bytes(st.session_state.df_paso_4),
            "04_CONVERSATION_THREADS_CLUSTERED.csv",
            "text/csv",
            use_container_width=True,
            key="download_final",
        )
