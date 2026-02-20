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

    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .step-active { border-left: 4px solid #2d6a9f; }
    .step-done { border-left: 4px solid #38a169; background: #f0fff4; }

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
        "current_step": 0,
        "sentiment_model": None,
        "topic_model": None,
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


def show_preview(df, title, key_suffix=""):
    st.markdown(f'<div class="preview-header">Vista previa: {title} ({len(df)} filas x {len(df.columns)} columnas)</div>', unsafe_allow_html=True)

    col_search, col_rows = st.columns([3, 1])
    with col_search:
        filter_text = st.text_input("Filtrar filas", "", key=f"filter_{key_suffix}", placeholder="Escribe para buscar en todas las columnas...")
    with col_rows:
        n_rows = st.selectbox("Filas", [10, 25, 50, 100], index=0, key=f"rows_{key_suffix}")

    display_df = df.copy()
    if filter_text:
        mask = display_df.astype(str).apply(lambda row: row.str.contains(filter_text, case=False, na=False).any(), axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df.head(n_rows), use_container_width=True, height=400)

    cols_info = st.columns(min(6, len(df.columns)))
    new_cols = [c for c in df.columns if c not in (st.session_state.get("_prev_cols", []))]
    if new_cols:
        st.markdown("**Columnas agregadas en este paso:** " + ", ".join(f"`{c}`" for c in new_cols))
    st.session_state["_prev_cols"] = list(df.columns)


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


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## DDI Pipeline")
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


# =============================================================================
# PROGRESS INDICATOR
# =============================================================================

steps = ["Cargar Archivo", "Sentimiento V2", "Ground Truth", "Topic Detection", "Clustering + S3"]
step_icons = ["📂", "🧠", "✅", "🏷️", "🔗"]
current = st.session_state.current_step

cols = st.columns(len(steps))
for i, (col, step_name, icon) in enumerate(zip(cols, steps, step_icons)):
    if i < current:
        col.markdown(f"<div style='text-align:center;padding:0.5rem;background:#c6f6d5;border-radius:8px;color:#22543d;'><b>{icon} {step_name}</b><br><small>Completado</small></div>", unsafe_allow_html=True)
    elif i == current:
        col.markdown(f"<div style='text-align:center;padding:0.5rem;background:#bee3f8;border-radius:8px;border:2px solid #3182ce;color:#2a4365;'><b>{icon} {step_name}</b><br><small>Actual</small></div>", unsafe_allow_html=True)
    else:
        col.markdown(f"<div style='text-align:center;padding:0.5rem;background:#edf2f7;border-radius:8px;color:#4a5568;'><b>{icon} {step_name}</b><br><small style='color:#a0aec0;'>Pendiente</small></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# STEP 0: UPLOAD FILE
# =============================================================================

if current == 0:
    st.markdown("### 📂 Paso 0: Cargar archivo de menciones")
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
                st.session_state._prev_cols = list(df.columns)

                empty_count = (df['Comentario'].fillna("").astype(str).str.strip() == "").sum()
                render_metric_cards([
                    (f"{len(df)}", "Menciones cargadas"),
                    (f"{len(df.columns)}", "Columnas"),
                    (f"{len(df) - empty_count}", "Con texto"),
                    (f"{empty_count}", "Sin texto"),
                ])

                show_preview(df, uploaded.name, "upload")

                if st.button("Continuar al Paso 1 →", type="primary", use_container_width=True):
                    st.session_state.current_step = 1
                    st.rerun()

        except Exception as e:
            st.error(f"Error al cargar archivo: {e}")


# =============================================================================
# STEP 1: SENTIMIENTO
# =============================================================================

elif current == 1:
    st.markdown("### 🧠 Paso 1: Analisis de Sentimiento con RoBERTuito V2.0")

    df = st.session_state.df_original
    render_metric_cards([
        (f"{len(df)}", "Menciones a procesar"),
        ("~3-5 seg/mención", "Tiempo estimado"),
    ])

    if st.session_state.df_paso_1 is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("Paso 1 ya completado. Puedes revisar los resultados o avanzar.")
        df1 = st.session_state.df_paso_1

        dist = df1['sentiment_v2'].value_counts()
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = px.pie(values=dist.values, names=dist.index, title="Distribución de Sentimiento",
                         color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
            fig.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(df1, x='sentiment_v2_score', nbins=30, title="Distribución de Confianza",
                                color_discrete_sequence=['#3182ce'])
            fig2.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig2, use_container_width=True)

        show_preview(df1, "01_MENCIONES_CON_SENTIMIENTO_V2.csv", "paso1")

        col_dl, col_next = st.columns(2)
        with col_dl:
            st.download_button(
                "Descargar CSV Paso 1",
                df_to_csv_bytes(df1),
                "01_MENCIONES_CON_SENTIMIENTO_V2.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_next:
            if st.button("Continuar al Paso 2 →", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
    else:
        if not USE_LOCAL_MODEL and not HF_TOKEN:
            st.warning("Configura **HF_TOKEN** en el archivo `.env` para cargar el modelo desde Hugging Face.")
        if USE_LOCAL_MODEL and not LOCAL_MODEL_PATH:
            st.warning("Configura **LOCAL_MODEL_PATH** en el archivo `.env`.")

        can_run = (USE_LOCAL_MODEL and LOCAL_MODEL_PATH) or (not USE_LOCAL_MODEL and HF_TOKEN)

        if can_run and st.button("Ejecutar Paso 1", type="primary", use_container_width=True):
            with st.status("Ejecutando Paso 1...", expanded=True) as status:
                st.write("Cargando modelo RoBERTuito V2.0...")
                try:
                    if st.session_state.sentiment_model is None:
                        st.session_state.sentiment_model = load_sentiment_model(USE_LOCAL_MODEL, LOCAL_MODEL_PATH, HF_TOKEN, HF_MODEL_ID)
                    st.write("Modelo cargado. Procesando menciones...")

                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress_1(current_val, total_val):
                        pct = current_val / total_val
                        progress_bar.progress(pct)
                        progress_text.text(f"Procesando mención {current_val}/{total_val}")

                    result = run_paso_1(df, st.session_state.sentiment_model, update_progress_1)
                    st.session_state.df_paso_1 = result
                    status.update(label="Paso 1 completado", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="Error en Paso 1", state="error")
                    st.error(f"Error: {e}")


# =============================================================================
# STEP 2: GROUND TRUTH
# =============================================================================

elif current == 2:
    st.markdown("### ✅ Paso 2: Ground Truth con OpenAI GPT-3.5")

    df1 = st.session_state.df_paso_1
    render_metric_cards([
        (f"{len(df1)}", "Menciones a validar"),
        ("~45-60 min", "Tiempo estimado"),
    ])

    if st.session_state.df_paso_2 is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("Paso 2 ya completado.")
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
                    fig.add_trace(go.Bar(name='RoBERTuito V2', x=labels_all, y=[dist_v2.get(l, 0) for l in labels_all], marker_color='#3182ce'))
                    fig.add_trace(go.Bar(name='Ground Truth', x=labels_all, y=[dist_gt.get(l, 0) for l in labels_all], marker_color='#38a169'))
                    fig.update_layout(barmode='group', title="V2 vs Ground Truth", margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    fig_cm = px.imshow(
                        cm['matrix'], x=cm['labels'], y=cm['labels'],
                        labels=dict(x="Prediccion V2", y="Ground Truth", color="Cantidad"),
                        color_continuous_scale="Blues", title="Matriz de Confusion"
                    )
                    fig_cm.update_layout(margin=dict(t=40, b=20))
                    st.plotly_chart(fig_cm, use_container_width=True)

        show_preview(df2, "02_GROUND_TRUTH_CON_METRICAS.csv", "paso2")

        col_dl, col_next = st.columns(2)
        with col_dl:
            st.download_button(
                "Descargar CSV Paso 2",
                df_to_csv_bytes(df2),
                "02_GROUND_TRUTH_CON_METRICAS.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_next:
            if st.button("Continuar al Paso 3 →", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
    else:
        if not OPENAI_API_KEY:
            st.warning("Configura **OPENAI_API_KEY** en el archivo `.env`.")

        if OPENAI_API_KEY and st.button("Ejecutar Paso 2", type="primary", use_container_width=True):
            with st.status("Ejecutando Paso 2...", expanded=True) as status:
                st.write("Conectando con OpenAI API...")
                progress_bar = st.progress(0)
                progress_text = st.empty()

                def update_progress_2(current_val, total_val):
                    pct = current_val / total_val
                    progress_bar.progress(pct)
                    progress_text.text(f"Validando mención {current_val}/{total_val}")

                try:
                    result_df, result_metrics = run_paso_2(df1, OPENAI_API_KEY, update_progress_2)
                    st.session_state.df_paso_2 = result_df
                    st.session_state.metrics_paso_2 = result_metrics
                    status.update(label="Paso 2 completado", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="Error en Paso 2", state="error")
                    st.error(f"Error: {e}")


# =============================================================================
# STEP 3: TOPICS
# =============================================================================

elif current == 3:
    st.markdown("### 🏷️ Paso 3: Topic Detection Hibrido")

    df2 = st.session_state.df_paso_2
    render_metric_cards([
        (f"{len(df2)}", "Menciones"),
        (f"{len(PREDEFINED_TOPICS)}", "Temas predefinidos"),
        ("BART + GPT-3.5", "Metodo hibrido"),
    ])

    if st.session_state.df_paso_3 is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("Paso 3 ya completado.")
        df3 = st.session_state.df_paso_3
        stats = st.session_state.topic_stats

        col1, col2 = st.columns(2)
        with col1:
            if stats and 'topic_distribution' in stats:
                top_topics = dict(sorted(stats['topic_distribution'].items(), key=lambda x: x[1], reverse=True)[:15])
                fig = px.bar(x=list(top_topics.values()), y=list(top_topics.keys()), orientation='h',
                             title="Top 15 Temas Detectados", color_discrete_sequence=['#2d6a9f'])
                fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=40, b=20, l=150))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if stats and 'method_distribution' in stats:
                md = stats['method_distribution']
                fig2 = px.pie(values=list(md.values()), names=list(md.keys()), title="Metodos Utilizados",
                              hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)

        if stats and stats.get('emergent_themes'):
            st.info(f"**Temas emergentes descubiertos:** {', '.join(stats['emergent_themes'])}")

        show_preview(df3, "03_MENCIONES_CON_TOPICS.csv", "paso3")

        col_dl, col_next = st.columns(2)
        with col_dl:
            st.download_button(
                "Descargar CSV Paso 3",
                df_to_csv_bytes(df3),
                "03_MENCIONES_CON_TOPICS.csv",
                "text/csv",
                use_container_width=True,
            )
        with col_next:
            if st.button("Continuar al Paso 4 →", type="primary", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
    else:
        if not OPENAI_API_KEY:
            st.info("Sin **OPENAI_API_KEY** en `.env`, se usara solo Zero-Shot (BART). Configura la key para habilitar el fallback LLM.")

        if st.button("Ejecutar Paso 3", type="primary", use_container_width=True):
            with st.status("Ejecutando Paso 3...", expanded=True) as status:
                st.write("Cargando clasificador Zero-Shot (facebook/bart-large-mnli)...")
                try:
                    if st.session_state.topic_model is None:
                        st.session_state.topic_model = load_topic_model()
                    st.write("Modelo cargado. Clasificando menciones...")

                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def update_progress_3(current_val, total_val):
                        pct = current_val / total_val
                        progress_bar.progress(pct)
                        progress_text.text(f"Clasificando mención {current_val}/{total_val}")

                    result_df, result_stats = run_paso_3(
                        df2, st.session_state.topic_model,
                        openai_api_key=OPENAI_API_KEY if OPENAI_API_KEY else None,
                        progress_callback=update_progress_3
                    )
                    st.session_state.df_paso_3 = result_df
                    st.session_state.topic_stats = result_stats
                    status.update(label="Paso 3 completado", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="Error en Paso 3", state="error")
                    st.error(f"Error: {e}")


# =============================================================================
# STEP 4: CLUSTERING + S3
# =============================================================================

elif current == 4:
    st.markdown("### 🔗 Paso 4: Conversation Clustering + Subida a S3")

    df3 = st.session_state.df_paso_3
    render_metric_cards([
        (f"{len(df3)}", "Menciones"),
        ("Local", "Procesamiento"),
        ("$0", "Costo"),
    ])

    if st.session_state.df_paso_4 is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("Paso 4 completado. Pipeline finalizado.")
        df4 = st.session_state.df_paso_4
        esc = st.session_state.escalation_summary
        crisis = st.session_state.crisis_conversations

        render_metric_cards([
            (f"{esc['normal']}", "Normal"),
            (f"{esc['attention']}", "Atencion"),
            (f"{esc['escalating']}", "Escalando"),
            (f"{esc['crisis']}", "Crisis"),
        ])

        col1, col2 = st.columns(2)
        with col1:
            esc_data = {k: v for k, v in esc.items()}
            fig = px.pie(values=list(esc_data.values()), names=list(esc_data.keys()),
                         title="Niveles de Escalation",
                         color=list(esc_data.keys()),
                         color_discrete_map={"normal": "#48bb78", "attention": "#ecc94b",
                                             "escalating": "#ed8936", "crisis": "#fc8181"},
                         hole=0.4)
            fig.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
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

        show_preview(df4, "04_CONVERSATION_THREADS_CLUSTERED.csv", "paso4")

        st.markdown("---")
        st.download_button(
            "Descargar CSV Final",
            df_to_csv_bytes(df4),
            "04_CONVERSATION_THREADS_CLUSTERED.csv",
            "text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; padding:1.5rem; background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%); border-radius:12px;">
            <h2 style="color:#22543d; margin:0;">Pipeline Completado</h2>
            <p style="color:#276749;">Todos los pasos han sido ejecutados exitosamente.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        if st.button("Ejecutar Paso 4", type="primary", use_container_width=True):
            with st.status("Ejecutando Paso 4...", expanded=True) as status:
                st.write("Analizando threads y detectando crisis...")

                progress_bar = st.progress(0)
                progress_text = st.empty()

                def update_progress_4(current_val, total_val):
                    pct = current_val / total_val
                    progress_bar.progress(pct)
                    progress_text.text(f"Analizando thread {current_val}/{total_val}")

                try:
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

                    status.update(label="Paso 4 completado", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="Error en Paso 4", state="error")
                    st.error(f"Error: {e}")
