# -*- coding: utf-8 -*-
"""
Pipeline DDI - Lógica de los 4 pasos
Basado en: Pipeline (2).ipynb
Limpiado de: google.colab, !pip install, files.upload/download
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

LABEL_MAPPING = {
    "positivo": "positive",
    "negativo": "negative",
    "neutro": "neutral"
}

SENTIMENT_MAPPING = {
    "positivo": "positive",
    "negativo": "negative",
    "neutro": "neutral",
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral"
}

PREDEFINED_TOPICS = [
    "Dinámica", "Almuerzo", "Loncherita Campero", "Apertura de restaurante",
    "Producto - Sabor", "Promoción con Pepsi", "Desayuno", "Pollo frito",
    "Ventana 1", "Ventana 1 - Gira del Pollito", "Entretenimiento",
    "Canal digital - App Campero", "Tienda de Pollito", "Alitas o Alas",
    "Producto - Caro", "A todo sabor", "Producto - Tamaño",
    "Calidad de producto", "Reapertura de restaurante", "Servicio - Domicilio",
    "Canal digital - Whatsapp", "Servicio al cliente", "Combos o Menús",
    "Lanzamiento Producto Nuevo", "Institucional", "Atención en mesa",
    "Comunicado", "Incremento de precios", "Banquetes",
    "Canal digital - Página Web", "Producto no disponible",
    "Contenido en colaboración", "Ganadores de promociones",
    "Canal de Llamada", "Publicidad engañosa", "Estafa o fraude",
    "Producto - Cocción", "Servicio - Foodcourt", "Reclutamiento",
    "Pollo en piezas", "Promoción - No disponible", "Detractor",
    "Producto - Higiene"
]

ENGAGEMENT_THRESHOLD_HIGH = 1000
NEGATIVE_RATIO_THRESHOLD = 0.6
NEGATIVE_RATIO_CRITICAL = 0.75


# =============================================================================
# PASO 1: SENTIMIENTO
# =============================================================================

def load_sentiment_model(use_local, local_path, hf_token, model_id):
    from transformers import pipeline as hf_pipeline

    if use_local and local_path:
        classifier = hf_pipeline(
            'text-classification',
            model=local_path,
            device=-1
        )
    else:
        classifier = hf_pipeline(
            'text-classification',
            model=model_id,
            token=hf_token,
            device=-1
        )
    return classifier


def run_paso_1(df, classifier, progress_callback=None):
    from pysentimiento.preprocessing import preprocess_tweet

    df = df.copy()
    df['Comentario'] = df['Comentario'].fillna("").astype(str)

    sentiments = []
    scores = []
    total = len(df)

    for idx, text in enumerate(df['Comentario'].values, 1):
        try:
            text_prep = preprocess_tweet(text)
            result = classifier(text_prep, truncation=True, max_length=128)
            sentiment = result[0]['label']
            sentiment = LABEL_MAPPING.get(sentiment, "neutral")
            score = round(result[0]['score'], 4)
            sentiments.append(sentiment)
            scores.append(score)
        except Exception as e:
            logger.warning(f"Error en mención {idx}: {e}")
            sentiments.append("error")
            scores.append(0.0)

        if progress_callback:
            progress_callback(idx, total)

    df['sentiment_v2'] = sentiments
    df['sentiment_v2_score'] = scores
    df['processed_timestamp'] = datetime.now().isoformat()

    return df


# =============================================================================
# PASO 2: GROUND TRUTH
# =============================================================================

GROUND_TRUTH_SYSTEM_PROMPT = """Eres un experto en análisis de sentimiento para redes sociales en español.
Tu tarea es clasificar el sentimiento de menciones sobre bancos, política, cementeras, riesgo reputacional y restaurantes en Guatemala.

Contexto:
- Dialecto: Español guatemalteco (incluye jerga local)
- Industrias: Banca (Bantrab, etc.), Cementeras (Cemento Progresos), Política (Ley de Competencia) y restaurantes (Pollo Campero, McDonald's)
- Plataformas: Twitter, Facebook, Instagram, TikTok

Instrucciones:
1. Lee el TEXTO de la mención
2. Considera el CONTEXTO: profesión del autor, query/tema
3. Detecta SARCASMO: "Qué magnífico servicio" con 3 horas de espera = NEGATIVO
4. Entiende JERGA: "chilero" = positivo, "shumo" = negativo, "talega" = dinero
5. Clasifica como: positive, negative, o neutral

Responde SOLO con una palabra: positive, negative, o neutral."""


def run_paso_2(df, openai_api_key, progress_callback=None):
    df = df.copy()
    client = OpenAI(api_key=openai_api_key)

    ground_truths = []
    ground_truth_scores = []
    request_count = 0
    total = len(df)

    for idx, row in df.iterrows():
        try:
            text = row['Comentario']
            context = f"Query: {row.get('query_name', 'N/A')}. Profesión autor: {row.get('author_professions', 'N/A')}"
            user_message = f"Clasifica este texto:\n\n{text}\n\n{context}"

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": GROUND_TRUTH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=10
            )

            prediction = response.choices[0].message.content.strip().lower()
            valid_sentiments = ['positive', 'negative', 'neutral']
            if prediction not in valid_sentiments:
                if 'negat' in prediction:
                    prediction = 'negative'
                elif 'posit' in prediction:
                    prediction = 'positive'
                else:
                    prediction = 'neutral'

            confidence = 0.8 if len(response.choices[0].message.content.split()) == 1 else 0.7
            ground_truths.append(prediction)
            ground_truth_scores.append(round(confidence, 2))
            request_count += 1

            if request_count % 20 == 0:
                time.sleep(2)

        except Exception as e:
            logger.warning(f"Error en mención {idx}: {e}")
            ground_truths.append("neutral")
            ground_truth_scores.append(0.0)

        if progress_callback:
            progress_callback(idx + 1, total)

    df['sentiment_ground_truth'] = ground_truths
    df['sentiment_ground_truth_score'] = ground_truth_scores
    df['validated_timestamp'] = datetime.now().isoformat()
    df['sentiment_v2_standard'] = df['sentiment_v2'].map(SENTIMENT_MAPPING).fillna(df['sentiment_v2'])
    df['sentiment_v2_standard'] = df['sentiment_v2_standard'].fillna('neutral')

    # Calcular métricas
    y_v2 = df['sentiment_v2_standard'].values
    y_truth = df['sentiment_ground_truth'].values
    valid_indices = ~(pd.isna(y_v2) | pd.isna(y_truth))
    y_v2_valid = y_v2[valid_indices]
    y_truth_valid = y_truth[valid_indices]

    metrics = {}
    if len(y_v2_valid) > 0:
        accuracy = accuracy_score(y_truth_valid, y_v2_valid)
        precision = precision_score(y_truth_valid, y_v2_valid, average='weighted', zero_division=0)
        recall = recall_score(y_truth_valid, y_v2_valid, average='weighted', zero_division=0)
        f1 = f1_score(y_truth_valid, y_v2_valid, average='weighted', zero_division=0)
        all_labels = ['negative', 'neutral', 'positive']
        cm = confusion_matrix(y_truth_valid, y_v2_valid, labels=all_labels)
        lift_pct = ((1 - accuracy) / accuracy) * 100 if accuracy > 0 else 0

        metrics = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_mentions": len(df),
            "valid_mentions": int(len(y_v2_valid)),
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "lift_percentage": round(lift_pct, 2)
            },
            "confusion_matrix": {"labels": all_labels, "matrix": cm.tolist()},
            "distribution": {
                "v2_0": df.loc[valid_indices, 'sentiment_v2_standard'].value_counts().to_dict(),
                "ground_truth": df.loc[valid_indices, 'sentiment_ground_truth'].value_counts().to_dict()
            },
            "agreement": {
                "perfect_match": int((y_v2_valid == y_truth_valid).sum()),
                "perfect_match_pct": round((y_v2_valid == y_truth_valid).sum() / len(y_v2_valid) * 100, 2),
                "disagreement": int((y_v2_valid != y_truth_valid).sum()),
                "disagreement_pct": round((y_v2_valid != y_truth_valid).sum() / len(y_v2_valid) * 100, 2)
            },
            "cost_estimate": {
                "total_requests": request_count,
                "cost_usd": round(request_count * 0.0005, 2),
                "note": "Basado en ~0.0005 USD por request"
            }
        }

    return df, metrics


# =============================================================================
# PASO 3: TOPICS
# =============================================================================

TOPICS_SYSTEM_PROMPT = """Eres un experto en análisis de menciones de redes sociales para bancos, idustrias como Cementeras y restaurantes de comida rapida en Guatemala.
Tu tarea es IDENTIFICAR EL TEMA PRINCIPAL de la mención.

TEMAS COMUNES:
- Dinámica: Incluye todos los posts que tengan CTA para campañas o acciones especiales.
-  Almuerzo: Refleja los productos que pertenecen a ofertas o promociones de almuerzo.
-  Loncherita Campero: Incluye lanzamientos, productos y actividades relacionadas con la marca Campero.
-  Apertura de restaurante: Refleja las inauguraciones de nuevos puntos de venta o restaurantes.
-  Producto - Sabor: Todo lo referente a críticas o comentarios sobre el sabor específico del producto.
-  Promoción con Pepsi: Incluye todo lo relacionado con la Gira Reffizzcante y promociones asociadas con Pepsi.
-  Desayuno: Incluye todo lo referente a productos hasta ahora relacionados con el desayuno.
-  Pollo frito: Todo lo que especifique "Pollo frito" en los mensajes.
-  Ventana 1: Incluye todo lo relacionado con las ofertas de enero o primeras promociones.
-  Ventana 1 - Gira del Pollito: Todo lo relacionado con la gira del pollito y sus promociones.
-  Entretenimiento: Contenido basado en tendencias o contenido que promueva entretenimiento.
-  Canal digital - App Campero: Quejas o sugerencias relacionadas con la aplicación móvil de Campero.
-  Tienda de Pollito: Todo lo relacionado con aperturas de tiendas y productos relacionados con pollito.
-  Alitas o Alas: Todo lo relacionado con productos de alitas o alas.
-  Producto - Caro: Percepción negativa sobre el precio vs. el producto recibido.
-  A todo sabor: Contenido institucional que incluye la frase "A todo sabor".
-  Producto - Tamaño: Percepción negativa sobre el tamaño de los productos.
-  Calidad de producto: Percepción negativa sobre la calidad de los productos.
-  Reapertura de restaurante: Todo lo relacionado con remodelaciones y reaperturas de puntos de venta.
-  Servicio - Domicilio: Quejas o sugerencias sobre el servicio de entrega a domicilio.
-  Canal digital - Whatsapp: Quejas o sugerencias específicas sobre el uso del canal de WhatsApp.
-  Servicio al cliente: Quejas o sugerencias generales sobre el servicio al cliente.
-  Combos o Menús: Todo lo referente a menús, combos y ofertas combinadas.
-  Lanzamiento Producto Nuevo: Todo lo relacionado con el lanzamiento de nuevos productos.
-  Institucional: Campañas generales de la marca o contenido institucional.
-  Atención en mesa: Quejas o sugerencias sobre la atención en el restaurante.
-  Comunicado: Comunicados oficiales de la marca.
-  Incremento de precios: Comentarios sobre la subida de precios de productos.
-  Banquetes: Información relacionada con los banquetes anunciados.
-  Canal digital - Página Web: Quejas o sugerencias relacionadas con la página web de la marca.
-  Producto no disponible: Quejas por falta de disponibilidad de productos.
-  Contenido en colaboración: Contenido trabajado con influencers y otros colaboradores.
  -Ganadores de promociones: Anuncios de los ganadores de promociones y concursos.
-  Canal de Llamada: Quejas o sugerencias específicas sobre el canal telefónico de atención al cliente.
-  Publicidad engañosa: Acusaciones sobre publicidad que no refleja la realidad del producto o servicio.
-  Estafa o fraude: Comentarios que mencionan términos como "fraude" o "estafa".
-  Producto - Cocción: Quejas sobre el estado de cocción del producto (por ejemplo, crudo o sobrecocido).
-  Servicio - Foodcourt: Comentarios relacionados con el servicio en el área de foodcourt.
-  Reclutamiento: Quejas o sugerencias relacionadas con el proceso de reclutamiento o aplicación a puestos laborales.
-  Pollo en piezas: Menciones sobre piezas de pollo como cuadril, pierna, etc.
-  Promoción - No disponible: Quejas sobre la falta de disponibilidad de una promoción.
-  Detractor: Personas que atacan a la marca sin una queja específica.
-  Producto - Higiene: Quejas sobre salubridad, limpieza en procesos de producción o distribución.
- Tema emergente (Cuakquier otro tema que no esté en la lista)


INSTRUCCIONES:
1. Lee el texto cuidadosamente
2. Identifica el tema PRINCIPAL (uno solo)
3. Si no coincide con lista común, describe el tema emergente con 2-3 palabras
4. Responde SOLO con el nombre del tema (máximo 4 palabras)"""


def load_topic_model():
    from transformers import pipeline as hf_pipeline
    zero_shot = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    return zero_shot


def classify_zero_shot(text, zero_shot_classifier, candidate_topics, threshold=0.80):
    try:
        if pd.isna(text) or not isinstance(text, str):
            return "Otros", 0.0, "invalid_text"
        text = str(text).strip()
        if len(text) < 3:
            return "Otros", 0.0, "empty_text"

        result = zero_shot_classifier(text, candidate_topics, multi_class=False, truncation=True)
        return result['labels'][0], round(result['scores'][0], 4), "zero_shot"
    except Exception as e:
        logger.warning(f"Error en Zero Shot: {e}")
        return "Otros", 0.0, "error"


def classify_llm_fallback(text, context_query, context_sentiment, client):
    if not client:
        return "Otros", 0.7, "fallback_no_api"
    try:
        user_message = f"Identifica el tema principal:\n\nTEXTO: {text}\n\nCONTEXTO:\n- Query/Tema: {context_query}\n- Sentimiento: {context_sentiment}\n\nTema principal (máximo 4 palabras):"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": TOPICS_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=10
        )
        topic = response.choices[0].message.content.strip()
        if len(topic) > 50:
            topic = topic[:50]
        return topic, 0.75, "llm_fallback"
    except Exception as e:
        logger.warning(f"Error en LLM Fallback: {e}")
        return "Otros", 0.0, "fallback_error"


def run_paso_3(df, zero_shot_classifier, openai_api_key=None, confidence_threshold=0.80, progress_callback=None):
    df = df.copy()
    df['Comentario'] = df['Comentario'].fillna("").astype(str)

    client = None
    if openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
        except Exception:
            client = None

    topics = []
    confidences = []
    methods = []
    emergent_topics = set()
    total = len(df)

    for idx, row in df.iterrows():
        text = row['Comentario']
        query_name = row.get('query_name', 'Desconocido')
        sentiment = row.get('sentiment_ground_truth', 'neutro')

        topic, confidence, method = classify_zero_shot(text, zero_shot_classifier, PREDEFINED_TOPICS, confidence_threshold)

        if confidence < confidence_threshold and client:
            topic, confidence, method = classify_llm_fallback(text, query_name, sentiment, client)
            if topic not in PREDEFINED_TOPICS:
                emergent_topics.add(topic)

        topics.append(topic)
        confidences.append(confidence)
        methods.append(method)

        if progress_callback:
            progress_callback(idx + 1, total)

    df['tema_detectado'] = topics
    df['tema_confianza'] = confidences
    df['tema_metodo'] = methods
    df['tema_timestamp'] = datetime.now().isoformat()

    topic_stats = {
        "topic_distribution": df['tema_detectado'].value_counts().to_dict(),
        "method_distribution": df['tema_metodo'].value_counts().to_dict(),
        "emergent_themes": list(emergent_topics),
        "confidence_mean": round(df['tema_confianza'].mean(), 3),
    }

    return df, topic_stats


# =============================================================================
# PASO 4: CLUSTERING
# =============================================================================

def calculate_sentiment_score(sentiment):
    if sentiment == 'positive':
        return 1.0
    elif sentiment == 'negative':
        return -1.0
    return 0.0


def analyze_thread_sentiment(thread_df):
    sentiments = [calculate_sentiment_score(s) for s in thread_df['sentiment_ground_truth']]
    if len(sentiments) == 0:
        return {'avg_sentiment': 0.0, 'negative_count': 0, 'positive_count': 0,
                'neutral_count': 0, 'negative_ratio': 0.0, 'progression': 'flat'}

    negative_count = (thread_df['sentiment_ground_truth'] == 'negative').sum()
    positive_count = (thread_df['sentiment_ground_truth'] == 'positive').sum()
    neutral_count = (thread_df['sentiment_ground_truth'] == 'neutral').sum()
    negative_ratio = negative_count / len(thread_df)

    progression = 'single_mention'
    if len(sentiments) > 1:
        first_half = np.mean(sentiments[:len(sentiments)//2])
        second_half = np.mean(sentiments[len(sentiments)//2:])
        if second_half > first_half:
            progression = 'improving'
        elif second_half < first_half:
            progression = 'worsening'
        else:
            progression = 'stable'

    return {
        'avg_sentiment': round(np.mean(sentiments), 3),
        'negative_count': int(negative_count),
        'positive_count': int(positive_count),
        'neutral_count': int(neutral_count),
        'negative_ratio': round(negative_ratio, 3),
        'progression': progression
    }


def detect_escalation_level(thread_df, analysis):
    negative_ratio = analysis['negative_ratio']
    progression = analysis['progression']

    engagement = 0
    for col in ['reach', 'impressions', 'impact']:
        if col in thread_df.columns:
            engagement += thread_df[col].fillna(0).sum()

    if negative_ratio >= NEGATIVE_RATIO_CRITICAL and engagement >= ENGAGEMENT_THRESHOLD_HIGH:
        return "🔴 CRISIS", 90
    elif negative_ratio >= NEGATIVE_RATIO_CRITICAL:
        return "🔴 CRISIS", 85
    elif negative_ratio >= NEGATIVE_RATIO_THRESHOLD and progression == 'worsening':
        return "🟠 ESCALATING", 70
    elif negative_ratio >= NEGATIVE_RATIO_THRESHOLD:
        return "🟠 ESCALATING", 60
    elif 0.3 <= negative_ratio < NEGATIVE_RATIO_THRESHOLD:
        return "🟡 ATTENTION", 40
    return "🟢 NORMAL", 15


def run_paso_4(df, progress_callback=None):
    df = df.copy()

    if 'threadId' in df.columns:
        group_col = 'threadId'
    else:
        if 'author_name' in df.columns:
            df['pseudo_threadId'] = df['tema_detectado'] + '_' + df['author_name'].fillna('unknown')
        else:
            df['pseudo_threadId'] = df['tema_detectado']
        group_col = 'pseudo_threadId'

    thread_groups = df.groupby(group_col)
    thread_analyses = []
    crisis_conversations = []
    total_threads = len(thread_groups)

    for i, (thread_id, thread_df) in enumerate(thread_groups):
        sentiment_analysis = analyze_thread_sentiment(thread_df)
        escalation_level, risk_score = detect_escalation_level(thread_df, sentiment_analysis)

        engagement_total = 0
        for col in ['reach', 'impressions', 'impact']:
            if col in thread_df.columns:
                engagement_total += int(thread_df[col].fillna(0).sum())

        thread_analysis = {
            'thread_id': str(thread_id),
            'mention_count': len(thread_df),
            'primary_topic': thread_df['tema_detectado'].mode()[0] if len(thread_df) > 0 else 'Unknown',
            'authors': thread_df['author_name'].nunique() if 'author_name' in thread_df.columns else 0,
            'escalation_level': escalation_level,
            'crisis_risk_score': int(risk_score),
            'sentiment_analysis': sentiment_analysis,
            'engagement_total': engagement_total
        }
        thread_analyses.append(thread_analysis)
        if escalation_level.startswith("🔴"):
            crisis_conversations.append(thread_analysis)

        if progress_callback:
            progress_callback(i + 1, total_threads)

    thread_escalation_map = {a['thread_id']: a['escalation_level'] for a in thread_analyses}
    thread_risk_map = {a['thread_id']: a['crisis_risk_score'] for a in thread_analyses}

    df['escalation_level'] = df[group_col].astype(str).map(thread_escalation_map)
    df['crisis_risk_score'] = df[group_col].astype(str).map(thread_risk_map)
    df['clustering_timestamp'] = datetime.now().isoformat()

    escalation_summary = {
        "normal": sum(1 for t in thread_analyses if t['escalation_level'].startswith('🟢')),
        "attention": sum(1 for t in thread_analyses if t['escalation_level'].startswith('🟡')),
        "escalating": sum(1 for t in thread_analyses if t['escalation_level'].startswith('🟠')),
        "crisis": sum(1 for t in thread_analyses if t['escalation_level'].startswith('🔴')),
    }

    return df, thread_analyses, escalation_summary, crisis_conversations
