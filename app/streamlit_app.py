import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Genre Predictor", layout="wide")


def cargar_css(css_path: str):
    css_file = Path(css_path)
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)


def get_base64_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


def obtener_cover_por_genero(genero):
    covers = {
        "mainstream": "app/assets/mainstream.jpg",
        "urban": "app/assets/urban.jpg",
        "acoustic": "app/assets/acoustic.jpg",
        "other": "app/assets/other.jpeg",
    }
    return covers.get(genero, "app/assets/cover_demo.jpg")


def nombre_genero_bonito(genero):
    nombres = {
        "mainstream": "Mainstream",
        "urban": "Urban",
        "acoustic": "Acoustic",
        "other": "Other",
    }
    return nombres.get(genero, str(genero).replace("_", " ").title())


@st.cache_resource
def cargar_modelo():
    return joblib.load("models/modelo_final_lightgbm_4clases.joblib")


def obtener_info_preprocessor(modelo):
    preprocessor = modelo.named_steps["preprocessor"]

    numeric_cols = []
    categorical_cols = []
    categorias_por_columna = {}

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            numeric_cols = list(cols)
        elif name == "cat":
            categorical_cols = list(cols)
            encoder = transformer
            for col, cats in zip(categorical_cols, encoder.categories_):
                categorias_por_columna[col] = list(cats)

    return numeric_cols, categorical_cols, categorias_por_columna


def construir_entrada(valores_usuario: dict, numeric_cols: list, categorical_cols: list):
    fila = {}

    for col in numeric_cols:
        fila[col] = valores_usuario[col]

    for col in categorical_cols:
        fila[col] = valores_usuario[col]

    return pd.DataFrame([fila])


if "prediccion_genero" not in st.session_state:
    st.session_state.prediccion_genero = None

if "prediccion_probs" not in st.session_state:
    st.session_state.prediccion_probs = None


cargar_css("app/assets/styles.css")

genero_actual = st.session_state.prediccion_genero
ruta_cover = obtener_cover_por_genero(genero_actual)
cover_base64 = get_base64_image(ruta_cover)

try:
    modelo = cargar_modelo()
    numeric_cols, categorical_cols, categorias_por_columna = obtener_info_preprocessor(modelo)
    modelo_ok = True
    error_modelo = None
except Exception as e:
    modelo_ok = False
    error_modelo = str(e)
    numeric_cols, categorical_cols, categorias_por_columna = [], [], {}


st.markdown('<div class="topbar">MUSIC STREAMING PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="main-card">', unsafe_allow_html=True)

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown('<div class="mini-label">Now Playing</div>', unsafe_allow_html=True)

    vinyl_html = f"""
    <div class="vinyl-wrap">
        <div class="vinyl">
            {"<img class='cover-art' src='data:image/jpeg;base64," + cover_base64 + "' />" if cover_base64 else ""}
            <div class="vinyl-hole"></div>
        </div>
    </div>
    """
    st.markdown(vinyl_html, unsafe_allow_html=True)

    titulo_track = "Unknown Track"
    subtitulo_track = "Audio-based macro-genre classification demo"

    if st.session_state.prediccion_genero:
        titulo_track = nombre_genero_bonito(st.session_state.prediccion_genero)
        subtitulo_track = "Predicted macro-genre cover"

    st.markdown(f"""
        <div class="track-box">
            <div class="track-title">{titulo_track}</div>
            <div class="subtle-text">{subtitulo_track}</div>
            <div class="control-row">
                <div class="control-pill">Play</div>
                <div class="control-pill">Predict</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if modelo_ok:
        st.markdown(
            f"<div class='subtle-text' style='margin-top: 1rem;'><b>{len(numeric_cols) + len(categorical_cols)}</b> features detectadas en el modelo.</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">GENRE<br>PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle-text">Clasifica una canción en uno de los 4 macro-géneros a partir de sus características de audio.</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.prediccion_genero:
        pred = st.session_state.prediccion_genero
        probabilidades = st.session_state.prediccion_probs

        st.markdown(f"""
            <div class="predict-card">
                <div class="predict-label">Predicted Genre</div>
                <div class="predict-genre">{nombre_genero_bonito(pred)}</div>
            </div>
        """, unsafe_allow_html=True)

        if probabilidades:
            st.markdown('<div class="mini-label">Top Results</div>', unsafe_allow_html=True)
            top_probs = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)[:4]

            for clase, prob in top_probs:
                st.markdown(f"""
                    <div class="playlist-card">
                        <div class="playlist-row">
                            <div class="playlist-name">{nombre_genero_bonito(clase)}</div>
                            <div class="playlist-prob">{prob:.0%}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    popularity = st.number_input("Popularity", min_value=0, max_value=100, value=50)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.30)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.50)
    duration_ms = st.number_input("Duration (ms)", min_value=0, value=210000)
    energy = st.slider("Energy", 0.0, 1.0, 0.50)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.20)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.20)
    loudness = st.number_input("Loudness", value=-8.0)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.10)
    tempo = st.number_input("Tempo", min_value=0.0, value=120.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.50)

    if modelo_ok:
        key_options = categorias_por_columna.get("key", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        mode_options = categorias_por_columna.get("mode", [0, 1])
        ts_options = categorias_por_columna.get("time_signature", [3, 4, 5])

        key_val = st.selectbox("Key", key_options, index=0)
        mode_val = st.selectbox("Mode", mode_options, index=0)
        time_signature_val = st.selectbox("Time Signature", ts_options, index=0)
    else:
        key_val = 0
        mode_val = 0
        time_signature_val = 4

    if st.button("Predict Genre"):
        if not modelo_ok:
            st.error(f"No se pudo cargar el modelo: {error_modelo}")
        else:
            valores_usuario = {
                "popularity": popularity,
                "acousticness": acousticness,
                "danceability": danceability,
                "duration_ms": duration_ms,
                "energy": energy,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "loudness": loudness,
                "speechiness": speechiness,
                "tempo": tempo,
                "valence": valence,
                "key": key_val,
                "mode": mode_val,
                "time_signature": time_signature_val,
            }

            entrada = construir_entrada(valores_usuario, numeric_cols, categorical_cols)
            pred = modelo.predict(entrada)[0]

            probabilidades = None
            if hasattr(modelo, "predict_proba"):
                proba = modelo.predict_proba(entrada)[0]
                clases = modelo.classes_
                probabilidades = dict(zip(clases, proba))

            st.session_state.prediccion_genero = pred
            st.session_state.prediccion_probs = probabilidades

            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)