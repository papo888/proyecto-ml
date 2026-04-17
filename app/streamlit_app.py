import base64
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Genre Predictor", layout="wide")


# -------------------------------------------------
# Función temporal
# -------------------------------------------------
def predecir_genero_dummy(df: pd.DataFrame):
    return "pop_rock", {
        "pop_rock": 0.62,
        "urban": 0.12,
        "electronic": 0.09,
        "fine_arts": 0.05,
        "media_and_screen": 0.04,
        "roots_and_soul": 0.05,
        "specialty_content": 0.03,
    }


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def cargar_css(css_path: str):
    css_file = Path(css_path)
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)


def get_base64_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


# -------------------------------------------------
# Cargar estilos e imagen
# -------------------------------------------------
cargar_css("app/assets/styles.css")
cover_base64 = get_base64_image("app/assets/cover_demo.jpg")


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="topbar">SAVIOM · MUSIC STREAMING</div>', unsafe_allow_html=True)

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

    st.markdown("""
        <div class="track-box">
            <div class="track-title">Unknown Track</div>
            <div class="subtle-text">Audio-based genre classification demo</div>
            <div class="control-row">
                <div class="control-pill">Play</div>
                <div class="control-pill">Predict</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">GENRE<br>PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle-text">Descubre el género de una canción a partir de sus características de audio.</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    danceability = st.slider("Danceability", 0.0, 1.0, 0.50)
    energy = st.slider("Energy", 0.0, 1.0, 0.50)
    valence = st.slider("Valence", 0.0, 1.0, 0.50)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.30)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.10)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.20)
    tempo = st.number_input("Tempo", min_value=0.0, value=120.0)
    loudness = st.number_input("Loudness", value=-8.0)

    if st.button("Predict Genre"):
        entrada = pd.DataFrame([{
            "danceability": danceability,
            "energy": energy,
            "valence": valence,
            "acousticness": acousticness,
            "speechiness": speechiness,
            "instrumentalness": instrumentalness,
            "tempo": tempo,
            "loudness": loudness,
        }])

        genero, probabilidades = predecir_genero_dummy(entrada)

        st.markdown(f"""
            <div class="predict-card">
                <div class="predict-label">Predicted Genre</div>
                <div class="predict-genre">{genero.replace("_", " ").title()}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="mini-label">Top Results</div>', unsafe_allow_html=True)

        top_probs = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
        for clase, prob in top_probs:
            st.markdown(f"""
                <div class="playlist-card">
                    <div class="playlist-row">
                        <div class="playlist-name">{clase.replace("_", " ").title()}</div>
                        <div class="playlist-prob">{prob:.0%}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)