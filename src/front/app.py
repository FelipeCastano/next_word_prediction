import streamlit as st
import requests

st.title("Predictor de la siguiente palabra")

# Estado inicial
if "lang" not in st.session_state:
    st.session_state.lang = 2  # Español por defecto
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = []  # Lista de sugerencias

# Selección de idioma usando radio en lugar de checkboxes
lang_option = st.radio("Selecciona el idioma:", options=["Español", "Inglés"], index=0 if st.session_state.lang == 2 else 1)
st.session_state.lang = 2 if lang_option == "Español" else 1

# Input del usuario
input_text = st.text_input("Escribe algo:", value=st.session_state.input_text, key="user_input")
st.session_state.input_text = input_text

# Función para construir el payload
def construir_payload(input_text):
    tokens = input_text.strip().split()
    if input_text.endswith(" "):
        starts_with = ""
        text_tokens = tokens[-3:]
    else:
        starts_with = tokens[-1] if tokens else ""
        text_tokens = tokens[-4:-1] if len(tokens) >= 2 else []
    text = " ".join(text_tokens)
    return text, starts_with

# Función para obtener sugerencias de la API
def generar_prediccion():
    if st.session_state.lang and input_text.strip():
        text, starts_with = construir_payload(input_text)
        payload = {
            "text": text,
            "lang": st.session_state.lang,
            "starts_with": starts_with
        }

        try:
            response = requests.post("http://api:8000/get_next_word", json=payload)
            response.raise_for_status()
            result = response.json()

            # Procesar las sugerencias
            sugerencias = []
            for key in sorted(result.keys()):
                sugerencia = result[key]
                sugerencias.append({
                    "word": sugerencia["word"],
                    "prob": sugerencia["prob"]
                })

            st.session_state.prediction = sugerencias

        except requests.RequestException as e:
            st.error(f"Error al conectar con la API: {e}")
            st.session_state.prediction = []
    else:
        st.session_state.prediction = []

# Detectar cambios en el input
if input_text != st.session_state.last_input_text:
    st.session_state.last_input_text = input_text
    generar_prediccion()

# Mostrar botones de sugerencias sin probabilidad
if st.session_state.prediction:
    st.markdown("### Palabras sugeridas:")
    cols = st.columns(len(st.session_state.prediction))
    for i, sugerencia in enumerate(st.session_state.prediction):
        word = sugerencia["word"]
        with cols[i]:
            if st.button(f"{word}", key=f"sugerencia_{i}"):
                if input_text.endswith(" "):
                    st.session_state.input_text = input_text + word + " "
                else:
                    tokens = input_text.strip().split()
                    if tokens:
                        tokens[-1] = word
                    else:
                        tokens = [word]
                    st.session_state.input_text = " ".join(tokens) + " "

                st.session_state.last_input_text = st.session_state.input_text
                st.rerun()
