import streamlit as st
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

st.title("Next Word Predictor")

# Initial state
if "lang" not in st.session_state:
    st.session_state.lang = 2  # Default to Spanish
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = []  # List of suggestions

# Language selection using radio instead of checkboxes
lang_option = st.radio("Select language:", options=["Spanish", "English"], index=0 if st.session_state.lang == 2 else 1)
st.session_state.lang = 2 if lang_option == "Spanish" else 1

# User input
input_text = st.text_input("Type something:", value=st.session_state.input_text, key="user_input")
st.session_state.input_text = input_text

# Function to build the payload
def build_payload(input_text):
    tokens = input_text.strip().split()
    if input_text.endswith(" "):
        starts_with = ""
        text_tokens = tokens[-3:]
    else:
        starts_with = tokens[-1] if tokens else ""
        text_tokens = tokens[-4:-1] if len(tokens) >= 2 else []
    text = " ".join(text_tokens)
    return text, starts_with

# Function to get suggestions from the API
def generate_prediction():
    if st.session_state.lang and input_text.strip():
        text, starts_with = build_payload(input_text)
        payload = {
            "text": text,
            "lang": st.session_state.lang,
            "starts_with": starts_with
        }

        # Log the request payload
        logging.info(f"Sending payload to API: {payload}")

        try:
            response = requests.post("http://api:8000/get_next_word", json=payload)
            response.raise_for_status()
            result = response.json()

            # Log the API response
            logging.info(f"Received response from API: {result}")

            # Process suggestions
            suggestions = []
            for key in sorted(result.keys()):
                suggestion = result[key]
                suggestions.append({
                    "word": suggestion["word"],
                    "prob": suggestion["prob"]
                })

            st.session_state.prediction = suggestions

        except requests.RequestException as e:
            st.error(f"Error connecting to the API: {e}")
            logging.error(f"API request failed: {e}")
            st.session_state.prediction = []
    else:
        st.session_state.prediction = []

# Button to trigger suggestion
if st.button("Make suggestion"):
    generate_prediction()

# Display suggestion buttons without probabilities
if st.session_state.prediction:
    st.markdown("### Suggested words:")
    cols = st.columns(len(st.session_state.prediction))
    for i, suggestion in enumerate(st.session_state.prediction):
        word = suggestion["word"]
        with cols[i]:
            if st.button(f"{word}", key=f"suggestion_{i}"):
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
