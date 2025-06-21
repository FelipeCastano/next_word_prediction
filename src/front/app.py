import streamlit as st
import requests
from streamlit.components.v1 import html

# Tag to color
TAG_COLORS = {
    "NN": "#1f77b4",
    "NNP": "#ff7f0e",
    "VB": "#2ca02c",
    "JJ": "#d62728",
    "RB": "#9467bd",
    "PRP": "#8c564b",
    "DT": "#e377c2",
    "IN": "#7f7f7f",
    "CC": "#bcbd22",
    "CD": "#17becf",
    "RP": "#aec7e8",
    "UH": "#ff9896",
    ".": "#c5b0d5",
    "SYM": "#c49c94",
    "FW": "#f7b6d2",
    "": "#dddddd"
}

# Tag to full name
TAG_TRANSLATIONS = {
    "NN": "Noun",
    "NNP": "Proper Noun",
    "VB": "Verb",
    "JJ": "Adjective",
    "RB": "Adverb",
    "PRP": "Pronoun",
    "DT": "Determiner",
    "IN": "Preposition",
    "CC": "Coordinating Conjunction",
    "CD": "Cardinal Number",
    "RP": "Particle",
    "UH": "Interjection",
    ".": "Punctuation",
    ",": "Punctuation",
    ":": "Punctuation",
    "SYM": "Symbol",
    "FW": "Foreign Word",
    "": "Unknown"
}

def render_tagged_tokens(tokens, tags):
    html_content = "<div style='display: flex; gap: 10px; flex-wrap: wrap;'>"
    for token, tag in zip(tokens, tags):
        color = TAG_COLORS.get(tag, "#dddddd")
        label = TAG_TRANSLATIONS.get(tag, "Unknown")
        html_content += f"""
            <div style="
                border: 3px solid {color};
                padding: 8px;
                border-radius: 8px;
                text-align: center;
                width: fit-content;
                min-width: 60px;
            ">
                <div style="font-weight: bold;">{token}</div>
                <div style="font-size: 12px; color: #555;">{label}</div>
            </div>
        """
    html_content += "</div>"
    html(html_content, height=200)

# UI in English
st.title("Sentence Tagging")

text_input = st.text_area("Enter your text:", height=300)
language = st.selectbox("Select language:", ["Spanish", "English"])
language_code = {"Spanish": 0, "English": 1}

if st.button("Analyze"):
    if text_input.strip():
        payload = {"text": text_input, "lang": language_code[language]}
        try:
            response = requests.post("http://api:8000/get_tag", json=payload)
            if response.status_code == 200:
                result = response.json()
                render_tagged_tokens(result["tokens"], result["tags"])
            else:
                st.error(f"Server error: {response.status_code}")
        except Exception as e:
            st.error(f"Backend connection error: {e}")
    else:
        st.warning("Please enter some text.")
