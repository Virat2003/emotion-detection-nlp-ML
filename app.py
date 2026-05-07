import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="EmotiScan · Sentiment Analyzer",
    page_icon="🧠",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0b0c10;
    --surface:   #13151c;
    --border:    #1f2130;
    --text:      #e8eaf0;
    --muted:     #6b7080;
    --accent:    #7df9c8;
    --sadness:   #5b9cf6;
    --anger:     #f96d5b;
    --love:      #f96db0;
    --surprise:  #f9d85b;
    --fear:      #c07df9;
    --joy:       #7df9c8;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem !important; max-width: 720px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-size: clamp(2.2rem, 6vw, 3.4rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem;
}
.hero-sub {
    font-size: 0.92rem;
    color: var(--muted);
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Input card  (targets st.container border wrapper) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.75rem 1.75rem 1.25rem !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4) !important;
    margin-bottom: 0.5rem !important;
}
.input-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}

/* Override Streamlit textarea */
textarea {
    background-color: #0b0c10 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    resize: vertical !important;
    caret-color: var(--accent) !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(125,249,200,0.08) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: transparent !important;
    border: 1.5px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 10px !important;
    transition: background 0.2s, color 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 12px 50px rgba(0,0,0,0.5);
    animation: slide-up 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes slide-up {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.result-emoji {
    font-size: 3.8rem;
    line-height: 1;
    margin-bottom: 0.4rem;
    display: block;
    filter: drop-shadow(0 0 18px currentColor);
}
.result-emotion {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.01em;
}

/* emotion-specific tints */
.sadness  { background: linear-gradient(135deg, #0d1629, #0b1221); border-color: var(--sadness) !important; color: var(--sadness); }
.anger    { background: linear-gradient(135deg, #1e0d0a, #150906); border-color: var(--anger)   !important; color: var(--anger);   }
.love     { background: linear-gradient(135deg, #1e0a13, #120508); border-color: var(--love)    !important; color: var(--love);    }
.surprise { background: linear-gradient(135deg, #1c1806, #110e03); border-color: var(--surprise)!important; color: var(--surprise);}
.fear     { background: linear-gradient(135deg, #12091e, #0b0514); border-color: var(--fear)    !important; color: var(--fear);    }
.joy      { background: linear-gradient(135deg, #061912, #040f0a); border-color: var(--joy)     !important; color: var(--joy);     }

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    padding: 2rem 0 1rem;
}

/* emotion legend */
.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1rem;
}
.legend-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
    background: var(--surface);
}
</style>
""", unsafe_allow_html=True)

# ── NLTK downloads ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()

# ── Load model & vectorizer ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("model.pkl",      "rb") as f: model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f: cv    = pickle.load(f)
    return model, cv

try:
    model, cv = load_artifacts()
    model_ready = True
except FileNotFoundError as e:
    model_ready = False
    missing = str(e)

# ── Text cleaning ────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(txt: str) -> str:
    txt = re.sub(r'https?://\S+|www\.\S+', '', txt)
    txt = re.sub(r'\d+', '', txt)
    txt = txt.lower()
    words = word_tokenize(txt)
    return " ".join(w for w in words if w not in stop_words)

# ── Emotion config ───────────────────────────────────────────────────────────
EMOTIONS = {
    0: {"label": "Sadness",  "emoji": "😢", "cls": "sadness"},
    1: {"label": "Anger",    "emoji": "😡", "cls": "anger"},
    2: {"label": "Love",     "emoji": "❤️",  "cls": "love"},
    3: {"label": "Surprise", "emoji": "😲", "cls": "surprise"},
    4: {"label": "Fear",     "emoji": "😨", "cls": "fear"},
    5: {"label": "Joy",      "emoji": "😊", "cls": "joy"},
}

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Natural Language Processing</div>
    <h1 class="hero-title">EmotionScan</h1>
    <p class="hero-sub">Paste any text below and discover the emotion behind it.</p>
</div>
""", unsafe_allow_html=True)

# ── Model error banner ───────────────────────────────────────────────────────
if not model_ready:
    st.error(f"⚠️ Could not load model files — make sure `model.pkl` and `vectorizer.pkl` are in the same directory.\n\n`{missing}`")
    st.stop()

# ── Input card ───────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<p class="input-label">Your Text</p>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="text_input",
        label_visibility="collapsed",
        placeholder="Type or paste your sentence here…",
        height=160,
        key="user_input",
    )
    predict_clicked = st.button("⟶  Analyze Emotion", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────────────
if predict_clicked:
    raw = user_input.strip()

    if not raw:
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing…"):
            cleaned = clean_text(raw)
            vector  = cv.transform([cleaned])
            pred    = model.predict(vector)[0]

        emotion = EMOTIONS.get(pred, {"label": "Unknown", "emoji": "❓", "cls": "joy"})

        st.markdown(f"""
        <div class="result-card {emotion['cls']}">
            <div class="result-label">Detected Emotion</div>
            <span class="result-emoji">{emotion['emoji']}</span>
            <div class="result-emotion">{emotion['label']}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Legend ───────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
pills = "".join(
    f'<span class="legend-pill">{e["emoji"]} {e["label"]}</span>'
    for e in EMOTIONS.values()
)
st.markdown(f'<div class="legend">{pills}</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    EmotiScan · NLP Sentiment Analysis · 6-class emotion detection
</div>
""", unsafe_allow_html=True)