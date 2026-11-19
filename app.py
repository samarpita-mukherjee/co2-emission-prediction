"""
Streamlit app for the project defense.

Start it from the project folder with:   py -m streamlit run app.py
Everything was computed by main.py in advance (results/ and outputs/), 
so the app opens instantly and never retrains anything.
The only live part is the prediction of one car with the saved model.
"""
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CO2 emissions of cars", page_icon="\U0001F697", layout="wide", initial_sidebar_state="expanded")

# Large text and clean look for a screen share (Zoom shrinks the picture, so everything is bigger than usual)
st.markdown("""
<style>
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1500px; }
h1 { font-size: 2.7rem !important; color: #1F3864; }
h2, h3 { color: #2F5496; }
.stMarkdown p, .stMarkdown li { font-size: 1.28rem; line-height: 1.55; }
div[data-testid="stCaptionContainer"] p { font-size: 1.05rem; }
div[data-testid="stMetricValue"] { font-size: 3.1rem; color: #1F3864; }
div[data-testid="stMetricLabel"] p { font-size: 1.15rem; }
div[data-testid="stTable"] table { font-size: 1.12rem; }
div[data-testid="stTable"] th { background: #D9E2F3; }
button[data-baseweb="tab"] p { font-size: 1.2rem; }
section[data-testid="stSidebar"] a span, section[data-testid="stSidebar"] p { font-size: 1.05rem; }
.takeaway { border-left: 10px solid #2F5496; background: #EEF3FA; padding: 0.9rem 1.3rem; margin: 0.4rem 0 1.2rem 0;
            font-size: 1.5rem; line-height: 1.45; border-radius: 4px; color: #1F2937; }
.warnbox { border-left: 10px solid #BF8F00; background: #FFF2CC; padding: 0.7rem 1.1rem; margin: 0.4rem 0;
           font-size: 1.15rem; border-radius: 4px; }
.labelstrip { display: flex; gap: 6px; margin: 0.6rem 0 1rem 0; }
.labelbox { flex: 1; text-align: center; padding: 0.7rem 0.2rem; border-radius: 6px; font-weight: 700; font-size: 1.15rem; }
.labelbox small { display: block; font-weight: 400; font-size: 0.9rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Load the model once at the start, so the first prediction on the predictor page is instant
try:
    from app_pages.common import load_model
    load_model()
except Exception:
    pass

pages = {
    "Start": [
        st.Page("app_pages/home.py", title="Overview", icon="\U0001F3E0", default=True),
    ],
    "Part 1: Regression": [
        st.Page("app_pages/r1_data.py", title="1. Data and the problem", icon="\U0001F4CA"),
        st.Page("app_pages/r2_features.py", title="2. Features and leakage", icon="\U0001F50D"),
        st.Page("app_pages/r3_split.py", title="3. The split by car model", icon="\u2702\uFE0F"),
        st.Page("app_pages/r4_models.py", title="4. Pipeline and models", icon="\u2699\uFE0F"),
        st.Page("app_pages/r5_results.py", title="5. Final results", icon="\U0001F3AF"),
        st.Page("app_pages/r6_mistakes.py", title="6. Where it goes wrong", icon="\u26A0\uFE0F"),
        st.Page("app_pages/r7_labels.py", title="7. The A to G label", icon="\U0001F3F7\uFE0F"),
        st.Page("app_pages/r8_predictor.py", title="8. Predict a car (live)", icon="\U0001F697"),
        st.Page("app_pages/r9_limits.py", title="9. Limits", icon="\U0001F6A7"),
    ],
    "Part 2: Classification": [
        st.Page("app_pages/c1_setup.py", title="1. Set-up", icon="\U0001F4C1"),
        st.Page("app_pages/c2_results.py", title="2. Results", icon="\U0001F4C8"),
        st.Page("app_pages/c3_analysis.py", title="3. Analysis", icon="\U0001F9E9"),
    ],
    "Closing": [
        st.Page("app_pages/closing.py", title="Conclusion", icon="\u2705"),
    ],
}
st.navigation(pages).run()
