"""Shared helpers of the app: paths, cached loaders and small layout functions."""
import re
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "regression"               # written by main.py
RES_CLF = ROOT / "results" / "classification"       # the files of the classification part (added by the partner)
RES_JOINT = ROOT / "results" / "joint"
FIG = ROOT / "outputs" / "figures"                  # written by main.py
ASSETS = ROOT / "assets"
MODEL_PATH = ROOT / "outputs" / "co2_final_model.joblib"


@st.cache_data
def load_table(name):
    """A table saved by main.py (results/regression/<name>.csv); the first column is the index."""
    return pd.read_csv(RES / f"{name}.csv", index_col=0)


@st.cache_data
def load_facts():
    import json
    with open(RES / "key_facts.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)


def need(*paths):
    """Stop the page with a clear message if a file made by main.py is missing."""
    missing = [str(Path(p).relative_to(ROOT)) for p in paths if not Path(p).exists()]
    if missing:
        st.error("Missing file(s): " + ", ".join(missing) + ". Run `py main.py` once to create them.")
        st.stop()


def takeaway(text):
    """The one-sentence message of a page, in a coloured box. **bold** works."""
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(f'<div class="takeaway">{html}</div>', unsafe_allow_html=True)


def warnbox(text):
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(f'<div class="warnbox">{html}</div>', unsafe_allow_html=True)


def image(path, caption=None):
    need(path)
    st.image(str(path), caption=caption, width="stretch")


def pretty(df, decimals=2):
    """Round the numbers of a table to fixed decimals (as text), so st.table shows 12.80 and not 12.8."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}")
    return out


def next_link(page_file, label):
    st.page_link(page_file, label=label + "  \u2192")


def nice_name(filename):
    """01_split_by_model.csv -> 'Split by model'"""
    stem = Path(filename).stem
    stem = re.sub(r"^\d+[_\-\s]*", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


def render_folder(folder, title, empty_hint):
    """Show every file of a folder: notes.md as text, .csv as a table, .png/.jpg as a picture (in file-name order).
    This is how the classification part is added without writing any code: just put the files in the folder."""
    st.title(title)
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".csv", ".png", ".jpg", ".jpeg", ".md")) if folder.exists() else []
    if not files:
        st.info(empty_hint)
        return
    for p in files:
        suffix = p.suffix.lower()
        if suffix == ".md":
            st.markdown(p.read_text(encoding="utf-8"))
        elif suffix == ".csv":
            st.subheader(nice_name(p.name))
            st.table(pd.read_csv(p))
        else:
            st.image(str(p), caption=nice_name(p.name), width="stretch")
