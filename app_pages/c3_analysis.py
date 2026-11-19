import streamlit as st

from app_pages.common import RES_CLF, render_folder

render_folder(RES_CLF / "3_analysis", "Part 2: Classification, analysis",
              "Not added yet. Put the files of this page in results/classification/3_analysis/ "
              "(tables as .csv, figures as .png, text in notes.md).")
st.page_link("app_pages/closing.py", label="Next: conclusion  \u2192")
