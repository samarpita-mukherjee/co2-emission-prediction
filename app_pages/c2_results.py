import streamlit as st

from app_pages.common import RES_CLF, render_folder

render_folder(RES_CLF / "2_results", "Part 2: Classification, results",
              "Not added yet. Put the files of this page in results/classification/2_results/ "
              "(tables as .csv, figures as .png, text in notes.md).")
st.page_link("app_pages/c3_analysis.py", label="Next: analysis  \u2192")
