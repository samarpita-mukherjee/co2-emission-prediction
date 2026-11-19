import streamlit as st

from app_pages.common import RES_CLF, render_folder

render_folder(RES_CLF / "1_setup", "Part 2: Classification, set-up",
              "Not added yet. Put the files of this page in results/classification/1_setup/ "
              "(tables as .csv, figures as .png, text in notes.md).")
st.page_link("app_pages/c2_results.py", label="Next: results  \u2192")
