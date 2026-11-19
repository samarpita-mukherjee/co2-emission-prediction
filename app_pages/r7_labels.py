import html

import streamlit as st

from app_pages.common import FIG, RES, image, load_table, need, next_link, pretty, takeaway

need(RES / "class_counts.csv", RES / "class_scores.csv", FIG / "Fig12_label_confusion.png")
counts = load_table("class_counts")
scores = load_table("class_scores")
ours = scores.loc["Regression model, then cut"]
baseline = scores.iloc[0]

COLORS = ["#009A44", "#52AE32", "#C8D400", "#FFED00", "#FBBA00", "#EB690B", "#E2001A"]
TEXT = ["#FFFFFF", "#FFFFFF", "#1F2937", "#1F2937", "#1F2937", "#FFFFFF", "#FFFFFF"]

st.title("7. From the number to the CO2 label (A to G)")
takeaway(f"**Regression, then cut:** predict the number, then look up its class. The class is **at most one step away for {ours['within 1 class (%)']:.1f}%** of the cars.")

boxes = ""
for (letter, row), color, text in zip(counts.iterrows(), COLORS, TEXT):
    boxes += (f'<div class="labelbox" style="background:{color};color:{text}">{html.escape(letter)}'
              f'<small>{html.escape(str(row["CO2 range (g/km)"]))} g/km</small><small>{int(row["test cars"]):,} test cars</small></div>')
st.markdown(f'<div class="labelstrip">{boxes}</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1])
with left:
    a, b = st.columns(2)
    a.metric("Exact class", f"{ours['exact class (%)']:.1f}%")
    b.metric("Within one class", f"{ours['within 1 class (%)']:.1f}%")
    st.metric("Average distance", f"{ours['average distance (classes)']:.1f} classes")
    st.caption(f"Always guessing the most common training class: exact {baseline['exact class (%)']:.1f}%, "
               f"within one class {baseline['within 1 class (%)']:.1f}%.")
    st.markdown("Classes B, C and D are only **20 g/km wide**, about the size of the typical error, so neighbouring-class mistakes are expected. "
                "**Within one class** is the fairer number.")
with right:
    image(FIG / "Fig12_label_confusion.png", "True class (rows) against predicted class (columns)")
next_link("app_pages/r8_predictor.py", "Next: predict a car")
