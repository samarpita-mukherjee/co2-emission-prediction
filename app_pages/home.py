import streamlit as st

from app_pages.common import RES, RES_CLF, load_facts, need, next_link, takeaway

PRESENTERS = "Samarpita Mukherjee and Matthew Miller"

need(RES / "key_facts.json")
f = load_facts()

st.title("Predicting the CO2 emissions of cars")
st.subheader("Two approaches to one question: regression and classification")
if PRESENTERS:
    st.caption(PRESENTERS)

takeaway("**The question:** can we predict the CO2 emissions (g/km) of a **new car design** from information that an engineer knows **before the car exists**?")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows after cleaning", f"{f['rows_clean']:,}")
c2.metric("Different car models", f"{f['car_models']:,}")
c3.metric("Mercedes-Benz, % of rows", f"{f['mercedes_share_pct']:.0f}%")
c4.metric("Three van models, % of rows", f"{f['vans_share_pct']:.0f}%")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Part 1: Regression")
        st.markdown("Predict the **number** of g/km. The official A to G CO2 label is then derived from the predicted number.")
with right:
    with st.container(border=True):
        st.subheader("Part 2: Classification")
        summary = RES_CLF / "summary.md"
        if summary.exists():
            st.markdown(summary.read_text(encoding="utf-8"))
        else:
            st.markdown("Predict the CO2 **class** (A to G) directly.")

st.markdown("**Plan:** Part 1 (about 10 minutes), Part 2 (about 10 minutes), then questions.")
next_link("app_pages/r1_data.py", "Start with Part 1")
