import streamlit as st

from app_pages.common import RES, RES_CLF, RES_JOINT, load_facts, need, takeaway

need(RES / "key_facts.json")
f = load_facts()

st.title("Conclusion")
takeaway("Using only information known at design time, the regression model predicts CO2 for **unseen car models** with an average error of about 13 g/km.")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Part 1: Regression")
        st.markdown(
            "- Split by **car model**: the test measures new designs.\n"
            "- Columns that contain the answer (**leakage**) were removed.\n"
            "- LightGBM with **sample weights** is the best of seven models.\n"
            "- The A to G label follows from the predicted number.\n"
            f"- Limits: one maker is {f['mercedes_share_pct']:.0f}% of the data, the test set has no vans.")
with right:
    with st.container(border=True):
        st.subheader("Part 2: Classification")
        takeaways = RES_CLF / "takeaways.md"
        if takeaways.exists():
            st.markdown(takeaways.read_text(encoding="utf-8"))
        else:
            st.markdown("Key messages of the classification part.")

joint = RES_JOINT / "notes.md"
if joint.exists():
    with st.container(border=True):
        st.subheader("How the two parts connect")
        st.markdown(joint.read_text(encoding="utf-8"))

st.markdown("## Thank you. Questions?")
