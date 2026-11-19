import streamlit as st

from app_pages.common import RES, load_facts, load_table, need, takeaway

need(RES / "key_facts.json", RES / "error_by_fuel.csv")
f = load_facts()
by_fuel = load_table("error_by_fuel")
other = by_fuel.iloc[-1] if len(by_fuel) else None

st.title("9. What the result can and cannot say")
takeaway("An honest reading: good for **ordinary diesel and petrol cars**, not for hybrids, special cars or vans.")

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Limits")
        st.markdown(
            f"- 2013 catalogue of laboratory values.\n"
            f"- **{f['mercedes_share_pct']:.0f}%** of the rows are Mercedes-Benz (three van models: {f['vans_share_pct']:.0f}%).\n"
            f"- The test set has **no vans**, and one car model is about 63% of it: the score is noisy.\n"
            f"- Hybrids and special cars are predicted badly (battery and technology are not features).\n"
            f"- The split by car model is stricter than a random split: the score is lower, on purpose.")
with right:
    with st.container(border=True):
        st.subheader("With more time or data")
        st.markdown(
            "- Cars from **other makers**, especially hybrids, and a more recent catalogue.\n"
            "- Better features: engine displacement, turbo, hybrid system, aerodynamics.\n"
            "- Repeat the split with **several seeds** to measure how uncertain the test score is.\n"
            "- Compare with the classification approach on the **same split** and the same A to G classes.")
st.caption("Everything shown in this app was computed by main.py in advance; only the prediction on the live page runs on the saved model.")
