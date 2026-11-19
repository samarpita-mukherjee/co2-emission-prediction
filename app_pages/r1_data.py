import streamlit as st

from app_pages.common import FIG, RES, image, load_facts, need, next_link, takeaway

need(RES / "key_facts.json")
f = load_facts()
co2 = f["co2"]

st.title("1. The data and the problem")
takeaway("One row is one **variant** of a car. The data are very unbalanced: **one maker dominates**.")

left, right = st.columns([1, 1.2])
with left:
    a, b = st.columns(2)
    a.metric("Rows after cleaning", f"{f['rows_clean']:,}")
    b.metric("Car models", f"{f['car_models']:,}")
    a.metric("Mercedes-Benz, % of rows", f"{f['mercedes_share_pct']:.0f}%")
    b.metric("Three van models, % of rows", f"{f['vans_share_pct']:.0f}%")
    st.markdown(
        f"- Cleaning: **{f['rows_without_co2']}** rows without CO2 and **{f['duplicate_rows']}** exact duplicates removed "
        f"({f['rows_raw']:,} to {f['rows_clean']:,} rows).\n"
        f"- CO2 is almost symmetric: mean **{co2['mean']:.0f}**, median **{co2['median']:.0f}** g/km, skewness {co2['skewness']:.2f}, "
        f"so no transformation is needed.")
with right:
    image(FIG / "Fig2_consumption_vs_co2.png", "Mixed fuel consumption against CO2 (diesel, petrol, petrol hybrid)")

takeaway("Consumption lies on an almost perfect line with CO2 (correlation "
         f"{co2['corr_consumption_co2']:.2f}): it is **CO2 in other units**. A new design has no measured consumption, so using it would be **leakage**.")
next_link("app_pages/r2_features.py", "Next: features and leakage")
