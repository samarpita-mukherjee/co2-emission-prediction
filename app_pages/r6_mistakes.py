import streamlit as st

from app_pages.common import FIG, RES, image, load_table, need, next_link, takeaway

need(RES / "worst_mistakes.csv", RES / "error_by_fuel.csv", FIG / "Fig10_error_analysis.png")
worst = load_table("worst_mistakes")
by_fuel = load_table("error_by_fuel")

st.title("6. Where the model goes wrong")
takeaway("Diesel and petrol cars are predicted well. The **big mistakes are unusual cars**: the seven features do not describe special technology.")

image(FIG / "Fig10_error_analysis.png", "Left: predicted against actual CO2 (the dashed line is perfect). Right: average error by fuel type.")

left, right = st.columns([1.5, 1])
with left:
    st.markdown("**The biggest mistakes on the test set**")
    w = worst[["Description", "Fuel", "Actual", "Predicted"]].copy()
    for col in ["Actual", "Predicted"]:
        w[col] = w[col].round(0).astype(int)
    st.table(w.set_index("Description").head(6))
with right:
    st.markdown("**Average error by fuel**")
    names = {"GO": "Diesel", "ES": "Petrol"}
    for name, row in by_fuel.iterrows():
        st.metric(f"{names.get(name, 'Hybrid, gas, other')} ({int(row['test cars']):,} cars)", f"{row['MAE (g/km)']:.1f} g/km")
next_link("app_pages/r7_labels.py", "Next: the A to G label")
