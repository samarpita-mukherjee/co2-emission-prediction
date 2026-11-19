import altair as alt
import pandas as pd
import streamlit as st

from app_pages.common import RES, load_table, need, next_link, takeaway

need(RES / "ablation.csv")
abl = load_table("ablation")
core = float(abl.loc["CORE (the 7 kept features)", "MAE (g/km)"])

st.title("2. Which columns may be used? Features and leakage")
takeaway("**Rule:** a feature may only be used if an engineer knows it **before the car exists**.")

left, right = st.columns([1, 1.25])
with left:
    st.markdown("**Dropped columns**")
    st.table(pd.DataFrame({
        "Group": ["Identifiers, names", "CO2 in other units", "Same lab test as CO2", "Redundant or no benefit"],
        "Examples": ["type-approval code, model name", "consumption, administrative power", "NOX, CO, particles, HC", "minimum mass, gamme, brand"],
    }).set_index("Group"))
    st.markdown("**The 7 features kept:** fuel, power, mass, body type, gearbox type, number of gears, hybrid.")
    st.caption("Ablation: the same model is trained with and without a column (training part only, grouped cross-validation, "
               f"3 repeats). Reference error with the 7 features: {core:.2f} g/km.")

LABELS = {"remove Puissance maximale (kW)": "Remove power", "remove masse vide euro max (kg)": "Remove mass", "remove nb_rapports": "Remove number of gears",
          "remove Carburant": "Remove fuel", "remove boite_type": "Remove gearbox type", "remove Carrosserie": "Remove body type",
          "remove Hybride": "Remove hybrid", "add Marque": "Add brand", "add gamme": "Add gamme", "add masse vide euro min (kg)": "Add minimum mass",
          "add Consommation mixte (l/100km)": "Add consumption", "add Puissance administrative": "Add administrative power", "add NOX (g/km)": "Add NOX"}
KIND = {"Add brand": "left out on purpose", "Add gamme": "makes it worse", "Add minimum mass": "makes it worse",
        "Add consumption": "not allowed (leakage)", "Add administrative power": "not allowed (leakage)", "Add NOX": "not allowed (leakage)"}
COLORS = {"important feature": "#2F5496", "left out on purpose": "#7F7F7F", "makes it worse": "#8FAADC", "not allowed (leakage)": "#DD8452"}


def chart(rows, title):
    d = pd.DataFrame({"label": [LABELS[i] for i in rows], "change": [float(abl.loc[i, "change vs CORE"]) for i in rows]})
    d["kind"] = [KIND.get(l, "important feature") for l in d["label"]]
    base = alt.Chart(d).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelFontSize=17, labelLimit=420)),
        x=alt.X("change:Q", title=title, axis=alt.Axis(titleFontSize=15, labelFontSize=14)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())),
                        legend=alt.Legend(title=None, labelFontSize=14, orient="bottom")))
    bars = base.mark_bar()
    text = base.mark_text(align="left", dx=4, fontSize=16, color="#1F2937").encode(text=alt.Text("change:Q", format="+.2f"))
    return (bars + text).properties(height=45 * len(d) + 70)


with right:
    t1, t2 = st.tabs(["Remove a feature", "Add a column"])
    with t1:
        order = ["remove masse vide euro max (kg)", "remove Puissance maximale (kW)", "remove nb_rapports", "remove Carburant",
                 "remove boite_type", "remove Carrosserie", "remove Hybride"]
        st.altair_chart(chart(order, "Change in error (g/km) when the feature is removed"), use_container_width=True)
        st.markdown("Mass, power, gears and fuel clearly matter. Body type and hybrid are inside the noise: kept for **physical reasons**.")
    with t2:
        order = ["add Marque", "add gamme", "add masse vide euro min (kg)", "add Puissance administrative", "add NOX (g/km)", "add Consommation mixte (l/100km)"]
        st.altair_chart(chart(order, "Change in error (g/km) when the column is added (negative = lower error)"), use_container_width=True)
        st.markdown("Some columns lower the error **only because they contain the answer**. They stay out. Brand helps a little, but it names the maker, not the design.")
next_link("app_pages/r3_split.py", "Next: the split")
