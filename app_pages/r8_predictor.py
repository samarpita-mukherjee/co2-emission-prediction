import warnings

import pandas as pd
import streamlit as st

from app_pages.common import MODEL_PATH, RES, load_facts, load_model, load_table, need, takeaway
from src.evaluation import LABEL_TEXT, LABELS, co2_label
from src.modeling import EXAMPLE_CARS, predict_co2

warnings.filterwarnings("ignore", message="X does not have valid feature names")
need(MODEL_PATH, RES / "key_facts.json", RES / "final_test.csv", RES / "error_by_fuel.csv")
model = load_model()
f = load_facts()
test_mae = float(load_table("final_test").iloc[-1]["MAE (g/km)"])
by_fuel = load_table("error_by_fuel")

HYBRID_CODES = {"EH", "GH", "EE", "GL"}                 # every hybrid car has one of these fuel codes (fuel code x hybrid table)
FUEL_NAMES = {"GO": "diesel", "ES": "petrol", "EH": "petrol hybrid", "ES/GN": "petrol / natural gas"}
COLORS = ["#009A44", "#52AE32", "#C8D400", "#FFED00", "#FBBA00", "#EB690B", "#E2001A"]
TEXT = ["#FFFFFF", "#FFFFFF", "#1F2937", "#1F2937", "#1F2937", "#FFFFFF", "#FFFFFF"]

cats = f["categories"]
rare_fuel = set(f["infrequent"]["fuel"])
main_first = [c for c in ["GO", "ES", "EH"] if c in cats["fuel"]]
fuel_options = main_first + [c for c in cats["fuel"] if c not in main_first]
rng = f["ranges"]


def fuel_label(code):
    name = FUEL_NAMES.get(code, "rare code" if code in rare_fuel else "")
    return f"{code}  ({name})" if name else code


# ---- state: the default is the diesel estate car; the example buttons overwrite the widget values
DEFAULTS = {"fuel": "GO", "body": "BREAK", "gearbox": "A", "power": 110, "mass": 1650, "gears": 7, "hybrid": False}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_example(car):
    st.session_state.update({"fuel": car["carburant"], "body": car["carrosserie"], "gearbox": car["boite_type"],
                             "power": int(car["power_kw"]), "mass": int(car["mass_kg"]), "gears": int(car["nb_rapports"]), "hybrid": False})


st.title("8. Predict the CO2 of a car (live, saved model)")
takeaway("The model was trained once and saved. Here it predicts a **new car** from the seven design features. **Nothing is retrained.**")

st.markdown("**Load an example:**")
cols = st.columns(len(EXAMPLE_CARS))
for col, (label, car) in zip(cols, EXAMPLE_CARS):
    col.button(label, on_click=load_example, args=(car,), use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.selectbox("Fuel", fuel_options, format_func=fuel_label, key="fuel")
    st.slider("Power (kW)", int(rng["power_kw"][0]), int(rng["power_kw"][1]), step=1, key="power")
    st.slider("Maximum mass (kg)", int(rng["mass_kg"][0]), int(rng["mass_kg"][1]), step=10, key="mass")
    c1, c2 = st.columns(2)
    c1.selectbox("Body type", cats["body"], key="body")
    c2.selectbox("Gearbox type (M = manual, A = automatic)", cats["gearbox"], key="gearbox")
    st.slider("Number of gears", int(rng["gears"][0]), int(rng["gears"][1]), step=1, key="gears")
    st.checkbox("Hybrid", key="hybrid")

s = st.session_state
args = dict(carburant=s["fuel"], power_kw=s["power"], mass_kg=s["mass"], carrosserie=s["body"], boite_type=s["gearbox"],
            nb_rapports=s["gears"], hybride=int(s["hybrid"]))
columns = list(model.feature_names_in_)
pred = predict_co2(model, columns, **args)
cls = int(co2_label(pred))
low, high = int(co2_label(pred - test_mae)), int(co2_label(pred + test_mae))

with right:
    st.metric("Predicted CO2", f"{pred:.0f} g/km")
    st.markdown(f'<div class="labelbox" style="background:{COLORS[cls]};color:{TEXT[cls]};font-size:1.6rem;margin:0.3rem 0">'
                f'Class {LABELS[cls]}<small>{LABEL_TEXT[cls]} g/km</small></div>', unsafe_allow_html=True)
    if low == high:
        st.markdown(f"With the typical error of the model (about **{test_mae:.0f} g/km**), the class stays **{LABELS[cls]}**.")
    else:
        st.markdown(f"With the typical error of the model (about **{test_mae:.0f} g/km**), the class could be **{LABELS[low]} to {LABELS[high]}**.")

    # honest warnings: where the model is weak or has no evidence
    notes = []
    if s["hybrid"] and s["fuel"] not in HYBRID_CODES:
        notes.append("Hybrid cars have a hybrid fuel code (EH, GH, EE, GL). This combination does not exist in the data.")
    if (not s["hybrid"]) and s["fuel"] in HYBRID_CODES:
        notes.append("This fuel code belongs to hybrid cars: tick Hybrid.")
    if s["fuel"] in HYBRID_CODES or s["hybrid"]:
        other_err = float(by_fuel.iloc[-1]["MAE (g/km)"])
        notes.append(f"Hybrids are predicted badly (average error of the group \"other fuels\" on the test cars: about {other_err:.0f} g/km).")
    elif s["fuel"] in rare_fuel:
        notes.append("Rare fuel code (fewer than 50 training rows): grouped with other rare codes, so the prediction is less reliable.")
    if s["body"] == "MINIBUS":
        notes.append("Vans and minibuses: the test set had no vans, so this prediction was not tested.")
    for text in notes:
        st.markdown(f'<div class="warnbox">{text}</div>', unsafe_allow_html=True)

with st.expander("Under the hood: what the pipeline does with this car"):
    car = pd.DataFrame([{"Carburant": s["fuel"], "Hybride": int(s["hybrid"]), "Puissance maximale (kW)": s["power"],
                         "masse vide euro max (kg)": s["mass"], "Carrosserie": s["body"], "boite_type": s["gearbox"],
                         "nb_rapports": s["gears"]}])[columns]
    prep = model.named_steps["prep"]
    row = pd.Series(prep.transform(car)[0], index=prep.get_feature_names_out())
    shown = row[row != 0]
    table = pd.DataFrame({"Encoded column": [n.split("__", 1)[-1] for n in shown.index], "Value": [f"{v:.3f}" for v in shown.values]}).set_index("Encoded column")
    st.markdown(f"The encoder turns the 7 features into **{len(row)} numbers**. Only these are not zero "
                f"(text categories become 0/1 columns, numbers are scaled):")
    st.table(table)
    st.markdown("The model (LightGBM) then predicts from these numbers.")
import lightgbm
import sklearn
st.caption(f"Typical error on the test cars: {test_mae:.1f} g/km. Running with scikit-learn {sklearn.__version__} and LightGBM {lightgbm.__version__}.")
