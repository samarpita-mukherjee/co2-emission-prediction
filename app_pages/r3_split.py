import pandas as pd
import streamlit as st

from app_pages.common import ASSETS, RES, image, load_facts, load_table, need, next_link, takeaway

need(RES / "split_composition.csv", RES / "key_facts.json", ASSETS / "split.png")
f = load_facts()
comp = load_table("split_composition")
big = load_table("largest_test_models")

st.title("3. The split: by car model, not by row")
takeaway("Variants of one car model are almost identical. We split by **car model**, so the test cars are **designs the model has never seen**.")

left, right = st.columns([1.35, 1])
with left:
    image(ASSETS / "split.png", "Each colour is one car model. A random split puts siblings on both sides; our split keeps each model on one side.")
with right:
    show = pd.DataFrame({
        "Train": [f"{int(comp.loc['train', 'rows']):,}", f"{int(comp.loc['train', 'car models']):,}",
                  f"{comp.loc['train', '% Mercedes']:.0f}%", f"{comp.loc['train', '% the 3 vans']:.0f}%"],
        "Test": [f"{int(comp.loc['test', 'rows']):,}", f"{int(comp.loc['test', 'car models']):,}",
                 f"{comp.loc['test', '% Mercedes']:.0f}%", f"{comp.loc['test', '% the 3 vans']:.0f}%"]},
        index=["Rows", "Car models", "Mercedes-Benz", "The three vans"])
    st.table(show)
    top_name, top_share = big.index[0], float(big.iloc[0, 0])
    st.markdown(
        f"- The test set has **no vans**.\n"
        f"- One car model (**{top_name}**) is **{top_share:.0f}%** of the test rows, so the score is noisy.\n"
        f"- The 80% / 20% refers to car models: {f['car_models_test']} of {f['car_models']} models, "
        f"but only {100 * f['rows_test'] / f['rows_clean']:.0f}% of the rows.")
warn = "A random split would test the model on cars it has already seen, and the score would look better than it is. Seed 42 is one draw."
st.caption(warn)
next_link("app_pages/r4_models.py", "Next: pipeline and models")
