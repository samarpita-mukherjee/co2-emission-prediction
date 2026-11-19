import streamlit as st

from app_pages.common import ASSETS, FIG, RES, image, load_facts, load_table, need, next_link, pretty, takeaway

need(RES / "cv_models.csv", RES / "tuning.csv", RES / "key_facts.json", ASSETS / "pipeline.png", ASSETS / "folds.png", FIG / "Fig9_model_comparison.png")
f = load_facts()

st.title("4. The pipeline and the models")
takeaway("The **encoding is part of the pipeline**: it learns from the training rows only. **LightGBM with sample weights** is the best of seven models.")

t1, t2, t3, t4 = st.tabs(["Pipeline", "Cross-validation", "Seven models", "Tuning"])

with t1:
    image(ASSETS / "pipeline.png")
    st.markdown("- **fit:** the encoder learns its rules (categories, medians, averages) from the training rows, then the model learns.\n"
                "- **predict:** only applies the stored rules, never learns again.\n"
                "- In cross-validation a **new pipeline per round**, so nothing from the validation cars leaks in.")

with t2:
    left, right = st.columns([1.5, 1])
    with left:
        image(ASSETS / "folds.png")
    with right:
        st.markdown("- **GroupKFold:** the folds are made of **car models**, like the final split.\n"
                    "- Every training row is predicted by a model that **never saw its car model**.\n"
                    "- Repeated **5 times** with different fold assignments (seeds 0 to 4): one assignment can be lucky.\n"
                    "- The test set is **never** used to choose.")

with t3:
    left, right = st.columns([1.5, 1])
    cv = load_table("cv_models")[["MAE", "MAE_model_weighted", "R2"]].rename(columns={"MAE": "MAE (g/km)", "MAE_model_weighted": "Equal-weight MAE (g/km)"})
    with left:
        image(FIG / "Fig9_model_comparison.png")
    with right:
        st.table(pretty(cv))
        st.markdown("**Equal-weight MAE:** every car model counts the same, so the vans (73% of the rows) do not dominate. "
                    "**Sample weights** (1 / square root of the rows of a car model) help both boosted-tree models.")

with t4:
    tune = load_table("tuning").rename(columns={"MAE_equal": "Equal-weight MAE", "repeats won on MAE (of 5)": "Repeats won: MAE",
                                                  "repeats won on MAE_equal (of 5)": "Repeats won: equal-weight MAE"})
    st.table(pretty(tune))
    if f["tuning"]["use_tuned"]:
        st.markdown("**Rule fixed in advance:** adopt tuned settings only if better in at least 4 of 5 repeats on both measures. "
                    "They passed, so they are used.")
    else:
        st.markdown("**Rule fixed in advance:** adopt tuned settings only if better in at least 4 of 5 repeats on **both** measures. "
                    "The tuned settings help ordinary car models but are worse overall: a trade-off, so **not adopted**. The final model keeps the starting settings.")
next_link("app_pages/r5_results.py", "Next: final results")
