import streamlit as st

from app_pages.common import RES, load_facts, load_table, need, next_link, pretty, takeaway

need(RES / "final_test.csv", RES / "key_facts.json")
f = load_facts()
tbl = load_table("final_test")
final = tbl.iloc[-1]                    # the last row is the final model
base = tbl.iloc[0]                      # the first row is "always predict the average"

st.title("5. The final test")
takeaway(f"On **{f['rows_test']:,} test cars** of **{f['car_models_test']} car models never seen in training**, the final model is off by "
         f"**{final['MAE (g/km)']:.1f} g/km** on average. Guessing the average is off by {base['MAE (g/km)']:.1f}.")

a, b, c, d = st.columns(4)
a.metric("MAE (average error)", f"{final['MAE (g/km)']:.1f} g/km")
b.metric("RMSE", f"{final['RMSE (g/km)']:.1f} g/km")
c.metric("R2", f"{final['R2']:.2f}")
d.metric("Within 10 g/km", f"{final['% within 10 g/km']:.0f}%")

left, right = st.columns([1.3, 1])
with left:
    st.table(pretty(tbl, 2))
with right:
    st.markdown("- **MAE:** the average size of the error.\n"
                "- **RMSE:** like MAE, but big errors count more. It is larger than MAE: a few predictions are far off.\n"
                "- **R2:** the share of the differences between cars that the model explains (1 = perfect).\n"
                "- Tested **once**. Nothing was changed afterwards.")
st.caption("The test set has no vans, and one car model is about 63% of its rows: the score describes ordinary cars and is noisy.")
next_link("app_pages/r6_mistakes.py", "Next: where it goes wrong")
