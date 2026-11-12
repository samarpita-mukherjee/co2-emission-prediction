"""
evaluation.py 

The final test (done once), the model-comparison figure, the error analysis and the CO2 label classes A-G.
Figures are saved to files, so the project also runs on a server.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.pipeline import Pipeline

from src.data import GROUP_COL
from src.modeling import linear_preprocessor, make_ridge

FINAL_NAME = "Final model (LightGBM + weights)"


# =================================================================================================
# 1. Model comparison figure (cross-validation on the training set)
# =================================================================================================
def plot_model_comparison(table, title, path):
    """Two panels, one row per model, best model on top.
       LEFT  = ERROR   : MAE (all rows) and MAE with every car model counted equally; lower is better.
       RIGHT = ACCURACY: R2, the share of the CO2 variation the model explains (1 = perfect); higher is better.
       A good model is short on the left AND long on the right."""
    t = table[["MAE", "MAE_model_weighted", "R2"]].sort_values("MAE_model_weighted")
    y_pos = np.arange(len(t))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 0.62 * len(t) + 2.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    h = 0.38
    b1 = ax1.barh(y_pos - h / 2, t["MAE"], h, color="#4c72b0", label="MAE (all rows)")
    b2 = ax1.barh(y_pos + h / 2, t["MAE_model_weighted"], h, color="#dd8452", label="MAE, every car model equal")
    ax1.bar_label(b1, fmt="%.1f", padding=3, fontsize=8)
    ax1.bar_label(b2, fmt="%.1f", padding=3, fontsize=8)
    ax1.set(xlabel="Error in g/km (lower is better)", title="Error")
    ax1.set_xlim(0, t["MAE_model_weighted"].max() * 1.15)
    ax1.legend(loc="upper right", fontsize=8)
    b3 = ax2.barh(y_pos, t["R2"], 0.55, color="#55a868")
    ax2.bar_label(b3, fmt="%.2f", padding=3, fontsize=8)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set(xlabel="R2 (higher is better)", title="Accuracy (R2)")
    ax2.set_xlim(min(0, t["R2"].min()) - 0.3, 1.05)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(t.index); ax1.invert_yaxis()
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# =================================================================================================
# 2. Final test on the held-out test set (done once)
# =================================================================================================
def final_test(ds, final_model):
    """Reference models give the number a meaning: the dummy (always the average) and Ridge (a straight-line model).
    Returns the results table and the test predictions of every model."""
    dummy = Pipeline([("prep", linear_preprocessor()), ("m", DummyRegressor(strategy="mean"))]).fit(ds.X_train, ds.y_train)
    ridge = make_ridge().fit(ds.X_train, ds.y_train)

    models_on_test = {"Always predict the average": dummy,
                      "Ridge (straight-line model)": ridge,
                      FINAL_NAME: final_model}

    rows, test_pred = {}, {}
    for name, model in models_on_test.items():
        p = pd.Series(model.predict(ds.X_test), index=ds.X_test.index)
        err = ds.y_test - p
        rows[name] = {"MAE (g/km)": err.abs().mean(),
                      "RMSE (g/km)": float(np.sqrt((err ** 2).mean())),
                      "R2": r2_score(ds.y_test, p),
                      "% within 10 g/km": (err.abs() <= 10).mean() * 100}
        test_pred[name] = p
    test_table = pd.DataFrame(rows).T
    print(f"Test set: {len(ds.X_test)} cars of {ds.groups.loc[ds.X_test.index].nunique()} car models never seen in training\n")
    print(test_table.round(2).to_string())
    return test_table, test_pred


# =================================================================================================
# 3. Error analysis: where does the model make mistakes?
# =================================================================================================
def error_analysis(ds, pred, path):
    """Predicted vs actual, average error by fuel type, and the 10 biggest mistakes.
    error = actual CO2 - predicted CO2 (error < 0 = predicted too HIGH)."""
    y_test, X_test = ds.y_test, ds.X_test
    error = y_test - pred
    fuel_group = X_test["Carburant"].map(lambda f: f if f in ("GO", "ES") else "other (hybrid, gas...)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].scatter(y_test, pred, s=8, alpha=0.4, color="#2c7fb8")
    lims = [min(y_test.min(), pred.min()) - 5, max(y_test.max(), pred.max()) + 5]
    axes[0].plot(lims, lims, "k--", lw=1)                       # perfect predictions lie on this line
    axes[0].set(title="Predicted vs. actual CO2", xlabel="Actual CO2 (g/km)", ylabel="Predicted CO2 (g/km)")

    mae_by_fuel = error.abs().groupby(fuel_group).mean().sort_values()
    axes[1].barh(mae_by_fuel.index, mae_by_fuel.values, color="#55a868")
    for i, (f, v) in enumerate(mae_by_fuel.items()):
        axes[1].text(v, i, f"  {v:.1f}  (n={int((fuel_group == f).sum())})", va="center")
    axes[1].set(title="Average error by fuel type", xlabel="MAE (g/km)")
    axes[1].set_xlim(0, mae_by_fuel.max() * 1.35)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # The 10 biggest mistakes. Many rows are near-identical variants of one car, so we list each distinct car once.
    meta_full = ds.data[["Marque", GROUP_COL, "Désignation commerciale"]]
    worst = pd.DataFrame({
        "Brand": meta_full.loc[X_test.index, "Marque"],
        "Model": meta_full.loc[X_test.index, GROUP_COL],
        "Description": meta_full.loc[X_test.index, "Désignation commerciale"].str[:32],
        "Fuel": X_test["Carburant"], "kW": X_test["Puissance maximale (kW)"],
        "kg": X_test["masse vide euro max (kg)"],
        "Actual": y_test.round(0), "Predicted": pred.round(0), "Error": error.round(0)})
    worst_sorted = worst.loc[error.abs().sort_values(ascending=False).index]
    print("The 10 biggest mistakes (one line per distinct car):")
    print(worst_sorted.drop_duplicates(subset=["Model", "Description", "kW", "kg", "Actual", "Predicted"]).head(10).to_string())
    # HOW TO READ THIS: look at what is special about these cars (very powerful, very heavy, special technology, hybrids).
    # Our features do not describe everything about a car, so a few big misses are normal.


# =================================================================================================
# 4. From the number to the CO2 label (classes A-G): regression, then cut
# =================================================================================================
# The classes are the official French CO2 energy label for new cars (in force since 2006), exactly 7 classes:
#   Class        A          B        C        D        E        F        G
#   CO2 (g/km)   up to 100  101-120  121-140  141-160  161-200  201-250  above 250
# Source: Arrêté du 10 avril 2003 (annex I, as amended 10 Nov 2005), Légifrance. Not part of the dataset.
# Predicting class D when the truth is E is a small mistake; predicting A when the truth is G is a huge one.
# Plain "accuracy" treats both the same, so we also report "within one class" and the average distance in classes.
LABELS = ["A", "B", "C", "D", "E", "F", "G"]
LABEL_EDGES = [100, 120, 140, 160, 200, 250]       # upper limit (g/km) of A, B, C, D, E, F; G = above 250
LABEL_TEXT = ["<= 100", "101-120", "121-140", "141-160", "161-200", "201-250", "> 250"]


def co2_label(values):
    """CO2 in g/km -> class index 0..6 (A..G). The official label uses whole g/km, so we round first."""
    return np.searchsorted(LABEL_EDGES, np.round(np.asarray(values, dtype=float)), side="left")


def label_classes(ds, final_model, path):
    """Regression, then cut: use the final model's predicted number and look up its class."""
    y_train_cls, y_test_cls = co2_label(ds.y_train), co2_label(ds.y_test)

    counts = pd.DataFrame({"CO2 range (g/km)": LABEL_TEXT,
                           "test cars": np.bincount(y_test_cls, minlength=7)}, index=LABELS)
    counts["test %"] = (counts["test cars"] / counts["test cars"].sum() * 100).round(1)
    print(counts.to_string())

    pred_reg_cls = co2_label(final_model.predict(ds.X_test))

    def class_scores(pred_cls):
        distance = np.abs(pred_cls - y_test_cls)            # how many classes away from the truth
        return {"exact class (%)": (distance == 0).mean() * 100,
                "within 1 class (%)": (distance <= 1).mean() * 100,
                "average distance (classes)": distance.mean()}

    most_common = np.bincount(y_train_cls).argmax()          # the class that is most frequent in training
    class_table = pd.DataFrame({
        f"Always '{LABELS[most_common]}' (most common in training)": class_scores(np.full(len(y_test_cls), most_common)),
        "Regression model, then cut": class_scores(pred_reg_cls)}).T
    print()
    print(class_table.round(1).to_string())

    # Confusion matrix: rows = true class, columns = predicted class, numbers = test cars.
    # The diagonal = correct. Numbers next to the diagonal = one class off (small mistake).
    plt.figure(figsize=(6.5, 5.2))
    cm = confusion_matrix(y_test_cls, pred_reg_cls, labels=range(7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Regression model, then cut"); plt.xlabel("Predicted class"); plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
