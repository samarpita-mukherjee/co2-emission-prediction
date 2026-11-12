"""
data.py 

Loading, cleaning, the two engineered features, the decision about which columns are dropped,
and the train/test split by car model.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ---- Paths (relative to the project folder, so the project runs on any machine) -------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cl_JUIN_2013-complet3.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# ---- Names and settings used by all the other files ------------------------------------------------
TARGET = "CO2 (g/km)"
GROUP_COL = "Modèle dossier"          # the car model: used for the split, NEVER a feature
SPLIT_SEED = 42                       # fixed seed -> the same train/test cars on every machine
VANS = ["VIANO", "VITO", "SPRINTER"]  # the 3 Mercedes van models = about 73% of all rows

# The 7 final features, in the groups the encoders need
CAT = ["Carburant", "Carrosserie", "boite_type"]                          # text -> encode
NUM = ["Puissance maximale (kW)", "masse vide euro max (kg)", "nb_rapports"]
BIN = ["Hybride"]                                                         # already 0/1

# Rule: a feature may only be used if an engineer knows it BEFORE the car exists.
# Every other column is dropped, each with a reason.
DROPS = {
    # ---- identifiers and names: they identify a car instead of describing it ----------------------
    "CNIT": "Type-approval code, essentially unique per row: lets a model memorise rows, nothing transfers to a new design.",
    "Type Variante Version (TVV)": "Variant/version code, almost unique per row: memorisation, not generalisation.",
    "Désignation commerciale": "Free-text commercial name (e.g. '320d xDrive Berline'): too many values and it repeats power, weight, fuel and gearbox.",
    "Modèle UTAC": "A new design is a model the data has not seen, so a model name cannot help; it is also nearly the same as 'Modèle dossier'.",
    "Modèle dossier": "Car-model name: used only as the GROUP KEY of the split, never as a feature.",
    "Date de mise à jour": "Date of the last update of the record: says nothing about the car.",

    # ---- leakage: CO2 in other units, or computed from CO2 ----------------------------------------
    "Consommation urbaine (l/100km)": "CO2 in other units (fuel burnt = CO2 emitted). A new design has no measured consumption.",
    "Consommation extra-urbaine (l/100km)": "CO2 in other units, same reason as the urban consumption.",
    "Consommation mixte (l/100km)": "CO2 in other units: with it the error nearly disappears, which is the proof that it leaks the answer.",
    "Puissance administrative": "Fiscal power, computed from CO2 and kW: leakage.",

    # ---- outputs of the same lab test as CO2 -------------------------------------------------------
    "CO type I (g/km)": "Pollutant measured in the same lab test as CO2: not available for a new design.",
    "NOX (g/km)": "Pollutant measured in the same lab test as CO2: not available for a new design.",
    "Particules (g/km)": "Pollutant measured in the same lab test as CO2: not available for a new design.",
    "HC (g/km)": "Pollutant measured in the same lab test; it is also empty for most rows, and which pollutant column is filled depends on the fuel.",
    "HC+NOX (g/km)": "Pollutant measured in the same lab test; the pattern of empty cells also reveals the fuel type.",

    # ---- redundant, or no benefit ------------------------------------------------------------------
    "masse vide euro min (kg)": "Same information as the maximum mass (very high correlation); only the maximum mass is kept.",
    "gamme": "Brand-specific marketing class: adding it makes the error worse (see the ablation table), because it lets the model learn brand habits.",
    "Marque": "Adding it lowers the error slightly (see the ablation table), but we leave it out on purpose: it identifies the maker, "
              "not the design of the car, and about 87% of the rows are Mercedes-Benz, so it would mostly mean 'Mercedes or not'.",
    "Champ V9": "Homologation reference text (regulation and Euro standard): not something chosen in the design, and messy raw text.",
    "Boîte de vitesse": "Replaced by the two features 'boite_type' and 'nb_rapports' (the information is kept).",
}


# =================================================================================================
# 1. Load and clean
# =================================================================================================
def load_data(path=DATA_PATH):
    """Read the raw CSV (separator ';', French accents -> latin1)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}\nPut cl_JUIN_2013-complet3.csv in the data/ folder.")
    return pd.read_csv(path, encoding="latin1", sep=";")


def clean_data(df):
    """Remove rows without CO2 and exact duplicate rows.

    Rows without the target cannot be used for learning. Duplicates would count the same car twice and could
    end up in both train and test. Duplicates are found by comparing ALL columns, so this must be done
    before any column is dropped."""
    print("Rows without CO2:", int(df[TARGET].isna().sum()))
    print("Exact duplicate rows:", int(df.dropna(subset=[TARGET]).duplicated().sum()))

    df_clean = df.dropna(subset=[TARGET])
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print("Rows before / after cleaning:", len(df), "/", len(df_clean))
    return df_clean


# =================================================================================================
# 2. Features and split
# =================================================================================================
def add_features(df_clean):
    """Build the two features that are not in the raw data (no learning involved, so this is safe before the split).

    - 'Hybride' is text 'oui'/'non' -> binary 0/1.
    - 'Boîte de vitesse' (e.g. 'M 6') packs 2 facts: transmission type + number of gears."""
    data = df_clean.copy()                                    # work on a copy
    data["Hybride"] = (data["Hybride"] == "oui").astype(int)

    parts = data["Boîte de vitesse"].str.split()
    data["boite_type"] = parts.str[0]                         # M, A, V, D ...
    data["nb_rapports"] = parts.str[1].astype(float)          # number of gears
    return data


def split_by_car_model(data):
    """80% / 20% split by car model ('Modèle dossier'). Returns the row POSITIONS of train and test.

    Variants of one car model are nearly identical. With a random row split, a model could memorise a car's
    siblings and look brilliant on the test, which says nothing about a truly NEW design. GroupShuffleSplit puts
    all variants of a car model on the same side, so the test cars are models never seen in training."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_SEED)
    train_pos, test_pos = next(gss.split(data, groups=data[GROUP_COL]))
    print("Train rows:", len(train_pos), "| test rows:", len(test_pos))
    print("Car models in train:", data[GROUP_COL].iloc[train_pos].nunique(),
          "| in test:", data[GROUP_COL].iloc[test_pos].nunique())
    return train_pos, test_pos


@dataclass
class Dataset:
    """Everything the other files need, in one object."""
    data: pd.DataFrame        # cleaned data with all columns (needed to describe results, e.g. car names)
    X: pd.DataFrame           # the 7 features, all rows
    y: pd.Series              # CO2, all rows
    groups: pd.Series         # car model of every row (group key)
    train_pos: np.ndarray     # row positions of the training rows
    test_pos: np.ndarray      # row positions of the test rows
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    g_train: pd.Series
    g_test: pd.Series
    n_variants: pd.Series     # number of rows of the car model of each row (used for the weights)
    is_merc: pd.Series        # True for Mercedes-Benz rows


def build_dataset(data, train_pos, test_pos):
    """Apply the drop decisions (DROPS) and cut X, y into train and test.
    X_test stays untouched until the final test: all model selection uses cross-validation on X_train."""
    missing_cols = [c for c in DROPS if c not in data.columns]
    if missing_cols:
        print("WARNING - columns listed in DROPS but not in the data:", missing_cols)

    df_model = data.drop(columns=list(DROPS))
    print("Remaining columns:", df_model.columns.tolist())

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]
    groups = data[GROUP_COL].copy()

    meta = data.loc[X.index, ["Marque", GROUP_COL]]            # brand and car model of every row (only to describe results)
    n_variants = meta.groupby(GROUP_COL)[GROUP_COL].transform("count")
    is_merc = (meta["Marque"] == "MERCEDES-BENZ")

    return Dataset(data=data, X=X, y=y, groups=groups, train_pos=train_pos, test_pos=test_pos,
                   X_train=X.iloc[train_pos], X_test=X.iloc[test_pos],
                   y_train=y.iloc[train_pos], y_test=y.iloc[test_pos],
                   g_train=groups.iloc[train_pos], g_test=groups.iloc[test_pos],
                   n_variants=n_variants, is_merc=is_merc)


# =================================================================================================
# 3. Checks (printed)
# =================================================================================================
def check_final_features(ds):
    """Final feature check: the features, missing values, categories, and fuel code vs Hybride."""
    X = ds.X
    print("Features:", X.columns.tolist())
    print("Missing values in the features:", int(X.isna().sum().sum()))
    for c in X.select_dtypes(exclude="number").columns:
        vc = X[c].value_counts()
        print(f"{c}: {len(vc)} levels, {int((vc < 50).sum())} of them with fewer than 50 rows")

    # Is 'Hybride' redundant with the fuel code (EH / GH ...)?
    print("\nCarburant x Hybride:\n", pd.crosstab(X["Carburant"], X["Hybride"]))


def check_split(ds):
    """Sanity check (stops if a car model is in both train and test) and train / test composition."""
    assert set(ds.g_train).isdisjoint(set(ds.g_test)), "Same car model in train AND test"
    print("OK: no car model is in both train and test.\n")

    tr, te = ds.X_train.index, ds.X_test.index
    comp = pd.DataFrame({
        "rows":         [len(tr), len(te)],
        "car models":   [ds.groups.loc[tr].nunique(), ds.groups.loc[te].nunique()],
        "% Mercedes":   [ds.is_merc.loc[i].mean() * 100 for i in (tr, te)],
        "% the 3 vans": [ds.groups.loc[i].isin(VANS).mean() * 100 for i in (tr, te)],
    }, index=["train", "test"]).round(1)
    print(comp.to_string())
    print("\nThe three largest car models of the test set (% of the test rows):")
    print((ds.g_test.value_counts(normalize=True).head(3) * 100).round(1))
    # HOW TO READ THIS: with SPLIT_SEED = 42 all three van models are in train, so the test set has no vans.
    # The test score therefore measures "ordinary cars of unseen models", a fair test for a NEW car design.
