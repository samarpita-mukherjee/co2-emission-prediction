"""
modeling.py 

Encoders and model builders, sample weights, grouped cross-validation, the feature ablation,
the comparison of seven models, the tuning check, and the final model (train, predict, save).
All model selection uses ONLY the training rows (X_train); the test rows are not touched here.
"""
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import BIN, CAT, GROUP_COL, NUM, OUTPUT_DIR, TARGET

# A cross-validation result depends on WHICH car models happen to land in which fold. To avoid a lucky or
# unlucky split we repeat the cross-validation with 5 different random fold assignments and average them.
# The same seed always gives the same folds, on every machine.
SEEDS = [0, 1, 2, 3, 4]

START_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)    # starting settings of LightGBM
WEIGHT_POWER = 0.5          # training weight of a row = 1 / (rows of its car model) ** 0.5

MODEL_PATH = OUTPUT_DIR / "co2_final_model.joblib"


# =================================================================================================
# 1. Encoders and model builders (two pipelines: linear vs tree)
# =================================================================================================
def linear_preprocessor():
    """One-hot + scaling: for Ridge, Random Forest and LightGBM."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=50, sparse_output=False), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("bin", "passthrough", BIN),
    ])


def tree_preprocessor():
    """Integer codes (no scaling): for HistGradientBoosting, which treats them as categories."""
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ("num", "passthrough", NUM),          # HistGB handles NaN natively
        ("bin", "passthrough", BIN),
    ])


# The last pipeline step is always called "m": this is how the sample weights are passed (m__sample_weight).
def make_dummy():
    return Pipeline([("prep", tree_preprocessor()), ("m", DummyRegressor(strategy="mean"))])


def make_ridge():
    return Pipeline([("prep", linear_preprocessor()), ("m", Ridge(alpha=1.0))])


def make_hgb():
    mask = [True] * len(CAT) + [False] * (len(NUM) + len(BIN))   # first block = categorical
    return Pipeline([("prep", tree_preprocessor()),
                     ("m", HistGradientBoostingRegressor(categorical_features=mask,
                                                         max_iter=300, random_state=0))])


def make_rf():
    # Random forest cannot use integer codes as categories -> use the one-hot pipeline
    return Pipeline([("prep", linear_preprocessor()),
                     ("m", RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=0))])


def make_lgbm_tuned(params):
    return Pipeline([("prep", linear_preprocessor()),
                     ("m", lgb.LGBMRegressor(random_state=0, verbose=-1, **params))])


def make_lgbm():
    return make_lgbm_tuned(START_PARAMS)


# =================================================================================================
# 2. Sample weights, grouped cross-validation, metrics
# =================================================================================================
# WHY: 73% of the rows are 3 van models. A plain MAE is therefore mostly "how well do we predict vans".
# To see how well the model generalises to OTHER cars we also report:
#   - MAE_model_weighted : every car MODEL counts equally (a model with 2,800 rows no longer outweighs a model with 3 rows)
#   - MAE_Mercedes / MAE_other : error split by brand group
def sample_weights(ds, idx, power):
    """Training weights 1 / n_variants**power. power=0 -> no weighting, 0.5 -> 'soft', 1 -> full balancing."""
    return (1.0 / ds.n_variants.loc[idx] ** power).values


def cv_oof(factory, ds, weight_power=0.0, n_splits=5, seed=0):
    """Out-of-fold predictions on the TRAINING rows with GroupKFold: every row is predicted by a model that never
    saw that car model during training (same logic as the final train/test split)."""
    Xd, yd, gd = ds.X_train, ds.y_train, ds.g_train
    pred = pd.Series(np.nan, index=Xd.index)
    for tr, te in GroupKFold(n_splits, shuffle=True, random_state=seed).split(Xd, yd, gd):
        model = factory()
        if weight_power > 0:   # 'm__sample_weight' = pass the weights to the last pipeline step named 'm'
            model.fit(Xd.iloc[tr], yd.iloc[tr], m__sample_weight=sample_weights(ds, Xd.index[tr], weight_power))
        else:
            model.fit(Xd.iloc[tr], yd.iloc[tr])
        pred.iloc[te] = model.predict(Xd.iloc[te])
    return pred


def metrics(ds, y_true, pred, idx):
    abs_err = (y_true - pred).abs()
    merc = ds.is_merc.loc[idx].values
    w_model = 1.0 / ds.n_variants                       # equal weight per car model
    return {"MAE": abs_err.mean(),
            "MAE_model_weighted": np.average(abs_err, weights=w_model.loc[idx]),
            "MAE_Mercedes": abs_err[merc].mean(),
            "MAE_other": abs_err[~merc].mean(),
            "R2": r2_score(y_true, pred)}


# =================================================================================================
# 3. Feature ablation (on the TRAINING part only)
# =================================================================================================
def run_ablation(d, cats, nums, seeds=(0, 1, 2)):
    """Mean and std of the MAE (g/km), repeated grouped 5-fold cross-validation on the rows of d."""
    X_ = d[cats + nums].copy()
    for c in cats:
        X_[c] = X_[c].astype("category").cat.codes
    mask = [True] * len(cats) + [False] * len(nums)
    y_ = d[TARGET].values
    g_ = d[GROUP_COL]
    uniq = np.unique(g_)
    maes = []
    for s in seeds:
        fold_of = dict(zip(uniq, np.random.RandomState(s).randint(0, 5, len(uniq))))
        fold = g_.map(fold_of).values
        pred = np.zeros(len(d))
        for k in range(5):
            fit, check = fold != k, fold == k
            m = HistGradientBoostingRegressor(categorical_features=mask, max_iter=200, random_state=0)
            pred[check] = m.fit(X_[fit], y_[fit]).predict(X_[check])
        maes.append(mean_absolute_error(y_, pred))
    return float(np.mean(maes)), float(np.std(maes))


def ablation_table(data_train):
    """Train the same model with and without a column and compare the error (takes about 2-3 minutes).
    data_train = the training rows with ALL columns. A change smaller than about the +- value is noise.
    Columns that are not known before the car exists (consumption, administrative power, NOX) may lower the
    error, but they stay out whatever their score."""
    core_cat = ["Carburant", "boite_type", "Carrosserie"]
    core_num = ["Hybride", "Puissance maximale (kW)", "nb_rapports", "masse vide euro max (kg)"]

    experiments = {"CORE (the 7 kept features)": (core_cat, core_num)}
    for f in core_cat:
        experiments[f"remove {f}"] = ([c for c in core_cat if c != f], core_num)
    for f in core_num:
        experiments[f"remove {f}"] = (core_cat, [c for c in core_num if c != f])
    for extra, kind in [("Marque", "cat"), ("gamme", "cat"), ("masse vide euro min (kg)", "num"),
                        ("Consommation mixte (l/100km)", "num"), ("Puissance administrative", "num"),
                        ("NOX (g/km)", "num")]:
        experiments[f"add {extra}"] = (core_cat + ([extra] if kind == "cat" else []),
                                       core_num + ([extra] if kind == "num" else []))

    results = []
    for name, (cats, nums) in experiments.items():
        mae, std = run_ablation(data_train, cats, nums)
        results.append({"experiment": name, "MAE (g/km)": mae, "+- (noise)": std})
    ablation = pd.DataFrame(results).set_index("experiment")
    ablation["change vs CORE"] = ablation["MAE (g/km)"] - ablation.loc["CORE (the 7 kept features)", "MAE (g/km)"]
    print("Ablation on the training part only (lower MAE = better; a change smaller than about the +- value is noise):")
    print(ablation.round(2).to_string())
    return ablation


# =================================================================================================
# 4. Comparison of seven models (grouped 5-fold CV on the TRAINING set, 5 repeats)
# =================================================================================================
def compare_models(ds):
    """Dummy (always the training mean) is the floor every real model must beat. Sample weighting (power 0.5)
    gives rarer car models more influence, which should reduce the van bias."""
    candidates = [("Dummy (mean)",        make_dummy, 0.0),
                  ("Ridge",               make_ridge, 0.0),
                  ("HistGB",              make_hgb,   0.0),
                  ("HistGB + weights",    make_hgb,   WEIGHT_POWER),
                  ("Random Forest",       make_rf,    0.0),
                  ("LightGBM",            make_lgbm,  0.0),
                  ("LightGBM + weights",  make_lgbm,  WEIGHT_POWER)]

    rows = {}
    for name, factory, wp in candidates:
        t0 = time.time()
        per_repeat = []                                 # one set of scores for each random fold assignment
        for seed in SEEDS:
            oof = cv_oof(factory, ds, weight_power=wp, seed=seed)
            per_repeat.append(metrics(ds, ds.y_train, oof, ds.X_train.index))
        rows[name] = pd.DataFrame(per_repeat).mean().to_dict()          # average of the 5 repeats
        print(f"{name:20s} done in {time.time() - t0:5.1f}s")

    cv_table = pd.DataFrame(rows).T.sort_values("MAE_model_weighted")
    print("\nCross-validated results (g/km, lower MAE = better; average of 5 repeats):")
    print(cv_table.round(2).to_string())
    # HOW TO READ THIS TABLE
    # * Dummy shows the cost of knowing nothing. Every other row should be far below it.
    # * 'MAE' is dominated by vans. 'MAE_model_weighted' and 'MAE_other' show how the model does on the many other
    #   car models: they are always HIGHER than plain MAE, i.e. plain MAE is optimistic.
    # * Compare "X" with "X + weights": if MAE_model_weighted drops (and MAE does not get worse), the weighting is useful.
    return cv_table


# =================================================================================================
# 5. Can tuning improve the model?
# =================================================================================================
SEARCH_SPACE = dict(
    n_estimators=[200, 400, 600],        # number of small trees: more = more detail, slower, can over-fit
    learning_rate=[0.03, 0.05, 0.1],     # how big each tree's correction is: small = careful but needs more trees
    num_leaves=[15, 31, 63],             # how complex each tree is: bigger = can over-fit
    min_child_samples=[10, 20, 50],      # minimum cars per leaf: bigger = smoother, less memorising
    subsample=[0.7, 1.0],                # share of rows each tree sees (0.7 = randomness against over-fitting)
    colsample_bytree=[0.7, 1.0],         # share of columns each tree sees
    reg_lambda=[0, 1, 10])               # penalty on extreme values: bigger = more cautious


def cv_scores(ds, make_model):
    """One row per repeat (5 random fold assignments), two columns (lower = better):
       MAE       = average error over all rows (dominated by the vans)
       MAE_equal = average error when every CAR MODEL counts the same (fairer for ordinary cars)"""
    rows = []
    for seed in SEEDS:
        oof = cv_oof(make_model, ds, weight_power=WEIGHT_POWER, seed=seed)
        err = (ds.y_train - oof).abs()
        rows.append({"MAE": err.mean(),
                     "MAE_equal": np.average(err, weights=1.0 / ds.n_variants.loc[ds.X_train.index])})
    return pd.DataFrame(rows)


def tune_lgbm(ds):
    """Try 8 random settings of LightGBM (several minutes). The tuned settings are ADOPTED only if they beat the
    starting settings in at least 4 of the 5 repeats on BOTH MAE and MAE_equal. Returns the settings to use."""
    start_scores = cv_scores(ds, lambda: make_lgbm_tuned(START_PARAMS))       # starting settings, 5 repeats

    tried, all_scores = [], []
    t0 = time.time()
    for params in ParameterSampler(SEARCH_SPACE, n_iter=8, random_state=0):
        full_params = {**params, "subsample_freq": 1 if params["subsample"] < 1 else 0}   # LightGBM needs this to use subsample
        all_scores.append(cv_scores(ds, lambda: make_lgbm_tuned(full_params)))            # 5 repeats each
        tried.append(full_params)
    print(f"8 combinations tested (5 repeats each) in {time.time() - t0:.0f}s")

    best_i = int(np.argmin([s["MAE_equal"].mean() for s in all_scores]))
    best_scores = all_scores[best_i]

    wins_equal = int((best_scores["MAE_equal"].values < start_scores["MAE_equal"].values).sum())
    wins_plain = int((best_scores["MAE"].values < start_scores["MAE"].values).sum())
    use_tuned = wins_equal >= 4 and wins_plain >= 4
    params_to_use = tried[best_i] if use_tuned else START_PARAMS

    comparison = pd.DataFrame(
        {"MAE": [start_scores["MAE"].mean(), best_scores["MAE"].mean()],
         "MAE_equal": [start_scores["MAE_equal"].mean(), best_scores["MAE_equal"].mean()],
         "repeats won on MAE (of 5)": ["-", wins_plain],
         "repeats won on MAE_equal (of 5)": ["-", wins_equal]},
        index=["starting settings", "best tuned settings"])
    print("\nBest tuned settings:", tried[best_i])
    print("\nCross-validation on the training set, average of 5 repeats:")
    print(comparison.round(2).to_string())
    print("\nAdopt the tuned settings?", use_tuned)
    print("Settings used from now on:", params_to_use)
    return {"params": params_to_use, "use_tuned": use_tuned, "comparison": comparison}


# =================================================================================================
# 6. Final model: train, predict a new car, save
# =================================================================================================
def train_final_model(ds, params):
    """LightGBM with sample weights, fitted on the training rows only."""
    final_model = make_lgbm_tuned(params)
    final_model.fit(ds.X_train, ds.y_train, m__sample_weight=sample_weights(ds, ds.X_train.index, WEIGHT_POWER))
    return final_model


def predict_co2(model, columns, carburant, power_kw, mass_kg, carrosserie, boite_type, nb_rapports, hybride=0):
    """Predicted CO2 (g/km) of one car. carburant: GO = diesel, ES = petrol, EH = petrol hybrid ...
    boite_type: M = manual, A = automatic ...  'columns' = the feature columns in training order."""
    car = pd.DataFrame([{"Carburant": carburant, "Hybride": hybride, "Puissance maximale (kW)": power_kw,
                         "masse vide euro max (kg)": mass_kg, "Carrosserie": carrosserie,
                         "boite_type": boite_type, "nb_rapports": nb_rapports}])
    return float(model.predict(car[columns])[0])        # same columns, same order as in training


EXAMPLE_CARS = [
    ("Small petrol hatchback", dict(carburant="ES", power_kw=66,  mass_kg=1100, carrosserie="BERLINE", boite_type="M", nb_rapports=5)),
    ("Diesel estate car",      dict(carburant="GO", power_kw=110, mass_kg=1650, carrosserie="BREAK",   boite_type="A", nb_rapports=7)),
    ("Large petrol SUV",       dict(carburant="ES", power_kw=300, mass_kg=2300, carrosserie="TS TERRAINS/CHEMINS", boite_type="A", nb_rapports=8)),
]


def predict_examples(model, columns):
    """Plausibility check: the small petrol car should come out lowest and the large SUV highest."""
    for label, car in EXAMPLE_CARS:
        print(f"{label:25s} -> predicted CO2: {predict_co2(model, columns, **car):6.1f} g/km")


def save_model(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)                            # reload later with joblib.load(path)
    print(f"Saved: {path}")
