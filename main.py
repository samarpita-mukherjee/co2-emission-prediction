"""
Predicting the CO2 emissions (g/km) of cars - regression approach.

Run the whole project with:   python main.py
Figures and the saved model go to the outputs/ folder.
"""
import time
import warnings

import matplotlib
matplotlib.use("Agg")                       # save figures to files, never open windows
warnings.filterwarnings("ignore")

from src import data, eda, evaluation, modeling


def step(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    t0 = time.time()
    data.FIG_DIR.mkdir(parents=True, exist_ok=True)

    step("CO2 emission prediction: Regression approach")
    # ---- 1. Load and clean --------------------------------------------------------------------------
    step("1. Load and clean the data")
    df = data.load_data()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    df_clean = data.clean_data(df)

    # ---- 2. Exploration (figures and tables, nothing here feeds the model) ----------------------------
    step("2. Exploration: figures are saved in outputs/figures")
    eda.run_eda(df_clean, data.FIG_DIR)

    # ---- 3. Features and the split by car model -------------------------------------------------------
    step("3. Build the features and split by car model (train 80% / test 20%)")
    featured = data.add_features(df_clean)
    train_pos, test_pos = data.split_by_car_model(featured)

    # ---- 4. Feature choice: ablation on the TRAINING part only ----------------------------------------
    step("4. Feature choice: ablation on the training part only")
    modeling.ablation_table(featured.iloc[train_pos])

    # ---- 5. Drop the other columns --------------------------------------------------------------
    step("5. Drop the other columns, final feature check, sanity check")
    ds = data.build_dataset(featured, train_pos, test_pos)
    data.check_final_features(ds)
    print()
    data.check_split(ds)

    # ---- 6. Compare seven models (cross-validation on the training set) -------------------------------
    step("6. Compare seven models: grouped 5-fold cross-validation, 5 repeats, training set only")
    cv_table = modeling.compare_models(ds)
    evaluation.plot_model_comparison(
        cv_table, "Model comparison (cross-validation on the training set): error and accuracy",
        data.FIG_DIR / "Fig9_model_comparison.png")
    best_name = cv_table.index[0]
    print("\nBest model of the comparison (lowest MAE_model_weighted):", best_name)
    if best_name != "LightGBM + weights":
        print("NOTE: the winner is not LightGBM + weights. The next steps assume it: check before going on.")

    # ---- 7. Can tuning improve the model? --------------------------------------------------------------
    step("7. Can tuning improve LightGBM? ")
    tuning = modeling.tune_lgbm(ds)

    # ---- 8. Final model: test set, mistakes, CO2 label ------------------------------------------------
    step("8. Final test on the held-out test set (done once)")
    final_model = modeling.train_final_model(ds, tuning["params"])
    test_table, test_pred = evaluation.final_test(ds, final_model)

    step("9. Where does the model make mistakes?")
    evaluation.error_analysis(ds, test_pred[evaluation.FINAL_NAME], data.FIG_DIR / "Fig10_error_analysis.png")

    step("10. From the number to the CO2 label (classes A-G)")
    evaluation.label_classes(ds, final_model, data.FIG_DIR / "Fig12_label_confusion.png")

    # ---- 11. Predict new cars and save the model --------------------------------------------------------
    step("11. Predict the CO2 of new cars, and save the model")
    modeling.predict_examples(final_model, ds.X_train.columns)
    modeling.save_model(final_model)

    step("Regression approach completed")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes. Figures: {data.FIG_DIR}")


if __name__ == "__main__":
    main()
