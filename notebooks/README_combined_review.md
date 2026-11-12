# Predicting the CO₂ Emissions of Cars: Regression and Classification

## Project overview

This project investigates whether vehicle CO₂ emissions can be predicted from technical characteristics available before a vehicle design is produced.

Two complementary machine learning approaches are explored:

- **Regression:** Predicting continuous CO₂ emissions in grams per kilometre (g/km).
- **Classification:** Predicting the official French CO₂ emissions category (A–G).

Both approaches use the same underlying French vehicle dataset and the same data-cleaning and feature-selection decisions.

The regression analysis establishes the data preparation methodology, including the decisions about which columns to retain and which to remove. The classification analysis builds on this prepared dataset and investigates the prediction of CO₂ emission categories.

The project also investigates whether separating minibuses from other passenger vehicles affects classification performance.

---

# Part 1: CO₂ Emissions Prediction (Regression Approach)

## Research question

Can we predict the CO₂ emissions (g/km) of a *new* car design from information an engineer already knows before the car exists: fuel, power, mass, body type, gearbox, hybrid or not?

**Data:** `data/cl_JUIN_2013-complet3.csv`, a 2013 catalogue of French car CO₂ labelling data (44,850 rows, 26 columns; 44,192 rows after cleaning).

## Results

Final test on 4,533 cars of 90 car models that the model never saw in training:

| Model | MAE (g/km) | RMSE (g/km) | R² | Predictions within 10 g/km |
|---|---:|---:|---:|---:|
| Always predict the average (baseline) | 50.74 | 56.15 | -1.55 | 5.2% |
| Ridge (straight-line model) | 17.30 | 21.92 | 0.61 | 36.4% |
| **Final model: LightGBM + sample weights** | **12.83** | **17.07** | **0.76** | **51.4%** |

Turned into the official French CO₂ label (classes A to G), the same predictions give the exact class for 57.9% of the cars and a class at most one step away for 98.5% (always guessing the most common class: 6.9% and 42.3%).

These numbers come from `python main.py` (seeds are fixed, so a rerun gives the same numbers).

## How to run

```bash
pip install -r requirements.txt
python main.py
```

It takes about 8 to 10 minutes on one CPU. Everything is printed in the terminal, the figures are saved in `outputs/figures/` and the trained model in `outputs/co2_final_model.joblib`.

Tested with Python 3.12 (scikit-learn 1.6 or newer is needed).

## Project structure

```text
main.py                the only file we run: it calls the steps below in order
src/
  data.py              loading, cleaning, the two built features, the DROPS decisions
                        (one reason per dropped column), the split by car model, the checks
  modeling.py          encoders, the seven models, sample weights, grouped cross-validation,
                        the feature ablation, the tuning check, the final model (train, predict, save)
  evaluation.py        final test, model-comparison figure, error analysis, CO₂ label classes A–G
  eda.py               exploration figures and tables
data/                  the CSV file
notebooks/             the full analysis as a notebook, with explanations and outputs
outputs/figures/       figures written by main.py
```

## Method description

1. **Cleaning:** Rows without CO₂ and exact duplicates are removed.
2. **Features:** A feature may only be used if it is known before the car exists. So all columns measured in the same lab test as CO₂ or computed from it (consumption, administrative power, pollutants) are dropped: mixed consumption alone brings the error down to about 1 g/km, which is exactly why it must not be used. Identifiers and names are dropped too. Seven features remain: fuel, power, mass, body type, gearbox type, number of gears, hybrid.
3. **Split by car model:** Train 80% / test 20%, but all variants of a car model go to the same side.
4. **Feature check on the training part only:** An ablation (train with and without each column) confirmed the seven features. The test set plays no role in any choice.
5. **Seven models compared:** By grouped 5-fold cross-validation (5 repeats, training set only): dummy, Ridge, HistGB, HistGB + weights, Random Forest, LightGBM, LightGBM + weights. LightGBM with sample weights is the best. The weights (1 / sqrt of the rows of a car model) stop three Mercedes van models, which are 73% of the rows, from dominating the training.
6. **Tuning check:** Eight random settings of LightGBM were tried. They are adopted only if they win in at least 4 of 5 repeats on both error measures. They did not, so the starting settings are kept.
7. **Final test:** On the held-out cars (done once), then error analysis and the label classes A–G (regression first, then cut the predicted number into classes).

Tried and not used: one separate model per fuel type (it did not improve the results overall), a power-to-weight feature (no consistent gain), and the brand as a feature (it lowers the error a little, but it identifies the maker, not the design, and 87% of the rows are Mercedes-Benz).

## Why the split is done by car model

Variants of one car model are almost identical. With a random split by rows, siblings of the same model end up in both train and test, so the model is partly tested on cars it has effectively already seen, and the score looks better than it is. The split by car model tests what we actually want to know: how well the model predicts a design it has never seen. A lower score from this split is therefore the honest one, and numbers from a random split are not comparable with it.

## Limits

- The data are a 2013 catalogue of laboratory values. Mercedes-Benz is 87% of the rows, and three van models alone are 73%.
- The test set contains no vans, and one car model (CLASSE E) is about 63% of its rows. The test score describes ordinary cars of unseen models, and it is noisy.
- Hybrids, gas cars and special cars are predicted badly, because battery, engine and technology details are not among the features.
- The classes B, C and D are only 20 g/km wide, about the size of the typical error, so "within one class" is the fairer number for the label classes.

## Regression notebook

`notebooks/Liora_final_project_simplified.ipynb` contains the same analysis with explanations before every step, all the exploration figures and the printed results. Open it with Jupyter from the `notebooks/` folder.

---

# Part 2: CO₂ Emissions Classification

## Classification overview

The classification analysis builds on the cleaned dataset and feature-selection decisions established in the regression part of this project.

While regression predicts the continuous CO₂ emissions value in g/km, classification predicts the vehicle's CO₂ emissions category directly.

The data cleaning process and the decisions regarding which columns to retain and which to remove are explained in the regression section above. These decisions are carried forward into the classification analysis rather than repeated here.

The objective is to compare different classification models and investigate whether separating minibuses from other passenger vehicles affects classification performance.

## Classification target

The classification target uses the official French CO₂ emissions categories A–G.

For the machine learning algorithms, these categories are represented numerically as 0–6. This numerical encoding does not change the underlying classification; it provides a numerical representation of the official categories for use as the target variable.

| Official French label | Numerical class | CO₂ emissions (g/km) |
|---|---:|---:|
| A | 0 | 0–100 |
| B | 1 | 100–120 |
| C | 2 | 120–140 |
| D | 3 | 140–160 |
| E | 4 | 160–200 |
| F | 5 | 200–250 |
| G | 6 | Above 250 |

The categories are ordinal: a higher numerical class represents higher CO₂ emissions.

## Classification objectives

- Build on the cleaned dataset and feature selection established in the regression analysis.
- Predict the official French CO₂ emissions category.
- Compare classification performance across all vehicles, non-minibuses and minibuses.
- Develop and evaluate Random Forest and XGBoost classifiers.
- Investigate which vehicle characteristics contribute most to model predictions.

## Modelling approach

Three vehicle groups are investigated:

- **All vehicles:** The complete dataset.
- **Non-minibus vehicles:** Passenger vehicles excluding minibuses.
- **Minibuses:** Minibuses analysed separately.

This allows the investigation of whether vehicle type and the composition of the dataset influence classification performance.

## Data splitting

`GroupShuffleSplit` is used to divide each dataset into training and test sets. The vehicle model identifier (`Modèle dossier`) is used as the grouping variable to reduce the risk of similar vehicle variants appearing in both sets.

For hyperparameter tuning, `GroupKFold` cross-validation is used to maintain group separation between training and validation folds.

## Classification models

Random Forest is used as the initial classification algorithm for all three vehicle groups.

Random Forest hyperparameter tuning is subsequently performed on the all-vehicle dataset.

XGBoost is then investigated as an alternative classification algorithm, also using hyperparameter tuning and grouped cross-validation.

## Comparison of the three vehicle groups

An initial Random Forest classifier was trained and evaluated separately on all vehicles, non-minibus vehicles and minibuses.

| Metric | All vehicles | Non-minibuses | Minibuses |
|---|---:|---:|---:|
| Accuracy | 53.50% | 58.83% | 85.30% |
| Weighted precision | 58.18% | 59.09% | 86.28% |
| Weighted recall | 53.50% | 58.83% | 85.30% |
| Weighted F1 | 52.16% | 58.74% | 85.42% |

## Interpreting the minibus results

The minibus model achieved considerably higher classification scores than the models trained on all vehicles and non-minibus vehicles.

However, this result needs to be interpreted in the context of the dataset. Although minibuses represent a substantial proportion of the total vehicle records, they comprise relatively few distinct vehicle models. Consequently, a small number of models account for a disproportionately large share of the minibus data.

This limited model diversity makes the minibus classification task more concentrated around a smaller number of vehicle designs. The high accuracy should therefore not necessarily be interpreted as evidence that the model generalises better to previously unseen minibus designs.

The use of `GroupShuffleSplit` helps address this issue by ensuring that records belonging to the same vehicle model are kept together during the train/test split.

The non-minibus model also performed better than the all-vehicle model on these initial results. This suggests that vehicle composition and the distribution of CO₂ classes influence classification performance.

## Model evaluation

Model performance is assessed using:

- Accuracy
- Macro F1-score
- Weighted F1-score
- Confusion matrices
- Mean Absolute Error (MAE)
- Percentage of predictions within one CO₂ class of the actual class

Macro F1 is particularly important because the seven CO₂ classes are not equally represented in the dataset.

MAE and the within-one-class measure are also included because the target categories have a natural order: predicting class 3 when the actual class is 2 is a smaller error than predicting class 6.

## Results: tuned all-vehicle models

The following table compares the tuned Random Forest and XGBoost models trained on all vehicles.

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 57.16% | 62.67% |
| Macro F1 | 58.19% | 62.51% |
| Weighted F1 | 55.87% | 63.00% |
| Within one class | 97.60% | 97.79% |
| MAE | 0.452 | 0.395 |

XGBoost achieves better overall performance across most evaluation metrics, while Random Forest performs marginally better on the within-one-class measure.

## Feature importance and error analysis

Feature importance is investigated to identify which vehicle characteristics contribute most to the Random Forest predictions.

Vehicle mass, engine power, gearbox type and body type are among the important features.

Further analysis examines the difficulty of distinguishing between CO₂ classes 2 and 3. The distributions of vehicle mass and engine power show considerable overlap between these classes, helping to explain some of the classification errors.

## Classification conclusion

The classification analysis complements the regression approach by investigating the prediction of CO₂ emission categories rather than exact emissions values.

XGBoost provides the strongest overall results for the all-vehicle classification task, although performance varies across individual classes.

The comparison between all vehicles, non-minibuses and minibuses provides an additional perspective on the effect of vehicle composition on classification performance.

Together, the regression and classification approaches provide two complementary ways of predicting vehicle CO₂ emissions.

---

## Technologies used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- LightGBM
- XGBoost
- Jupyter Notebook
- Joblib

## Reproducibility

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

For the regression analysis, run:

```bash
python main.py
```

For the classification analysis, open the classification notebook in Jupyter and run the cells in order.

The original dataset should be placed in the `data/` directory, subject to its redistribution licence.

## Notebooks

The repository contains the notebooks supporting the regression and classification analyses, including the exploratory work, explanations, modelling steps and evaluation results.
