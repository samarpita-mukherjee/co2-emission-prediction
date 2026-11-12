# Predicting the CO2 emissions of cars (Regression Approach)

Question: Can we predict the CO2 emissions (g/km) of a *new* car design from information an engineer already
knows before the car exists: fuel, power, mass, body type, gearbox, hybrid or not?

Data: 'data/cl_JUIN_2013-complet3.csv', a 2013 catalogue of French car CO2 labelling data
(44,850 rows, 26 columns; 44,192 rows after cleaning).

## Results

Final test on 4,533 cars of 90 car models that the model never saw in training:

| | MAE (g/km) | RMSE (g/km) | R2 | Predictions within 10 g/km |
|---|---|---|---|---|
| Always predict the average (baseline) | 50.74 | 56.15 | -1.55 | 5.2% |
| Ridge (straight-line model) | 17.30 | 21.92 | 0.61 | 36.4% |
| **Final model: LightGBM + sample weights** | **12.83** | **17.07** | **0.76** | **51.4%** |

Turned into the official French CO2 label (classes A to G), the same predictions give the exact class for 57.9% of
the cars and a class at most one step away for 98.5% (always guessing the most common class: 6.9% and 42.3%).

These numbers come from 'python main.py' (seeds are fixed, so a rerun gives the same numbers).

## How to run

'''
pip install -r requirements.txt
python main.py
'''

It takes about 8 to 10 minutes on one CPU. Everything is printed in the terminal, the figures are saved in
'outputs/figures/' and the trained model in 'outputs/co2_final_model.joblib'.
Tested with Python 3.12 (scikit-learn 1.6 or newer is needed).

## Project structure

'''
main.py                the only file we run: it calls the steps below in order
src/
  data.py              loading, cleaning, the two built features, the DROPS decisions (one reason per dropped column),
                       the split by car model, the checks
  modeling.py          encoders, the seven models, sample weights, grouped cross-validation, the feature ablation,
                       the tuning check, the final model (train, predict, save)
  evaluation.py        final test, model-comparison figure, error analysis, CO2 label classes A-G
  eda.py               exploration figures and tables 
data/                  the CSV file
notebooks/             the full analysis as a notebook, with explanations and outputs
outputs/figures/       figures written by main.py
'''

## Method description

1. Cleaning: Rows without CO2 and exact duplicates are removed.
2. Features: A feature may only be used if it is known before the car exists. So all columns measured in the same
   lab test as CO2 or computed from it (consumption, administrative power, pollutants) are dropped: mixed consumption
   alone brings the error down to about 1 g/km, which is exactly why it must not be used. Identifiers and names are
   dropped too. Seven features remain: fuel, power, mass, body type, gearbox type, number of gears, hybrid.
3. Split by car model: Train 80% / test 20%, but all variants of a car model go to the same side.
4. Feature check on the training part only: An ablation (train with and without each column) confirmed the
   seven features. The test set plays no role in any choice.
5. Seven models compared: by grouped 5-fold cross-validation (5 repeats, training set only): dummy, Ridge, HistGB,
   HistGB + weights, Random Forest, LightGBM, LightGBM + weights. LightGBM with sample weights is the best.
   The weights (1 / sqrt of the rows of a car model) stop three Mercedes van models, which are 73% of the rows, from
   dominating the training.
6. Tuning check: Eight random settings of LightGBM were tried. They are adopted only if they win in at least
   4 of 5 repeats on both error measures. They did not, so the starting settings are kept.
7. Final test: on the held-out cars (done once), then error analysis and the label classes A-G
   (regression first, then cut the predicted number into classes).

Tried and not used: one separate model per fuel type (it did not improve the results overall), a power-to-weight
feature (no consistent gain), and the brand as a feature (it lowers the error a little, but it identifies the maker,
not the design, and 87% of the rows are Mercedes-Benz).

## Why the split is done by car model

Variants of one car model are almost identical. With a random split by rows, siblings of the same model end up in
both train and test, so the model is partly tested on cars it has effectively already seen, and the score looks
better than it is. The split by car model tests what we actually want to know: how well the model predicts a design it
has never seen. A lower score from this split is therefore the honest one, and numbers from a random split are not
comparable with it.

## Limits

- The data are a 2013 catalogue of laboratory values. Mercedes-Benz is 87% of the rows, and three van models alone are 73%.
- The test set contains no vans, and one car model (CLASSE E) is about 63% of its rows. The test score describes
  ordinary cars of unseen models, and it is noisy.
- Hybrids, gas cars and special cars are predicted badly, because battery, engine and technology details are not
  among the features.
- The classes B, C and D are only 20 g/km wide, about the size of the typical error, so "within one class" is the
  fairer number for the label classes.

## Notebook

'notebooks/Liora_final_project_simplified.ipynb' contains the same analysis with explanations before every step,
all the exploration figures and the printed results. Open it with Jupyter from the 'notebooks/' folder.
