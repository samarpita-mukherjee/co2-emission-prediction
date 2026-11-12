"""
eda.py

Nothing in the model depends on this file: it is doing the exploration of data.
Figures are saved to files.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data import TARGET

CO2 = TARGET
CONSO = "Consommation mixte (l/100km)"
KW = "Puissance maximale (kW)"
MASS = "masse vide euro min (kg)"          # exploration uses the minimum mass; the model uses the maximum mass (correlation ~0.99)
GEAR = "Boîte de vitesse"

# Fuels used in the scatter plots: diesel (GO), petrol (ES), petrol-hybrid (EH).
# Together they cover approx. 99.5% of the rows. The other 9 fuel codes would only clutter the legend.
MAIN_FUELS = ["GO", "ES", "EH"]


def target_and_consumption(df_clean, fig_dir):
    """Section 3: main numbers of the target, and Figure 2 (mixed consumption vs CO2).
    CO2 is close to symmetric, so no transformation is needed. Consumption lies on an almost perfect line with CO2:
    it is CO2 in other units, so it would be cheating to use it as a feature."""
    print(f"CO2 (g/km): mean {df_clean[CO2].mean():.0f}, median {df_clean[CO2].median():.0f}, "
          f"std {df_clean[CO2].std():.0f}, min {df_clean[CO2].min():.0f}, max {df_clean[CO2].max():.0f}, "
          f"skewness {df_clean[CO2].skew():.2f}")
    print(f"Share of the three main fuels (GO, ES, EH): {df_clean['Carburant'].isin(MAIN_FUELS).mean() * 100:.1f}%")
    print(f"Correlation consumption vs CO2: {df_clean[[CONSO, CO2]].corr().iloc[0, 1]:.2f}")

    df_main = df_clean[df_clean["Carburant"].isin(MAIN_FUELS)]
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_main, x=CONSO, y=CO2, hue="Carburant", hue_order=MAIN_FUELS, alpha=0.6)
    plt.title("Mixed Consumption vs. CO2 Emissions")
    plt.legend(title="Fuel")
    plt.savefig(fig_dir / "Fig2_consumption_vs_co2.png", dpi=300, bbox_inches="tight")
    plt.close()
    return df_main


def correlation_matrix(df_clean, fig_dir):
    """Section 4: correlation of the numeric columns with CO2 (HC, HC+NOX and particles are left out: mostly empty)."""
    df_test = df_clean.drop(columns=["HC (g/km)", "HC+NOX (g/km)", "Particules (g/km)"])
    num = df_test.select_dtypes("number")
    corr = num.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.savefig(fig_dir / "Fig2_1_correlation_matrix.png", dpi=300)
    plt.close()


def fuel_and_power(df_clean, df_main, fig_dir):
    """Section 5: Figure 3 (CO2 by fuel type) and Figure 4 (engine power vs CO2)."""
    plt.figure(figsize=(8, 4))
    top_fuels = ["GO", "ES", "EH", "ES/GN"]   # GO = diesel, ES = petrol, EH = petrol hybrid, ES/GN = petrol/natural gas
    df_filtered_fuel = df_clean[df_clean["Carburant"].isin(top_fuels)]
    sns.boxplot(x="Carburant", y=CO2, data=df_filtered_fuel, order=top_fuels, hue="Carburant", palette="Set2", legend=False)
    plt.title("CO2 Emissions Distribution by Fuel Type")
    plt.savefig(fig_dir / "Fig3_co2_fuel.png", bbox_inches="tight", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=KW, y=CO2, data=df_main, hue="Carburant", hue_order=MAIN_FUELS, alpha=0.3, legend=True)
    sns.regplot(x=KW, y=CO2, data=df_main, scatter=False, color="darkred")
    plt.legend(title="Fuel", loc="upper left")
    plt.title("Engine Power (kW) vs. CO2 Emissions (g/km)")
    plt.savefig(fig_dir / "Fig4_power_co2.png", bbox_inches="tight", dpi=300)
    plt.close()


def transmission(df_clean, fig_dir):
    """Section 6: Figure 5 - transmission type vs mixed fuel consumption."""
    plt.figure(figsize=(6, 4))
    top_trans = ["M 6", "A 5", "A 7", "M 5", "A 6"]
    df_filtered_trans = df_clean[df_clean[GEAR].isin(top_trans)]
    # errorbar=("ci", 95) draws the 95% confidence interval of each mean (the thin black line on each bar)
    sns.barplot(x=GEAR, y=CONSO, data=df_filtered_trans, order=top_trans,
                hue=GEAR, palette="muted", legend=False, errorbar=("ci", 95))
    plt.title("Average Mixed Fuel Consumption by Transmission Type")
    plt.ylabel("Mixed consumption (l/100km)")
    plt.savefig(fig_dir / "Fig5_trans_fuel.png", bbox_inches="tight", dpi=300)
    plt.close()


def brands(df_clean, fig_dir):
    """Section 7: Figure 6 - mean CO2 of the brands with at least 30 rows."""
    min_rows = 30   # minimum number of rows a brand needs to be included
    brand_counts = df_clean["Marque"].value_counts()
    selected_makes = brand_counts[brand_counts >= min_rows].index
    df_filtered_makes = df_clean[df_clean["Marque"].isin(selected_makes)]
    brand_co2 = (df_filtered_makes.groupby("Marque")[CO2]
                 .mean()
                 .sort_values(ascending=False)
                 .reset_index())

    plt.figure(figsize=(0.4 * len(brand_co2) + 2, 6))       # taller as the number of brands grows
    sns.barplot(x="Marque", y=CO2, data=brand_co2, palette="magma", hue="Marque", legend=False)
    plt.xticks(rotation=60, ha="right")                      # tilt the brand names so they don't overlap
    plt.xlabel("Brand")
    plt.ylabel("Mean CO2 (g/km)")
    plt.title(f"Mean CO2 of Brands with at Least {min_rows} Variants (Rows)")
    plt.tight_layout()
    plt.savefig(fig_dir / "Fig6_maker_meanCO2.png", bbox_inches="tight", dpi=300)
    plt.close()


def mass_power_spearman(df_clean):
    """Section 8: Spearman correlation of mass and power with CO2, for diesel and petrol, with and without Mercedes.
    Mercedes is 87% of the data (three van models alone are 73%); leaving it out shows which findings are robust."""
    fuels = {"GO": "Diesel (GO)", "ES": "Petrol (ES)"}      # the other fuel codes have too few rows for a reliable correlation
    rows = []
    for code, label in fuels.items():
        s = df_clean[df_clean["Carburant"] == code]                    # all rows of this fuel
        o = s[s["Marque"] != "MERCEDES-BENZ"]                           # the same rows without Mercedes
        for col, name in [(MASS, "mass"), (KW, "power")]:
            rows.append({"fuel": label, "feature": name,
                         "rows (all)": len(s), "rows (no Mercedes)": len(o),
                         "Spearman (all rows)": stats.spearmanr(s[col], s[CO2])[0],
                         "Spearman (no Mercedes)": stats.spearmanr(o[col], o[CO2])[0]})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def run_eda(df_clean, fig_dir):
    """Run the whole exploration ."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    df_main = target_and_consumption(df_clean, fig_dir)
    correlation_matrix(df_clean, fig_dir)
    fuel_and_power(df_clean, df_main, fig_dir)
    transmission(df_clean, fig_dir)
    brands(df_clean, fig_dir)
    print()
    mass_power_spearman(df_clean)
