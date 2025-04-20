#!/usr/bin/env python3
import os                                 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from interpret.glassbox import ExplainableBoostingRegressor
import plotly.io as pio

# ----------------------------- Load Data -----------------------------
labelsPath   = "RawProvidedData/MITInterview/scores.csv"
featuresPath = "features.csv"

X = pd.read_csv(featuresPath)
y = pd.read_csv(labelsPath).rename(columns={'Participant': 'id'})

# Align by ID, sort, drop id
y = y[y["id"].isin(X["id"])]
X = X.sort_values("id").reset_index(drop=True)
y = y.sort_values("id").reset_index(drop=True)
X = X.drop(columns=["id"])
y = y.drop(columns=["id"])

# ----------------------------- Normalize y -----------------------------
labels = y["Overall"]
labels_normalized = (labels - labels.min()) / (labels.max() - labels.min())

# ----------------------------- Train/Test Split -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_normalized, test_size=0.5, random_state=42
)

# ----------------------------- Normalize X -----------------------------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ----------------------------- Explainable Boosting Machine -----------------------------
print("\n==== Explainable Boosting Machine ====")
ebm = ExplainableBoostingRegressor(
        random_state=42,
        interactions=10,
        max_bins=256,
        learning_rate=0.01,
        outer_bags=8
)
ebm.fit(X_train_scaled, y_train)

# ----------------------------- Predict & Evaluate -----------------------------
y_pred = ebm.predict(X_test_scaled)
corr, pval = pearsonr(y_test, y_pred)
rel_err = np.mean(np.abs(y_pred - y_test))

print(f"Pearson’s r: {corr:.4f}")
print(f"p‑value:     {pval:.4g}")
print(f"Mean RelErr: {rel_err:.4f}")

# ----------------------------- Explainability plots -----------------------------
PLOTS_DIR = "ebm_plots"   
os.makedirs(PLOTS_DIR, exist_ok=True)

pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_scale  = 2

global_exp = ebm.explain_global()
for idx, name in enumerate(global_exp.data()['names']):
    fig = global_exp.visualize(idx)
    safe = name.replace(" ", "_").replace(":", "-").replace("/", "-")
    outfile = f"{PLOTS_DIR}/{idx:02d}_{safe}.png"
    fig.write_image(outfile)
    print("saved", outfile)

print(f"\nAll plots written to ./{PLOTS_DIR}")
