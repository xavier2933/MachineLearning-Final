#!/usr/bin/env python3
# ------------------------------------------------------------------
#  Progressive feature‑dropping for BOTH XGBoost and Keras models
# ------------------------------------------------------------------
import os, warnings, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # force CPU for TF
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from xgboost        import XGBRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers

# ----------------------------- Config -----------------------------
LABELS_PATH   = "RawProvidedData/MITInterview/scores.csv"
FEATURES_PATH = "features.csv"

BASE_DROP = ["id"]
DROP_ORDER = [
    "interviewer_length", "interviewee_length", "cluster",
    "total_word_count", "similarity",
    "minimum_sentence_sentiment", "maximum_sentence_sentiment",
    "word_length_2", "word_length_3", "word_length_4", "word_length_5",
    "speaker_balance",
]

PARAM_GRID = {                         # XGB grid
    "n_estimators": [20, 60, 120],
    "max_depth":    [5, 10, 20]
}

EPOCHS      = 120                      # max epochs for DL
BATCH_SIZE  = 8
PATIENCE    = 15

# ----------------------------- Load once -----------------------------
X_full = pd.read_csv(FEATURES_PATH)
y_df   = pd.read_csv(LABELS_PATH).rename(columns={'Participant':'id'})

y_df   = y_df[y_df["id"].isin(X_full["id"])]
X_full = X_full.sort_values("id").reset_index(drop=True)
y_df   = y_df.sort_values("id").reset_index(drop=True)

labels = y_df["Overall"]
y_norm = (labels - labels.min()) / (labels.max() - labels.min())

# ----------------------------- Loop -----------------------------
records, dropped = [], BASE_DROP.copy()

for rnd in range(len(DROP_ORDER) + 1):
    print(f"\n=== ROUND {rnd}: dropping {dropped} ===")
    t0 = time.perf_counter()

    # 1 Drop columns & split
    X = X_full.drop(columns=dropped, errors="ignore")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_norm, test_size=0.5, random_state=42)

    # 2 Scale
    scaler = MinMaxScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    # ---------- XGBoost ----------
    xgb_grid = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42),
        PARAM_GRID, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1
    ).fit(X_tr_s, y_tr)

    xgb_pred = xgb_grid.best_estimator_.predict(X_te_s)
    r_xgb,  _ = pearsonr(y_te, xgb_pred)
    mae_xgb    = np.mean(np.abs(xgb_pred - y_te))

    # ---------- Deep‑Learning ----------
    tf.keras.backend.clear_session()
    model = models.Sequential([
        layers.Input((X_tr_s.shape[1],)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.3),
        layers.Dense(64,  activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.2),
        layers.Dense(32,  activation='relu'),
        layers.Dense(1,   activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(5e-4), loss='mse')

    model.fit(
        X_tr_s, y_tr,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)],
        verbose=0
    )
    dl_pred = model.predict(X_te_s, verbose=0).flatten()
    r_dl, _   = pearsonr(y_te, dl_pred)
    mae_dl    = np.mean(np.abs(dl_pred - y_te))

    dt = time.perf_counter() - t0
    print(f"  XGB  r={r_xgb:.3f}  MAE={mae_xgb:.3f}")
    print(f"  DL   r={r_dl :.3f}  MAE={mae_dl :.3f}  ({dt/60:.1f} min)")

    records.append({
        "round": rnd,
        "dropped": dropped.copy(),
        "n_features": X.shape[1],
        "xgb_r": r_xgb, "xgb_MAE": mae_xgb,
        "dl_r":  r_dl,  "dl_MAE":  mae_dl,
    })

    if rnd < len(DROP_ORDER):          # add next column
        dropped.append(DROP_ORDER[rnd])

# ----------------------------- Summary -----------------------------
summary = pd.DataFrame(records)
pd.set_option("display.max_colwidth", None)
print("\n===== SUMMARY =====")
print(summary[["round", "n_features", "xgb_r", "xgb_MAE", "dl_r", "dl_MAE"]])

summary.to_csv("drop_experiment_summary.csv", index=False)
print("\nSaved → drop_experiment_summary.csv")
