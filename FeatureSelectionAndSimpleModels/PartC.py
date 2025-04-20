import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
import matplotlib.pyplot as plt

# ----------------------------- Load Data -----------------------------

labelsPath = "RawProvidedData/MITInterview/scores.csv"
featuresPath = "features.csv"

X = pd.read_csv(featuresPath)
y = pd.read_csv(labelsPath)
y.rename(columns={'Participant': 'id'}, inplace=True)

# Align by ID
y = y[y["id"].isin(X["id"])]

# 2) Sort both by 'id' in ascending order
X = X.sort_values(by="id").reset_index(drop=True)
y = y.sort_values(by="id").reset_index(drop=True)

# Drop ID columns
X = X.drop(columns=["id"])
y = y.drop(columns=["id"])

# ----------------------------- Normalize y -----------------------------

target_col = "Overall"
labels = y[target_col]
min_y = labels.min()
max_y = labels.max()
labels_normalized = (labels - min_y) / (max_y - min_y)

# ----------------------------- Train/Test Split -----------------------------

# Use normalized labels
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels_normalized,
    test_size=0.5,
    random_state=42
)

# ----------------------------- Normalize X -----------------------------

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# ----------------------------- XGBoost Model -----------------------------
print("\n==== XGBoost Model ====")
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)

param_grid = {
    "n_estimators": [20, 40, 60, 80, 100, 200],
    "max_depth": [5, 10, 20, 30]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# ----------------------------- Predict and Evaluate -----------------------------

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Pearson correlation
corr, p_value = pearsonr(y_test, y_pred)
print("Pearson’s r:", corr)
print("p-value:", p_value)

# Mean Relative Error (in normalized space, max_y = 1, so it's just MAE)
rel_error = np.mean(np.abs(y_pred - y_test))
print("Mean Relative Error:", rel_error)

# Show prediction comparison
results_df = pd.DataFrame({
    "Actual Overall (Normalized)": y_test.values,
    "Predicted Overall (Normalized)": y_pred
})
print("\nPredicted vs. Actual:")
print(results_df.head(10).to_string(index=False))

# ----------------------------- Deep Learning  Model -----------------------------
print("\n==== Deep Learning Model ====")

model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Keep predictions in [0, 1]
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005),
    loss='mean_squared_error'
)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)
# ----------------------------- Predict and Evaluate -----------------------------
y_pred_dl = model.predict(X_test_scaled).flatten()

corr_dl, pval_dl = pearsonr(y_test, y_pred_dl)
rel_error_dl = np.mean(np.abs(y_pred_dl - y_test))

print("Pearson’s r:", corr_dl)
print("p-value:", pval_dl)
print("Mean Relative Error:", rel_error_dl)

results_dl = pd.DataFrame({
    "Actual Overall (Normalized)": y_test.values,
    "Predicted Overall (Deep Learning)": y_pred_dl
})
print("\nPredicted vs. Actual (Deep Learning):")
print(results_dl.head(10).to_string(index=False))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
plt.savefig("training_vs_validation_loss.png")