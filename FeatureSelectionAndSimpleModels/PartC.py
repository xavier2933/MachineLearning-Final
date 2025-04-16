import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats import pearsonr
from xgboost import XGBRegressor


labelsPath = "RawProvidedData/MITInterview/scores.csv"
featuresPath = "features.csv"

# 1) Load features and labels
X = pd.read_csv(featuresPath)
y = pd.read_csv(labelsPath)
y.rename(columns={'Participant': 'id'}, inplace=True)

# 2) Filter and align X & y by 'id'
y = y[y["id"].isin(X["id"])]
X = X.sort_values(by="id").reset_index(drop=True)
y = y.sort_values(by="id").reset_index(drop=True)

# 3) Drop 'id' columns now
X = X.drop(columns=["id"])
y = y.drop(columns=["id"])

# 4) Pick features and target
feature_cols = X.columns.tolist()  # or specify explicitly, e.g. ["filler%", "speaker_balance", ...]
target_col = "Overall"
features = X[feature_cols]
labels = y[target_col]

# 4a) Normalize y with min–max normalization so that min → 0 and max → 1
min_y = labels.min()
max_y = labels.max()
labels_normalized = (labels - min_y) / (max_y - min_y)

# 5) Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.5,
    random_state=42
)

# 6) Set up hyperparameter tuning with GridSearchCV for Random Forest
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
param_grid = {
    "n_estimators": [20, 40, 60, 80, 100, 200],
    "max_depth": [None, 5, 10, 20]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best Params from GridSearchCV:", grid_search.best_params_)

# 7) Retrieve the best estimator and train it on the train set
best_rf = grid_search.best_estimator_

# 8) Make predictions (predictions will be in the normalized [0,1] range)
y_pred = best_rf.predict(X_test)

# 9) Evaluate the model in the normalized space

# (A) Pearson correlation (scale-invariant)
corr, p_value = pearsonr(y_test, y_pred)
print("Pearson’s r:", corr)
print("p-value:", p_value)

# (B) Mean Relative Error (since maximum of normalized y is 1, this is just mean absolute error)
rel_error = np.mean(np.abs(y_pred - y_test))
print("Mean Relative Error:", rel_error)

# 10) Compare predicted vs. actual in the normalized [0,1] scale
results_df = pd.DataFrame({
    "Actual Overall": y_test.values,
    "Predicted Overall": y_pred
})
print("\nPredicted vs. Actual:")
print(results_df)