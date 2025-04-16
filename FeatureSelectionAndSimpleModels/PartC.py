import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

labelsPath = "RawProvidedData/MITInterview/scores.csv"
featuresPath = "features.csv"

X = pd.read_csv(featuresPath)
y = pd.read_csv(labelsPath)
y.rename(columns={'Participant': 'id'}, inplace=True)

# 1) Drop rows from y whose 'id' isn't in X
y = y[y["id"].isin(X["id"])]

# 2) Sort both by 'id' in ascending order
X = X.sort_values(by="id").reset_index(drop=True)
y = y.sort_values(by="id").reset_index(drop=True)

# 3) Drop 'id' column from X and 'y' column from y
X = X.drop(columns=["id"])
y = y.drop(columns=["id"])

# 4) choose feature_cols and target_col
feature_cols = ["filler%", "speaker_balance"]
target_col = "Overall"
features = X[feature_cols]
labels = y[target_col]

# 5) Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.2, 
    random_state=42
)

# 6) Train a Random Forest model
rf = RandomForestRegressor(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 7) Make predictions
y_pred = rf.predict(X_test)

# 8) Evaluate the model
#    (A) Pearson correlation
corr, p_value = pearsonr(y_test, y_pred)
print("Pearson’s r:", corr)
print("p-value:", p_value)

#    (B) Relative Error example
max_y = y_test.max()
rel_error = np.mean(np.abs(y_pred - y_test) / max_y)
print("Mean Relative Error:", rel_error)

# Compare predicted vs. actual
results_df = pd.DataFrame({
    "Actual Overall": y_test.values,
    "Predicted Overall": y_pred
})
print("\nPredicted vs. Actual")
print(results_df.to_string(index=False))