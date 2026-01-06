# --- 0. Install catboost if needed ---
# !pip install catboost

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool

merged_df5 = pd.read_csv('masters_df2.csv')

# -----------------------------
# 1. Prepare data
# -----------------------------
# Your full dataframe
target_col = "demographic_vo2_max"  # or your HRV variable
exclude_cols = ["user_id", "date", "rmssd", "filtered_demographic_vo2_max",target_col]

# Keep only numeric predictors
X_raw = df.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=[np.number])
y_raw = df[target_col].astype(float)

# Drop rows with any NaNs in X or y
data = pd.concat([X_raw, y_raw], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# CatBoost can use Pool objects (not mandatory but recommended)
train_pool = Pool(X_train, y_train)
valid_pool = Pool(X_test, y_test)

# -----------------------------
# 3. Define and train CatBoost model
# -----------------------------
model = CatBoostRegressor(
    loss_function="RMSE",
    depth=6,
    learning_rate=0.05,
    n_estimators=2000,          # many trees + early stopping
    random_seed=42,
    eval_metric="RMSE",
    od_type="Iter",             # overfitting detector
    od_wait=50,                 # stop if no improvement in 50 iterations
    verbose=100                 # print every 100 iters
)

model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True
)

# -----------------------------
# 4. Evaluation: RMSE, MAE, R², Pearson r
# -----------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

y_test = y_test.reset_index(drop=True)
y_pred = pd.Series(model.predict(X_test)).reset_index(drop=True)

rmse = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
pearson_r, _ = pearsonr(y_test, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
print(f"Pearson r: {pearson_r:.3f}")


# -----------------------------
# 5. Simple diagnostic plots
# -----------------------------
# 5.1. y_true vs y_pred
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True RMSSD")
plt.ylabel("Predicted RMSSD")
plt.title("CatBoost: True vs Predicted RMSSD")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])  # y=x line
plt.grid(True)
plt.show()

# 5.2. Residuals plot
residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted RMSSD")
plt.ylabel("Residuals (True - Pred)")
plt.title("CatBoost: Residuals vs Predicted")
plt.grid(True)
plt.show()

# -----------------------------
# 6. Feature importance
# -----------------------------
importances = model.get_feature_importance(train_pool)
feat_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 20 features by importance:")
print(feat_importance.head(20))

# Optional: bar plot
plt.figure(figsize=(8, 6))
top_k = 20
plt.barh(
    feat_importance["feature"].head(top_k)[::-1],
    feat_importance["importance"].head(top_k)[::-1]
)
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance (Top 20)")
plt.tight_layout()
plt.show()
