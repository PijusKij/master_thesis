import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool


# -----------------------------
# 0) Reproducibility controls
# -----------------------------
SEED = 42

# Make python hashing stable (ONLY affects things that rely on hash randomization)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Seed python + numpy (split reproducibility & any numpy randomness you add later)
random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# 1) Load + prepare data
# -----------------------------
merged_df5 = pd.read_csv("masters_df2.csv")

target_col = "rmssd"  # change if needed

exclude_cols = ["user_id", "date", target_col]

# Keep only numeric predictors
X_raw = (
    merged_df5
    .drop(columns=exclude_cols, errors="ignore")
    .select_dtypes(include=[np.number])
)

y_raw = merged_df5[target_col].astype(float)

# Drop rows with any NaNs in X or y (keeps alignment)
data = pd.concat([X_raw, y_raw], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# 2) Train/test split (deterministic)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

# Use Pool objects (recommended)
train_pool = Pool(X_train, y_train)
valid_pool = Pool(X_test, y_test)


# -----------------------------
# 3) Define + train CatBoost (deterministic)
# -----------------------------

model = CatBoostRegressor(
    loss_function="RMSE",
    depth=6,
    learning_rate=0.05,
    n_estimators=2000,
    random_seed=SEED,
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=50,
    verbose=100,

    # Determinism:
    thread_count=1,
    allow_writing_files=False,

    # Optional extra determinism (often slightly changes results vs defaults):
    bootstrap_type="No",
    rsm=1.0,
)

model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True
)


# -----------------------------
# 4) Evaluation (deterministic)
# -----------------------------
# Keep aligned arrays
y_test_arr = y_test.to_numpy(dtype=float)
y_pred_arr = model.predict(X_test).astype(float)

mse = mean_squared_error(y_test_arr, y_pred_arr)
rmse = float(np.sqrt(mse))
mae = mean_absolute_error(y_test_arr, y_pred_arr)
r2 = r2_score(y_test_arr, y_pred_arr)

# pearsonr is deterministic for same inputs
pearson_r, _ = pearsonr(y_test_arr, y_pred_arr)

print(f"\nRMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
print(f"Pearson r: {pearson_r:.3f}")


# -----------------------------
# 5) Diagnostic plots (deterministic given same data)
# -----------------------------
# 5.1 True vs Pred
plt.figure()
plt.scatter(y_test_arr, y_pred_arr, alpha=0.5)
plt.xlabel("True RMSSD")
plt.ylabel("Predicted RMSSD")
plt.title("CatBoost: True vs Predicted RMSSD")
min_val = float(min(y_test_arr.min(), y_pred_arr.min()))
max_val = float(max(y_test_arr.max(), y_pred_arr.max()))
plt.plot([min_val, max_val], [min_val, max_val])  # y=x line
plt.grid(True)
plt.show()

# 5.2 Residuals vs Pred
residuals = y_test_arr - y_pred_arr
plt.figure()
plt.scatter(y_pred_arr, residuals, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted RMSSD")
plt.ylabel("Residuals (True - Pred)")
plt.title("CatBoost: Residuals vs Predicted")
plt.grid(True)
plt.show()


# -----------------------------
# 6) Feature importance (deterministic for fixed model)
# -----------------------------
importances = model.get_feature_importance(train_pool)
feat_importance = (
    pd.DataFrame({"feature": X_train.columns, "importance": importances})
    .sort_values("importance", ascending=False)
)

print("\nTop 20 features by importance:")
print(feat_importance.head(20).to_string(index=False))

# Optional: bar plot
plt.figure(figsize=(8, 6))
top_k = 20
top = feat_importance.head(top_k).iloc[::-1]
plt.barh(top["feature"], top["importance"])
plt.xlabel("Importance")
plt.title("CatBoost Feature Importance (Top 20)")
plt.tight_layout()
plt.show()

