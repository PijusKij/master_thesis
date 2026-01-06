df = pd.read_csv('masters_df2.csv')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np


def preprocess_hrv_for_mitra(
    df: pd.DataFrame,
    target_col: str = "rmssd",        # your HRV target
    id_cols=("user_id", "date"),      # cols to drop from features if present
    min_non_null_frac: float = 0.7,   # drop very sparse columns
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Preprocess HRV tabular data for Mitra regression model.
    Returns: train_df, test_df ready for AutoGluon / Mitra.
    """
    df = df.copy()

    # ---- 1. Keep only reasonably complete columns ----
    non_null_frac = df.notnull().mean()
    keep_cols = non_null_frac[non_null_frac >= min_non_null_frac].index
    df = df[keep_cols]

    # ---- 2. Drop ID / time columns (not useful as features) ----
    for col in id_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    # ---- 3. Ensure target exists & drop rows with missing target ----
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in df.columns")

    df = df[~df[target_col].isna()]

    # ---- 4. Basic type cleaning ----
    # Convert bool → int for compatibility
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Infer numeric vs. non-numeric
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Exclude target from auto-cat detection if it is numeric
    non_num_cols = [c for c in df.columns if c not in num_cols]

    # Treat remaining non-numeric columns as categorical
    cat_cols = [c for c in non_num_cols if c != target_col]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    # ---- 5. Handle missing values ----
    # Numeric: median; Categorical: mode
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # ---- 6. Train / test split (Mitra wants label inside the DF) ----
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    return train_df, test_df

train_df, test_df = preprocess_hrv_for_mitra(
    df,
    target_col="rmssd",
    id_cols=("user_id", "date"),
)

from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import numpy as np

label_col = "rmssd"

# Create predictor for regression with Mitra
mitra_predictor = TabularPredictor(
    label=label_col,
    problem_type="regression",
    eval_metric="rmse",                      # AutoGluon will optimize RMSE
    path="AutogluonModels/mitra_rmssd",      # where to save the model
)

# --- Option 1: Zero-shot Mitra (no fine-tuning, fast) ---
mitra_predictor.fit(
    train_data=train_df,
    hyperparameters={
        "MITRA": {"fine_tune": False}
    },
)


# ---- 4. Predict on test set ----
y_true = test_df[label_col]
y_pred = mitra_predictor.predict(test_df)

# ---- 5. Metrics ----
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
r2   = r2_score(y_true, y_pred)
pearson_r, p_val = pearsonr(y_true, y_pred)

print("Mitra RMSSD regression performance:")
print(f"RMSE      : {rmse:.3f}")
print(f"MAE       : {mae:.3f}")
print(f"R²        : {r2:.3f}")
print(f"Pearson r : {pearson_r:.3f} (p={p_val:.3e})")
