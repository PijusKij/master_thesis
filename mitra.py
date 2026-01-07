import os
import random
import shutil
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from autogluon.tabular import TabularPredictor


# =============================
# 0) Reproducibility controls
# =============================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

# (Optional) reduce nondeterminism from thread scheduling
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

random.seed(SEED)
np.random.seed(SEED)


# =============================
# 1) Load
# =============================
df = pd.read_csv("masters_df.csv")


def preprocess_hrv_for_mitra(
    df: pd.DataFrame,
    target_col: str = "rmssd",
    id_cols=("user_id", "date"),
    min_non_null_frac: float = 0.7,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Deterministic preprocessing:
      - Stable column selection + stable column order
      - Stable sorting of rows before split
      - Deterministic imputations
      - Deterministic train/test split
    """
    d = df.copy()

    # --- Ensure stable row order before anything else ---
    # If you have user_id/date, sort by them. Otherwise sort by index.
    if ("user_id" in d.columns) and ("date" in d.columns):
        d["user_id"] = d["user_id"].astype(str)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.sort_values(["user_id", "date"], kind="mergesort")
    else:
        d = d.sort_index(kind="mergesort")
    d = d.reset_index(drop=True)

    # --- 1) Keep only reasonably complete columns (stable order preserved) ---
    non_null_frac = d.notnull().mean()
    keep_cols = [c for c in d.columns if non_null_frac.loc[c] >= min_non_null_frac]
    d = d[keep_cols]

    # --- 2) Drop ID/time columns ---
    for col in id_cols:
        if col in d.columns:
            d = d.drop(columns=col)

    # --- 3) Ensure target exists & drop rows with missing target ---
    if target_col not in d.columns:
        raise ValueError(f"target_col '{target_col}' not found in df.columns")
    d = d.loc[~d[target_col].isna()].reset_index(drop=True)

    # --- 4) Basic type cleaning ---
    bool_cols = d.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        d[bool_cols] = d[bool_cols].astype(int)

    # Identify numeric and categorical columns (excluding target)
    num_cols = d.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in d.columns if c not in num_cols and c != target_col]

    # Force categorical dtype deterministically
    for c in cat_cols:
        d[c] = d[c].astype("category")

    # --- 5) Handle missing values deterministically ---
    # Numeric: median; Categorical: mode (if ties, mode().iloc[0] is deterministic after stable ordering)
    for c in num_cols:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())

    for c in cat_cols:
        if d[c].isna().any():
            mode_val = d[c].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) else "missing"
            d[c] = d[c].fillna(fill_val)

    # --- 6) Train/test split (deterministic) ---
    train_df, test_df = train_test_split(
        d,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Ensure stable column order in outputs
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


train_df, test_df = preprocess_hrv_for_mitra(
    df,
    target_col="rmssd",
    id_cols=("user_id", "date"),
    random_state=SEED,
)

label_col = "rmssd"


# =============================
# 2) Train Mitra reproducibly
# =============================
# Important: reuse an existing path can cause "resume" behavior or reuse artifacts.
# For clean reproducibility, remove it (or use a unique path per run/seed).
model_path = f"AutogluonModels/mitra_rmssd_seed{SEED}"
if os.path.exists(model_path):
    shutil.rmtree(model_path)

mitra_predictor = TabularPredictor(
    label=label_col,
    problem_type="regression",
    eval_metric="rmse",
    path=model_path,
)

mitra_predictor.fit(
    train_data=train_df,
    hyperparameters={
        "MITRA": {
            "fine_tune": False,   # zero-shot as you wanted
        }
    },
    # These improve determinism:
    random_seed=SEED,      # AutoGluon-level seed
    num_cpus=1,            # reduce nondeterminism from multithreading
    num_gpus=0,            # GPU can introduce extra nondeterminism (set >0 only if needed)
    verbosity=2,
)


# =============================
# 3) Evaluate
# =============================
y_true = test_df[label_col].to_numpy(dtype=float)
y_pred = mitra_predictor.predict(test_df).to_numpy(dtype=float)

rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae  = float(mean_absolute_error(y_true, y_pred))
r2   = float(r2_score(y_true, y_pred))

if np.std(y_true) == 0 or np.std(y_pred) == 0:
    pearson_r, p_val = np.nan, np.nan
else:
    pearson_r, p_val = pearsonr(y_true, y_pred)

print("Mitra RMSSD regression performance:")
print(f"RMSE      : {rmse:.3f}")
print(f"MAE       : {mae:.3f}")
print(f"R²        : {r2:.3f}")
print(f"Pearson r : {pearson_r:.3f} (p={p_val:.3e})")
