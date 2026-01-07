import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

import torch

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

# ---------------------------
# 0) Reproducibility controls
# ---------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# Optional: reduce nondeterminism from BLAS/thread scheduling
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# If any nondeterministic op is used, this may raise an error:
torch.use_deterministic_algorithms(True)

# Ensures deterministic cublas (important on CUDA)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------------------
# 1) Load + define X/y
# ---------------------------
df = pd.read_csv("masters_df2.csv")

target_col = "demographic_vo2_max"  # change to your target (e.g., "rmssd")
exclude_cols = ["user_id", "date", target_col]

# Stable ordering (helps reproducibility if you later do group-based things)
if "user_id" in df.columns and "date" in df.columns:
    dtmp = df.copy()
    dtmp["user_id"] = dtmp["user_id"].astype(str)
    dtmp["date"] = pd.to_datetime(dtmp["date"], errors="coerce")
    df = dtmp.sort_values(["user_id", "date"], kind="mergesort").reset_index(drop=True)

# Numeric-only features (TabPFN needs numeric; if you have categoricals, encode them separately)
X_raw = df.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=[np.number])
y_raw = pd.to_numeric(df[target_col], errors="coerce")

data = pd.concat([X_raw, y_raw.rename(target_col)], axis=1).dropna()
X = data.drop(columns=[target_col]).to_numpy(dtype=np.float32)
y = data[target_col].to_numpy(dtype=np.float32)

# ---------------------------
# 2) Split (deterministic)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    shuffle=True
)

# ---------------------------
# 3) TabPFN (reproducible as possible)
# ---------------------------
# If your tabpfn version supports it, prefer specifying device explicitly.
# Many installations run on CPU by default; CPU tends to be most reproducible.
# For TabPFN v2 explicitly:
# regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)

regressor = TabPFNRegressor()  # defaults to TabPFN-2.5 weights in many setups

# Fit + predict (TabPFN fit is lightweight; mostly sets up internal state)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# ---------------------------
# 4) Metrics
# ---------------------------
rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
mae = float(mean_absolute_error(y_test, predictions))
r2 = float(r2_score(y_test, predictions))

if np.std(y_test) == 0 or np.std(predictions) == 0:
    pearson_r, p_value = np.nan, np.nan
else:
    pearson_r, p_value = pearsonr(y_test, predictions)

print("TabPFN regression performance:")
print(f"R²        : {r2:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")
print(f"Pearson r : {pearson_r:.4f}")
print(f"p-value   : {p_value:.4e}")
