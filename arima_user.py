import os
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 0) Reproducibility controls
# -----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# (Optional but helpful for bitwise-stable results across machines)
# Set BEFORE importing numpy/statsmodels ideally; kept here for convenience.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(SEED)


# -----------------------------
# 1) Load data
# -----------------------------
d = pd.read_csv("masters_df_imputed.csv")
cols = ["user_id", "date", "resting_hr", "rmssd", "age", "sex", "calories", "steps"]
df = d[cols].copy()


# -----------------------------
# 2) Helpers
# -----------------------------
def prepare_df(df: pd.DataFrame, date_col: str = "date", user_col: str = "user_id") -> pd.DataFrame:
    """
    Make ordering + dtypes stable. This is the main reproducibility requirement here.
    """
    out = df.copy()

    # Stable dtypes
    out[user_col] = out[user_col].astype(str)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # Encode sex deterministically if present
    if "sex" in out.columns:
        if out["sex"].dtype == "object" or str(out["sex"].dtype).startswith("string"):
            out["sex"] = (
                out["sex"].astype(str).str.strip().str.lower()
                .map({"m": 1, "male": 1, "f": 0, "female": 0})
                .fillna(0)
                .astype(int)
            )
        else:
            # If already numeric, coerce to int (stable)
            out["sex"] = pd.to_numeric(out["sex"], errors="coerce").fillna(0).astype(int)

    # Weekday: if present, force int (stable)
    if "weekday" in out.columns:
        out["weekday"] = pd.to_numeric(out["weekday"], errors="coerce").fillna(0).astype(int)

    # IMPORTANT: use a *stable* sort. mergesort is stable.
    out = out.sort_values([user_col, date_col], kind="mergesort").reset_index(drop=True)
    return out


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r; returns nan if degenerate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    # numpy corrcoef is deterministic for fixed inputs
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# -----------------------------
# 3) Main routine
# -----------------------------
def arima_per_user_grid(
    df: pd.DataFrame,
    user_col: str = "user_id",
    date_col: str = "date",
    target_col: str = "rmssd",
    windows=(1, 7, 14),
    order=(1, 0, 0),
    min_train_points_for_arima: int = 3,
    include_weekday: bool = False,
    verbose_failures: bool = False,
):
    """
    Per-user rolling backtest (deterministic):
      - Deterministic preprocessing/sort
      - Deterministic group iteration order (sort=True)
      - ARIMA fit is deterministic for fixed inputs/settings

    Notes:
      - This is ARIMAX when exog_cols != []
      - For w==1 or too-short windows, uses naive persistence y[t-1]
    """
    d = prepare_df(df, date_col=date_col, user_col=user_col)

    base_sets = [
        [],  # rmssd only
        ["resting_hr"],
        ["resting_hr", "steps"],
        ["resting_hr", "steps", "calories"],
    ]
    if include_weekday:
        base_sets = [s + ["weekday"] for s in base_sets]

    results = []

    for w in windows:
        for exog_cols in base_sets:
            y_true_all = []
            y_pred_all = []

            required = [user_col, date_col, target_col] + exog_cols
            d_run = d.dropna(subset=required).copy()

            # Deterministic group order: sort=True
            for uid, g in d_run.groupby(user_col, sort=True):
                g = g.sort_values(date_col, kind="mergesort").reset_index(drop=True)

                y = g[target_col].astype(float).to_numpy()
                X = g[exog_cols].astype(float).to_numpy() if exog_cols else None

                for t in range(w, len(y)):
                    y_true = y[t]
                    if np.isnan(y_true):
                        continue

                    train_y = y[t - w : t]
                    train_X = X[t - w : t] if X is not None else None

                    # If NaNs inside window, drop them (and align exog)
                    if np.any(np.isnan(train_y)):
                        mask = ~np.isnan(train_y)
                        train_y = train_y[mask]
                        if train_X is not None:
                            train_X = train_X[mask]

                    if len(train_y) == 0:
                        continue

                    # Fallback: persistence for tiny windows
                    if (w == 1) or (len(train_y) < min_train_points_for_arima):
                        y_pred = float(train_y[-1])
                    else:
                        try:
                            # Make ARIMA solver settings explicit (helps reproducibility across versions)
                            if train_X is None:
                                fit = ARIMA(
                                    train_y,
                                    order=order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(method_kwargs={"maxiter": 200, "disp": 0})
                                y_pred = float(fit.forecast(steps=1)[0])
                            else:
                                fit = ARIMA(
                                    train_y,
                                    exog=train_X,
                                    order=order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(method_kwargs={"maxiter": 200, "disp": 0})

                                x_next = X[t].reshape(1, -1)  # exog for step t
                                y_pred = float(fit.forecast(steps=1, exog=x_next)[0])

                        except Exception as e:
                            if verbose_failures:
                                print(f"[WARN] uid={uid} w={w} exog={exog_cols} t={t} ARIMA failed: {e}")
                            y_pred = float(train_y[-1])

                    y_true_all.append(float(y_true))
                    y_pred_all.append(float(y_pred))

            n = len(y_true_all)
            feature_set = "rmssd" + ("+" + "+".join(exog_cols) if exog_cols else "")

            if n == 0:
                results.append({
                    "history_days": int(w),
                    "feature_set": feature_set,
                    "n_forecasts": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "pearson_r": np.nan,
                })
                continue

            y_true_arr = np.asarray(y_true_all, dtype=float)
            y_pred_arr = np.asarray(y_pred_all, dtype=float)

            mae = float(mean_absolute_error(y_true_arr, y_pred_arr))

            rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

            r2 = float(r2_score(y_true_arr, y_pred_arr)) if np.std(y_true_arr) > 0 else np.nan
            r = _pearson_r(y_true_arr, y_pred_arr)

            results.append({
                "history_days": int(w),
                "feature_set": feature_set,
                "n_forecasts": int(n),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson_r": r,
            })

    out = (
        pd.DataFrame(results)
        .sort_values(["history_days", "feature_set"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


# -----------------------------
# 4) Run
# -----------------------------
metrics = arima_per_user_grid(
    df,
    user_col="user_id",
    date_col="date",
    target_col="rmssd",
    windows=(1, 7, 14),
    order=(1, 0, 0),
    include_weekday=False,
    verbose_failures=False,
)

print(metrics.to_string(index=False))
