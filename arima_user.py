import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the imputed dataset
d = pd.read_csv('masters_df_imputed.csv')
cols = ["user_id", "date", "resting_hr", "rmssd", "age", "sex", "calories", "steps"]
df = d[cols]

def prepare_df(df: pd.DataFrame, date_col="date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(["user_id", date_col]).reset_index(drop=True)

    # Encode sex if needed
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = (
            df["sex"].astype(str).str.lower()
            .map({"m": 1, "male": 1, "f": 0, "female": 0})
            .fillna(0).astype(int)
        )

    # Encode weekday if needed
    #if "weekday" in df.columns and df["weekday"].dtype == "object":
    #    wd_map = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
    #    df["weekday"] = df["weekday"].astype(str).str.lower().map(wd_map)

    if "weekday" in df.columns:
        df["weekday"] = df["weekday"].astype(int)

    return df



def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r with scipy if available; numpy fallback; returns nan if degenerate."""
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(y_true, y_pred)
        return float(r)
    except Exception:
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return np.nan
        return float(np.corrcoef(y_true, y_pred)[0, 1])


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
    Per-user rolling backtest:
      For each user and each t >= w:
        train_y = y[t-w:t]
        predict y[t] using ARIMA(train_y) or ARIMAX(train_y, exog=train_X)
      Aggregate forecasts over all users/t for overall metrics.

    Feature sets are implemented as exogenous regressors (ARIMAX).
    If feature set is "rmssd" only -> plain ARIMA.
    """
    #d = df.copy()
    #d[date_col] = pd.to_datetime(d[date_col])
    #d = d.sort_values([user_col, date_col]).reset_index(drop=True)
    d = prepare_df(df, date_col=date_col)
    # Define exogenous feature combos (mirrors your LSTM/GRU combos)
    # We keep weekday in all runs by default, since you did in GRU/LSTM.
    base_sets = [
        [],  # rmssd only -> no exog
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

            # enforce required columns for this run
            required = [user_col, date_col, target_col] + exog_cols
            d_run = d.dropna(subset=required).copy()

            for uid, g in d_run.groupby(user_col, sort=False):
                g = g.sort_values(date_col).reset_index(drop=True)

                y = g[target_col].astype(float).values
                X = None
                if len(exog_cols) > 0:
                    X = g[exog_cols].astype(float).values  # shape (T, K)

                for t in range(w, len(y)):
                    y_true = y[t]
                    if np.isnan(y_true):
                        continue

                    train_y = y[t - w : t]
                    if np.any(np.isnan(train_y)):
                        # Drop NaNs inside the window: align train_X accordingly
                        mask = ~np.isnan(train_y)
                        train_y = train_y[mask]
                        if X is not None:
                            train_X = X[t - w : t][mask]
                        else:
                            train_X = None
                    else:
                        train_X = X[t - w : t] if X is not None else None

                    if len(train_y) == 0:
                        continue

                    # Fallback logic (same spirit as your original)
                    if (w == 1) or (len(train_y) < min_train_points_for_arima):
                        y_pred = float(train_y[-1])
                    else:
                        try:
                            if train_X is None:
                                fit = ARIMA(train_y, order=order).fit()
                                y_pred = float(fit.forecast(steps=1)[0])
                            else:
                                fit = ARIMA(train_y, exog=train_X, order=order).fit()
                                # forecast needs exog for the next step (t)
                                x_next = X[t].reshape(1, -1)
                                y_pred = float(fit.forecast(steps=1, exog=x_next)[0])
                        except Exception as e:
                            if verbose_failures:
                                print(f"[WARN] uid={uid} w={w} exog={exog_cols} t={t} ARIMA failed: {e}")
                            y_pred = float(train_y[-1])

                    y_true_all.append(float(y_true))
                    y_pred_all.append(float(y_pred))

            n = len(y_true_all)
            if n == 0:
                results.append({
                    "history_days": w,
                    "feature_set": "rmssd" + ("+" + "+".join(exog_cols) if exog_cols else ""),
                    "n_forecasts": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "pearson_r": np.nan,
                })
                continue

            y_true_arr = np.array(y_true_all, dtype=float)
            y_pred_arr = np.array(y_pred_all, dtype=float)

            mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
            rmse = float(mean_squared_error(y_true_arr, y_pred_arr))  # true RMSE
            r2 = float(r2_score(y_true_arr, y_pred_arr)) if np.std(y_true_arr) > 0 else np.nan
            r = _pearson_r(y_true_arr, y_pred_arr)

            results.append({
                "history_days": w,
                "feature_set": "rmssd" + ("+" + "+".join(exog_cols) if exog_cols else ""),
                "n_forecasts": int(n),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson_r": r,
            })

    out = pd.DataFrame(results).sort_values(["history_days", "feature_set"]).reset_index(drop=True)
    return out


# ---- Run ----
metrics = arima_per_user_grid(
    df,
    user_col="user_id",
    date_col="date",
    target_col="rmssd",
    windows=(1, 7, 14),
    order=(1, 0, 0),
    include_weekday=False,   # set False if you want strictly the 4 combos without weekday
)

print(metrics.to_string(index=False))
