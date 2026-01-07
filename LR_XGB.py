import os
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, median_absolute_error
)

from xgboost import XGBRegressor


# ----------------------------
# Reproducibility controls
# ----------------------------
GLOBAL_SEED = 42
# If you want the strictest determinism across machines, keep n_jobs=1 for XGBoost.
XGB_N_JOBS = 1


# ----------------------------
# 1) Load + prepare data
# ----------------------------
merged_df5 = pd.read_csv("masters_df2.csv")

target_col = "demographic_vo2_max"  # change if needed

exclude_cols = [
    "user_id",
    "date",
    "rmssd",
    "filtered_demographic_vo2_max",
    target_col,
]

# Keep numeric predictors only
X_raw = (
    merged_df5
    .drop(columns=exclude_cols, errors="ignore")
    .select_dtypes(include=[np.number])
)

y_raw = merged_df5[target_col].astype(float)

# Drop rows with any missing values (in X or y) while preserving index alignment
data = pd.concat([X_raw, y_raw], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

# Keep aligned user_id if present (for per-user variation)
user_ids = None
if "user_id" in merged_df5.columns:
    user_ids = merged_df5.loc[data.index, "user_id"].astype(str).values


# ----------------------------
# 2) Models (NO pre-scaling; scaling is inside Pipeline to avoid leakage)
# ----------------------------
models = {
    "LinearRegression": Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "XGBoost": Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=GLOBAL_SEED,
            n_jobs=XGB_N_JOBS,
            # Optional: You can set tree_method for stability across GPUs/CPUs.
            # tree_method="hist",
        )),
    ]),
}


# ----------------------------
# 3) Bootstrap CI helpers (deterministic per model/metric)
# ----------------------------
def _seed_from(name: str, metric: str, base_seed: int = GLOBAL_SEED) -> int:
    """
    Deterministic seed derived from (base_seed, model name, metric name).
    Stable across runs and independent of call order.
    """
    # Python's hash() is salted per process by default, so don't use hash().
    # Use a stable custom hashing approach:
    s = f"{base_seed}::{name}::{metric}"
    # Simple stable 32-bit hash:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def bootstrap_ci(values, func, n=2000, alpha=0.05, seed=GLOBAL_SEED) -> tuple[float, float]:
    """
    Bootstrap CI with a local RNG seeded per call -> reproducible and order-independent.
    """
    vals = np.asarray(values)
    if len(vals) == 0:
        return (np.nan, np.nan)
    if len(vals) == 1:
        v = float(func(vals[np.newaxis, :], axis=1)[0])
        return (v, v)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    boots = func(vals[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(trues, preds, name="Model", ids=None, alpha=0.05, n_boot=2000):
    trues = np.asarray(trues, dtype=float)
    preds = np.asarray(preds, dtype=float)

    resid = trues - preds
    ae = np.abs(resid)

    mae = mean_absolute_error(trues, preds)
    rmse = float(np.sqrt(mean_squared_error(trues, preds)))
    r2 = r2_score(trues, preds)
    evs = explained_variance_score(trues, preds)
    medae = median_absolute_error(trues, preds)

    corr = (
        float(np.corrcoef(trues, preds)[0, 1])
        if (np.std(trues) > 0 and np.std(preds) > 0)
        else np.nan
    )

    # Variation metrics
    resid_sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    ae_sd = float(np.std(ae, ddof=1)) if len(ae) > 1 else 0.0
    ae_iqr = float(np.subtract(*np.percentile(ae, [75, 25]))) if len(ae) > 1 else 0.0
    pred_sd = float(np.std(preds, ddof=1)) if len(preds) > 1 else 0.0

    # Bootstrap CIs (order-independent, tied to model+metric)
    mae_seed = _seed_from(name, "mae")
    rmse_seed = _seed_from(name, "rmse")

    mae_lo, mae_hi = bootstrap_ci(
        ae,
        func=lambda x, axis: np.mean(x, axis=axis),
        n=n_boot,
        alpha=alpha,
        seed=mae_seed,
    )

    rmse_lo, rmse_hi = bootstrap_ci(
        resid**2,
        func=lambda x, axis: np.sqrt(np.mean(x, axis=axis)),
        n=n_boot,
        alpha=alpha,
        seed=rmse_seed,
    )

    print(f"\n📊 {name} performance (LOOCV)")
    print(f"   MAE          = {mae:.3f}  (95% CI {mae_lo:.3f}–{mae_hi:.3f})")
    print(f"   MedAE        = {medae:.3f}")
    print(f"   RMSE         = {rmse:.3f}  (95% CI {rmse_lo:.3f}–{rmse_hi:.3f})")
    print(f"   R²           = {r2:.3f}")
    print(f"   ExplainedVar = {evs:.3f}")
    print(f"   Pearson r    = {corr:.3f}")
    print(f"   Residual SD  = {resid_sd:.3f}")
    print(f"   |Error| SD   = {ae_sd:.3f} | IQR = {ae_iqr:.3f}")
    print(f"   Pred SD      = {pred_sd:.3f}")

    # Per-user variation (if IDs provided)
    user_stats = None
    if ids is not None:
        dfu = pd.DataFrame({"user_id": ids, "true": trues, "pred": preds})
        dfu["ae"] = np.abs(dfu["true"] - dfu["pred"])

        user_stats = (
            dfu.groupby("user_id")["ae"]
            .agg(["mean", "median", "std", "count"])
            .rename(columns={"mean": "user_mae", "median": "user_medae", "std": "user_ae_sd", "count": "n"})
        )

        print(
            "   Per-user MAE (mean ± SD across users): "
            f"{user_stats['user_mae'].mean():.3f} ± {user_stats['user_mae'].std(ddof=1):.3f} "
            f"(users={len(user_stats)})"
        )

    return user_stats


# ----------------------------
# 4) LOOCV (deterministic split)
# ----------------------------
loo = LeaveOneOut()

# Store predictions per model
results = {name: [] for name in models}

# NOTE: We use raw X (unscaled), because scaling happens inside each fold via Pipeline.
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name].append({
            "true": float(y_test.values[0]),
            "pred": float(np.squeeze(y_pred)),
        })


# ----------------------------
# 5) Aggregate + summaries
# ----------------------------
for name in results:
    trues = np.array([r["true"] for r in results[name]], dtype=float)
    preds = np.array([r["pred"] for r in results[name]], dtype=float)
    _ = summarize(trues, preds, name=name, ids=user_ids)


# ----------------------------
# 6) Optional: Collect into one DF for plotting
# ----------------------------
summary_df = pd.DataFrame({
    "true": np.array([r["true"] for r in results["XGBoost"]], dtype=float),
    "pred_lr": np.array([r["pred"] for r in results["LinearRegression"]], dtype=float),
    "pred_xgb": np.array([r["pred"] for r in results["XGBoost"]], dtype=float),
})

print("\nHead of summary_df:")
print(summary_df.head())
