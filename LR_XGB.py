import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,
    median_absolute_error
)
from xgboost import XGBRegressor
rng = np.random.default_rng(42)

merged_df5 = pd.read_csv('masters_df2.csv')

# ---- 1. Define target and clean data ----
target_col = "demographic_vo2_max"  # or your HRV variable
exclude_cols = ["user_id", "date", "rmssd", "filtered_demographic_vo2_max",target_col]

X_raw = merged_df5.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=[np.number])
y_raw = merged_df5[target_col].astype(float)

data = pd.concat([X_raw, y_raw], axis=1).dropna()
X = data.drop(columns=[target_col])
y = data[target_col]

# keep aligned user_id if present for per-user variation
user_ids = None
if "user_id" in merged_df5.columns:
    user_ids = merged_df5.loc[data.index, "user_id"].values

# ---- 2) Scale predictors ----
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# ---- 3) Models ----
models = {
    "LinearRegression": LinearRegression(),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

# ---- helpers for variation ----
def bootstrap_ci(values, func=np.mean, n=2000, alpha=0.05, _rng=rng):
    vals = np.asarray(values)
    idx = _rng.integers(0, len(vals), size=(n, len(vals)))
    boots = func(vals[idx], axis=1)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

def summarize(trues, preds, name="Model", ids=None):
    trues = np.asarray(trues, dtype=float)
    preds = np.asarray(preds, dtype=float)
    resid = trues - preds
    ae = np.abs(resid)

    mae  = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2   = r2_score(trues, preds)
    evs  = explained_variance_score(trues, preds)
    medae = median_absolute_error(trues, preds)
    # guard for correlation if constant
    corr = np.corrcoef(trues, preds)[0,1] if (np.std(trues)>0 and np.std(preds)>0) else np.nan

    # variation metrics
    resid_sd = np.std(resid, ddof=1)
    ae_sd    = np.std(ae, ddof=1)
    ae_iqr   = np.subtract(*np.percentile(ae, [75, 25]))
    pred_sd  = np.std(preds, ddof=1)

    # bootstrap CIs
    mae_lo, mae_hi   = bootstrap_ci(ae, func=np.mean)
    rmse_lo, rmse_hi = bootstrap_ci(resid**2, func=lambda x, axis: np.sqrt(np.mean(x, axis=axis)))

    print(f"\n📊 {name} performance (LOOCV)")
    print(f"   MAE        = {mae:.3f}  (95% CI {mae_lo:.3f}–{mae_hi:.3f})")
    print(f"   MedAE      = {medae:.3f}")
    print(f"   RMSE       = {rmse:.3f}  (95% CI {rmse_lo:.3f}–{rmse_hi:.3f})")
    print(f"   R²         = {r2:.3f}")
    print(f"   ExplainedVar = {evs:.3f}")
    print(f"   Pearson r  = {corr:.3f}")
    print(f"   Residual SD= {resid_sd:.3f}")
    print(f"   |Error| SD = {ae_sd:.3f} | IQR = {ae_iqr:.3f}")
    print(f"   Pred SD    = {pred_sd:.3f}")

    # per-user variation (if IDs provided)
    if ids is not None:
        dfu = pd.DataFrame({"user_id": ids, "true": trues, "pred": preds})
        dfu["ae"] = np.abs(dfu["true"] - dfu["pred"])
        user_stats = dfu.groupby("user_id")["ae"].agg(["mean","median","std","count"]).rename(
            columns={"mean":"user_mae","median":"user_medae","std":"user_ae_sd","count":"n"}
        )
        print("   Per-user MAE (mean ± SD across users): "
              f"{user_stats['user_mae'].mean():.3f} ± {user_stats['user_mae'].std(ddof=1):.3f} (users={len(user_stats)})")
        # return user_stats too if you want to inspect later
        return user_stats

    return None

# ---- 4) LOOCV ----
loo = LeaveOneOut()
results = {name: [] for name in models}

for train_idx, test_idx in loo.split(X_scaled):
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name].append({"true": y_test.values[0], "pred": float(np.squeeze(y_pred))})

# ---- 5) Aggregate + variation summaries ----
for name in results:
    preds = np.array([r["pred"] for r in results[name]])
    trues = np.array([r["true"] for r in results[name]])
    ids_subset = user_ids if user_ids is not None else None
    user_stats = summarize(trues, preds, name=name, ids=ids_subset)

# (Optional) Collect predictions for downstream plots
summary_df = pd.DataFrame({
    "true": np.array([r["true"] for r in results["XGBoost"]]),
    "pred_lr": np.array([r["pred"] for r in results["LinearRegression"]]),
    "pred_xgb": np.array([r["pred"] for r in results["XGBoost"]]),
})
summary_df.head()