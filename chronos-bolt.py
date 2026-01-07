import os
import random
import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


# -----------------------
# 0) Reproducibility knobs
# -----------------------
SEED = 42
FORCE_CPU = True  # set False if you want GPU; CPU is more reproducible

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Optional thread pinning (reduces nondeterminism from scheduling)
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
torch.use_deterministic_algorithms(True)


# -----------------------
# 1) Config
# -----------------------
MODEL_NAME = "amazon/chronos-bolt-small"

ID_COL = "user_id"
DATE_COL = "date"
TARGET_COL = "rmssd"

CONTEXT_LENS = [1, 7, 14]
PRED_HORIZON = 1
QUANTILES = [0.1, 0.5, 0.9]


# -----------------------
# 2) Load pipeline (deterministic device choice)
# -----------------------
device = "cpu" if FORCE_CPU or (not torch.cuda.is_available()) else "cuda"

pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_NAME,
    device_map=device,  # Chronos accepts "cpu"/"cuda" in many installs
)

# Put underlying model in eval mode just in case
try:
    pipeline.model.eval()
except Exception:
    pass


# -----------------------
# 3) Load + prepare data (stable)
# -----------------------
df = pd.read_csv("masters_df_imputed.csv")

df[ID_COL] = df[ID_COL].astype(str)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

df = (
    df.dropna(subset=[ID_COL, DATE_COL, TARGET_COL])
      .sort_values([ID_COL, DATE_COL], kind="mergesort")
      .reset_index(drop=True)
)


# -----------------------
# 4) Evaluate
# -----------------------
eval_rows = []

# Deterministic group iteration
for uid, g in df.groupby(ID_COL, sort=True):
    g = g.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
    values = g[TARGET_COL].to_numpy(dtype=np.float32)

    n = len(values)
    if n < 2:
        continue

    # predict y[t+1] from context ending at t
    for t in range(n - 1):
        y_true = float(values[t + 1])

        for ctx in CONTEXT_LENS:
            if t + 1 < ctx:
                continue

            ctx_slice = values[(t + 1 - ctx):(t + 1)]  # up to t inclusive

            # ensure deterministic tensor/device
            context_tensor = torch.tensor(ctx_slice, dtype=torch.float32, device=device)

            # Optional: re-seed per prediction so results are invariant to loop order
            # (useful if Chronos internally samples)
            local_seed = (SEED * 1_000_003 + ctx * 10_007 + t * 101) % (2**32 - 1)
            torch.manual_seed(local_seed)
            torch.cuda.manual_seed_all(local_seed)

            with torch.no_grad():
                quantiles, mean = pipeline.predict_quantiles(
                    inputs=[context_tensor],
                    prediction_length=PRED_HORIZON,
                    quantile_levels=QUANTILES,
                )

            y_pred_mean = float(mean[0, 0])

            eval_rows.append(
                {
                    ID_COL: uid,
                    "context_len": int(ctx),
                    "y_true": y_true,
                    "y_pred_mean": y_pred_mean,
                    "t_index": int(t),
                }
            )

eval_df = pd.DataFrame(eval_rows)


# -----------------------
# 5) Metrics (RMSE fixed)
# -----------------------
metric_rows = []

for ctx, g in eval_df.groupby("context_len", sort=True):
    y_true = g["y_true"].to_numpy(dtype=float)
    y_pred = g["y_pred_mean"].to_numpy(dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # FIXED
    r2 = float(r2_score(y_true, y_pred)) if np.std(y_true) > 0 else np.nan

    if len(g) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r, _ = pearsonr(y_true, y_pred)
        pearson_r = float(pearson_r)
    else:
        pearson_r = np.nan

    metric_rows.append(
        {
            "context_len_days": int(ctx),
            "n_points": int(len(g)),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Pearson_r": pearson_r,
        }
    )

metrics_df = pd.DataFrame(metric_rows).sort_values("context_len_days", kind="mergesort").reset_index(drop=True)

print("\nChronos-Bolt performance by context length (days):")
print(metrics_df.to_string(index=False))
