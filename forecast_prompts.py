import os
import pandas as pd
from itertools import combinations

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Create prompts for these history lengths (in days)
HISTORY_LENS = [1, 7, 14]

# Base feature pool to create combinations from (time-series only)
FEATURE_POOL = ["rmssd", "resting_hr",  "steps", "calories"] 

# If True, make combos of size 1..len(FEATURE_POOL). If False, use only full set.
USE_ALL_COMBINATIONS = True

PROMPT_TEMPLATE = """You are an expert in heart rate variability and time series forecasting.

Your task is to predict tomorrow's HRV (RMSSD, milliseconds) for a single person
using the last {history_days} days of time series data.

Static information for this person:
- Age: {age}
- Sex: {sex}

Daily values (older to newer):
{history_block}

Predict the RMSSD value for the NEXT day after the last date above.

You MUST respond with ONLY a single valid JSON object
with this exact structure (no extra keys, no text before or after):

{{
  "pred_rmssd": <float>
}}
"""

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def _safe_str(val):
    """Convert value into a string, using 'nan' if missing. Floats rounded to 2 decimals."""
    if pd.isna(val):
        return "nan"
    # round numeric values
    try:
        # bool is numeric in python; avoid turning True -> 1.00
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return f"{float(val):.2f}"
    except Exception:
        pass
    return str(val)


def make_history_block(history_df: pd.DataFrame, time_series_cols: list, date_col: str) -> str:
    """
    Convert last N days of data into a readable block for the LLM.
    Includes date and all columns listed in time_series_cols.
    """
    lines = []
    for _, row in history_df.iterrows():
        date_str = pd.to_datetime(row[date_col]).date()
        parts = []
        for col in time_series_cols:
            if col in history_df.columns:
                parts.append(f"{col}={_safe_str(row[col])}")
            else:
                parts.append(f"{col}=nan")
        lines.append(f"{date_str}: " + ", ".join(parts))
    return "\n".join(lines)


def all_feature_combinations(feature_pool: list, use_all_combinations: bool = True) -> list:
    """
    Returns list of tuples, each tuple is a combination of features.
    """
    if not use_all_combinations:
        return [tuple(feature_pool)]
    combos = []
    for r in range(1, len(feature_pool) + 1):
        combos.extend(list(combinations(feature_pool, r)))
    return combos


def create_hrv_prompts_df_multi(
    df: pd.DataFrame,
    history_lens: list = HISTORY_LENS,
    feature_pool: list = FEATURE_POOL,
    use_all_combinations: bool = USE_ALL_COMBINATIONS,
    user_col: str = "user_id",
    date_col: str = "date",
    target_col: str = "rmssd",
    static_cols: tuple = ("age", "sex"),
) -> pd.DataFrame:
    """
    Create prompts for 1-day-ahead RMSSD forecast for:
      - multiple history lengths (e.g., 1/7/14)
      - multiple feature combinations (subsets of feature_pool)

    Output columns:
      - prompt_id
      - user_id
      - target_date
      - y_true_rmssd
      - history_days
      - feature_set
      - prompt
    """
    df = df.copy()

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows missing required keys (user/date)
    df = df.dropna(subset=[user_col, date_col])

    # Remove duplicated rows (exact duplicates across all columns)
    df = df.drop_duplicates()

    # Sort for correct history slicing
    df = df.sort_values([user_col, date_col]).reset_index(drop=True)

    feature_sets = all_feature_combinations(feature_pool, use_all_combinations=use_all_combinations)

    rows = []

    for uid, g in df.groupby(user_col, sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)

        # Require static cols exist; take first non-null if possible
        age = g[static_cols[0]].dropna().iloc[0] if static_cols[0] in g.columns and g[static_cols[0]].notna().any() else "nan"
        sex = g[static_cols[1]].dropna().iloc[0] if static_cols[1] in g.columns and g[static_cols[1]].notna().any() else "nan"

        for history_days in history_lens:
            if len(g) <= history_days:
                continue

            for cols_tuple in feature_sets:
                ts_cols = list(cols_tuple)

                # We must have the target_col in df to supervise y_true.
                if target_col not in g.columns:
                    continue

                for i in range(history_days, len(g)):
                    history = g.iloc[i - history_days : i]
                    target_row = g.iloc[i]

                    # Require history target (rmssd) complete and target present
                    if history[target_col].isna().any():
                        continue
                    if pd.isna(target_row[target_col]):
                        continue

                    history_block = make_history_block(
                        history_df=history,
                        time_series_cols=ts_cols,
                        date_col=date_col,
                    )

                    # Build prompt
                    prompt = PROMPT_TEMPLATE.format(
                        history_days=history_days,
                        age=_safe_str(age),
                        sex=_safe_str(sex),
                        history_block=history_block,
                    )

                    rows.append(
                        {
                            "user_id": uid,
                            "target_date": target_row[date_col],
                            "y_true_rmssd": round(float(target_row[target_col]), 2),
                            "history_days": int(history_days),
                            "feature_set": "_".join(ts_cols),
                            "prompt": prompt,
                        }
                    )

    prompts_df = pd.DataFrame(rows)

    # Safety: remove duplicates in the produced prompt rows too
    if not prompts_df.empty:
        prompts_df = prompts_df.drop_duplicates()

    # Add prompt_id
    prompts_df = prompts_df.reset_index(drop=True)
    prompts_df.insert(0, "prompt_id", prompts_df.index.astype(int))

    return prompts_df


# ---------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------

# Example:
# HRV_CSV_PATH = "masters_df_imputed.csv"
df = pd.read_csv(HRV_CSV_PATH)

prompts_df = create_hrv_prompts_df_multi(
    df,
    history_lens=HISTORY_LENS,
    feature_pool=FEATURE_POOL,
    use_all_combinations=USE_ALL_COMBINATIONS,
    user_col="user_id",
    date_col="date",
    target_col="rmssd",
)

OUTPUT_DIR = "/hrv_llm_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

prompts_path = os.path.join(OUTPUT_DIR, "llm_hrv_prompts.parquet")
prompts_df.to_parquet(prompts_path, index=False)

print("Saved:", prompts_path)
print("Rows:", len(prompts_df))
print("Example feature_sets:", prompts_df["feature_set"].unique()[:10] if len(prompts_df) else [])
print("Example history_days:", prompts_df["history_days"].unique() if len(prompts_df) else [])

