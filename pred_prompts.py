import os
import pandas as pd
import numpy as np

# ========= BASIC PATH/GRID CONFIG =========
OUTPUT_DIR = "llm_prompts_tabular_no_history"

# Feature scenarios to test (NO weekday logic)
FEATURE_SCENARIOS = {
    "1": ["resting_hr"],
    "2": ["resting_hr", "steps"],
    "3": ["resting_hr", "steps", "calories"],
}

# Column Mappings
USER_COL = "user_id"
DATE_COL = "date"
AGE_COL = "age"
SEX_COL = "sex"
TARGET_COL = "rmssd"
HRV_CSV_PATH = "masters_df_imputed.csv"

# ==============================================================================
# 1. PROMPT (No Past Data) — weekday references removed
# ==============================================================================
PROMPT_TEMPLATE = """You are an expert Physiologist and Data Scientist.

### TASK
Estimate the **RMSSD (Heart Rate Variability)** for a subject based on their biometric snapshot for a specific day.
*Note: You do not have access to their history. You must infer values based on population norms and physiological signals.*

### DOMAIN LOGIC
1.  **Population Baseline (The Starting Point):**
    - RMSSD declines with Age.
    - *Example Norms:* 20s (~50-80ms), 30s (~35-60ms), 50s+ (~15-35ms).
    - Men often have slightly higher raw RMSSD than women in some age groups, but individual variance is high.
2.  **The "Stress" Signal (Resting HR):**
    - **Resting HR (RHR)** is your strongest proxy for the Autonomic Nervous System.
    - *Inverse Relationship:* If RHR is LOW (athletic/relaxed), RMSSD is likely HIGH. If RHR is HIGH (stressed/sick), RMSSD is likely LOW.
3.  **Activity Context:**
    - High Steps/Calories implies an active day. If RHR is low despite high activity, the subject is fit (High RMSSD).

### INPUT DATA
--------------------------------------------------
{input_features_block}
--------------------------------------------------

### REASONING STEPS
1.  **Establish Baseline:** Based on **Age/Sex**, what is the expected range?
2.  **Analyze Physiological State:** Look at **Resting HR**. Is this person currently stressed (High RHR) or recovered (Low RHR)?
3.  **Predict:** Adjust the Age-based baseline up or down based on the RHR and Activity signals.

### OUTPUT FORMAT
Respond with ONLY a single valid JSON object.
{{
  "reasoning": "Briefly describe the Age Baseline -> RHR adjustment logic.",
  "pred_rmssd": <float>
}}
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# 2. FORMATTING HELPERS (round all numbers to 2 decimals)
# ==============================================================================

def _get_sex_name(sex_int):
    # Adjust based on your data: 0=Female, 1=Male assumed here
    try:
        if pd.isna(sex_int):
            return "Unknown"
        sex_int = int(float(sex_int))
    except Exception:
        return str(sex_int)

    if sex_int == 1:
        return "Male"
    if sex_int == 0:
        return "Female"
    return str(sex_int)

def _fmt_2dp(val):
    """Format numeric values to 2 decimals, otherwise string. NaN -> Unknown."""
    if pd.isna(val):
        return "Unknown"
    # Avoid converting booleans to 1.00/0.00
    if isinstance(val, (bool, np.bool_)):
        return str(val)
    try:
        num = float(val)
        return f"{num:.2f}"
    except Exception:
        return str(val)

def _safe_val(val, suffix=""):
    out = _fmt_2dp(val)
    if out == "Unknown":
        return out
    return f"{out}{suffix}"

def create_feature_block(row, feature_cols):

    lines = []

    # 1. Static Context
    lines.append(f"- Subject Age: {_safe_val(row.get(AGE_COL, np.nan), ' years')}")
    lines.append(f"- Subject Sex: {_get_sex_name(row.get(SEX_COL, np.nan))}")

    # 2. Dynamic Features
    for col in feature_cols:
        val = row.get(col, np.nan)

        if col == "resting_hr":
            lines.append(f"- Resting Heart Rate: {_safe_val(val, ' bpm')}")
        elif col == "steps":
            lines.append(f"- Daily Steps: {_safe_val(val)}")
        elif col == "calories":
            lines.append(f"- Calories Burned: {_safe_val(val, ' kcal')}")
        else:
            lines.append(f"- {col}: {_safe_val(val)}")

    return "\n".join(lines)

# ==============================================================================
# 3. MAIN GENERATION FUNCTION
# ==============================================================================

def create_tabular_prompts_df(
    df: pd.DataFrame,
    feature_cols: list,
    user_col: str = USER_COL,
    date_col: str = DATE_COL,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    df = df.copy()

    # Parse + sort
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[user_col, date_col]) if (user_col in df.columns and date_col in df.columns) else df
    if user_col in df.columns and date_col in df.columns:
        df = df.sort_values([user_col, date_col])

    # 1) Drop exact duplicate rows
    df = df.drop_duplicates()

    # 2) Ensure no duplicated (user_id, date) rows (common issue in wearables)
    if user_col in df.columns and date_col in df.columns:
        df = df.drop_duplicates(subset=[user_col, date_col], keep="first")

    rows = []

    for _, row in df.iterrows():
        # Skip rows where target is missing (can't validate prediction)
        if target_col not in df.columns or pd.isna(row.get(target_col, np.nan)):
            continue

        input_block = create_feature_block(row, feature_cols)
        full_prompt = PROMPT_TEMPLATE.format(input_features_block=input_block)

        y_true = row.get(target_col, np.nan)
        try:
            y_true = round(float(y_true), 2)
        except Exception:
            # if it's not numeric for some reason, keep as-is
            pass

        rows.append(
            {
                "user_id": row.get(user_col),
                "target_date": row.get(date_col),
                "y_true_rmssd": y_true,
                "prompt": full_prompt,
            }
        )

    prompts_df = pd.DataFrame(rows)

    # Add Prompt ID
    if not prompts_df.empty:
        prompts_df = prompts_df.drop_duplicates()  # safety on output too
        prompts_df = prompts_df.reset_index(drop=True)
        prompts_df.insert(0, "prompt_id", prompts_df.index.astype(int))
        prompts_df = prompts_df[["prompt_id", "user_id", "target_date", "y_true_rmssd", "prompt"]]

    return prompts_df

# ==============================================================================
# 4. EXECUTION LOOP
# ==============================================================================

if __name__ == "__main__":
    df = pd.read_csv(HRV_CSV_PATH)

    meta_rows = []

    for scenario_name, feat_cols in FEATURE_SCENARIOS.items():
        filename_base = f"tabular_prompts_nohist_{scenario_name}"
        parquet_path = os.path.join(OUTPUT_DIR, filename_base + ".parquet")
        csv_path = os.path.join(OUTPUT_DIR, filename_base + ".csv")

        print(f"\n▶ Generating prompts for Scenario: {scenario_name}")
        print(f"   Features: {feat_cols}")

        prompts_df = create_tabular_prompts_df(df, feature_cols=feat_cols)

        if not prompts_df.empty:
            print(f"   → Generated {len(prompts_df)} prompts")
            prompts_df.to_parquet(parquet_path, index=False)
            prompts_df.to_csv(csv_path, index=False)

            print("\n--- Prompt Preview (ID: 0) ---")
            print(prompts_df.iloc[0]["prompt"])
            print("------------------------------\n")

            meta_rows.append(
                {
                    "scenario_name": scenario_name,
                    "n_prompts": len(prompts_df),
                    "features": ",".join(feat_cols),
                    "parquet_path": parquet_path,
                }
            )
        else:
            print("   ⚠️ No prompts generated.")

    if meta_rows:
        meta_df = pd.DataFrame(meta_rows)
        meta_path = os.path.join(OUTPUT_DIR, "prompt_grid_metadata.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"Metadata saved to: {meta_path}")
