from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
@dataclass(frozen=True)
class Config:
    data_dir: Path = Path(".")
    user_tz: str = "Europe/Vilnius"

    # folders / files relative to data_dir
    heartrate_dir: str = "heart_rate"
    hrv_dir: str = "heart_rate_variability"
    physical_activity_dir: str = "additional_physical_activity_data"

    anthropometrics_csv: str = "anthropometrics.csv"
    participant_info_csv: str = "participant_information.csv"


CFG = Config()


# ============================================================
# Generic IO / combine utilities
# ============================================================
def uid_from_key_default(key: str) -> str:
    # e.g. "A4F_92332_hrv" -> "92332"
    return str(key)[4:9]


def load_csv_folder(folder: Path, *, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load all .csv files in a folder into a dict keyed by filename stem."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    out: Dict[str, pd.DataFrame] = {}
    for f in sorted(folder.glob("*.csv")):
        df = pd.read_csv(f)
        out[f.stem] = df
        if verbose:
            print(f"Loaded {f.name} with shape {df.shape}")
    return out


def combine_user_frames(
    frames: Dict[str, pd.DataFrame],
    *,
    uid_from_key: Callable[[str], str] = uid_from_key_default,
    user_id_col: str = "user_id",
    verbose: bool = True,
) -> pd.DataFrame:
    """Concat dict of per-user DataFrames into one DataFrame and add `user_id`."""
    if not frames:
        return pd.DataFrame()

    dfs = []
    for key, df in frames.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            if verbose:
                print(f"[{key}] Skipped (empty/invalid)")
            continue
        temp = df.copy()
        temp[user_id_col] = uid_from_key(key)
        dfs.append(temp)

    if not dfs:
        if verbose:
            print("⚠️ No valid DataFrames found.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"✅ Combined {len(dfs)} users → {combined.shape[0]} total rows")
    return combined


def read_participant_info(path: Path) -> pd.DataFrame:
    """participant_information.csv: derive user_id from id and drop id."""
    pi = pd.read_csv(path)
    pi["user_id"] = pi["id"].astype(str).str[4:9]
    return pi.drop(columns=["id"], errors="ignore")


# ============================================================
# Dataset-specific cleaning (small + explicit)
# ============================================================
def clean_hrv(hrv: pd.DataFrame) -> pd.DataFrame:
    hrv = hrv.copy()
    if "timestamp" in hrv.columns:
        hrv["timestamp"] = pd.to_datetime(hrv["timestamp"], errors="coerce").dt.date
    return hrv


def clean_physical_activity(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["date"] = pd.to_datetime(pa["timestamp"], errors="coerce").dt.normalize()
    pa = pa.drop(columns=[c for c in ("timestamp", "distance") if c in pa.columns], errors="ignore")
    pa["user_id"] = pa["user_id"].astype(str).str.strip()
    return pa


def standardize_merge_keys(df: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    """Ensure user_id is string and date is datetime64[ns] normalized."""
    out = df.copy()
    out["user_id"] = out["user_id"].astype(str).str.strip()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    return out


# ============================================================
# RHR computation (cleaner, less nested, fewer copies)
# ============================================================
def compute_daily_rhr_anytime_only(
    dataframes_heartrate: Dict[str, pd.DataFrame],
    *,
    ts_col: str = "timestamp",
    hr_col: str = "heart_rate",
    user_local_tz: str = "Europe/Vilnius",
    resample_rule: str = "15s",
    min_rest_minutes: int = 5,
    lowvar_std_thresh: float = 2.0,
    rhr_hr_cap: tuple[float, float] = (35, 95),
    fallback_quantile: float = 0.05,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Daily RHR = lowest "stable" HR across local calendar day.
    Stable: 5-min rolling std <= lowvar_std_thresh AND 5-min rolling mean in rhr_hr_cap,
            sustained for >= min_rest_minutes contiguously.
    Fallback: daily quantile of 5-min rolling mean.
    Returns: DataFrame [user_id, date, resting_hr, method, samples_used, notes]
    """
    step_seconds = int(pd.Timedelta(resample_rule).total_seconds())
    need_n = max(1, int((min_rest_minutes * 60) / step_seconds))

    def normalize_to_utc(ts: pd.Series) -> pd.DatetimeIndex:
        s = pd.to_datetime(ts, errors="coerce")
        # tz-aware series
        if pd.api.types.is_datetime64tz_dtype(s):
            return pd.DatetimeIndex(s.dt.tz_convert("UTC"))
        # naive datetime64 series
        if pd.api.types.is_datetime64_dtype(s):
            s_local = s.dt.tz_localize(user_local_tz, nonexistent="shift_forward", ambiguous="NaT")
            return pd.DatetimeIndex(s_local.dt.tz_convert("UTC"))
        # object/mixed
        def to_utc_one(x):
            if pd.isna(x):
                return pd.NaT
            x = pd.to_datetime(x, errors="coerce")
            if pd.isna(x):
                return pd.NaT
            if getattr(x, "tzinfo", None) is not None:
                return x.tz_convert("UTC")
            return x.tz_localize(user_local_tz).tz_convert("UTC")
        return pd.DatetimeIndex(s.apply(to_utc_one))

    def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
        out = df[[ts_col, hr_col]].copy()
        out[hr_col] = pd.to_numeric(out[hr_col], errors="coerce")
        out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
        out = out.dropna(subset=[ts_col, hr_col]).sort_values(ts_col).drop_duplicates(subset=[ts_col])
        # wide sanity clip
        return out[(out[hr_col] >= 25) & (out[hr_col] <= 230)]

    def resample_roll(df_utc_indexed: pd.DataFrame) -> pd.DataFrame:
        s = (
            df_utc_indexed[hr_col]
            .resample(resample_rule)
            .median()
            .interpolate(limit=3, limit_direction="both")
        )
        hr5_mean = s.rolling("5min", min_periods=10).mean()
        hr5_std = s.rolling("5min", min_periods=10).std()
        out = pd.DataFrame({"hr": s, "hr5_mean": hr5_mean, "hr5_std": hr5_std}).dropna()
        return out

    def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
        """Return (start_idx, end_idx) inclusive segments where mask==True with length>=need_n."""
        if mask.size == 0:
            return []
        # pad with False to detect edges
        change = np.diff(np.r_[False, mask, False].astype(int))
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0] - 1
        segs = [(s, e) for s, e in zip(starts, ends) if (e - s + 1) >= need_n]
        return segs

    rows = []

    for raw_key, raw_df in dataframes_heartrate.items():
        user_id = uid_from_key_default(raw_key)  # keep consistent everywhere

        try:
            base = clean_raw(raw_df)
            if base.empty:
                if debug:
                    print(f"[{user_id}] empty after clean")
                continue

            idx_utc = normalize_to_utc(base[ts_col])
            base = base.loc[~idx_utc.isna()].copy()
            idx_utc = idx_utc[~idx_utc.isna()]
            if base.empty:
                if debug:
                    print(f"[{user_id}] empty after tz normalization")
                continue

            base.index = idx_utc
            base = base.sort_index()

            df15 = resample_roll(base)
            if df15.empty:
                if debug:
                    print(f"[{user_id}] empty after resample/rolling")
                continue

            # group by LOCAL calendar days
            df15_local = df15.copy()
            df15_local.index = df15_local.index.tz_convert(user_local_tz)

            for day, day_df in df15_local.groupby(df15_local.index.normalize()):
                # stable mask
                stable = (day_df["hr5_std"].to_numpy() <= lowvar_std_thresh) & (
                    (day_df["hr5_mean"].to_numpy() >= rhr_hr_cap[0])
                    & (day_df["hr5_mean"].to_numpy() <= rhr_hr_cap[1])
                )

                segs = contiguous_segments(stable)

                if segs:
                    # choose lowest hr5_mean within any stable segment
                    means = day_df["hr5_mean"].to_numpy()
                    best_val = np.inf
                    best_len = None
                    for s, e in segs:
                        v = np.nanmin(means[s : e + 1])
                        if v < best_val:
                            best_val = v
                            best_len = (e - s + 1)

                    rows.append(
                        {
                            "user_id": user_id,
                            "date": pd.Timestamp(day).date(),
                            "resting_hr": float(best_val),
                            "method": "anytime_lowvar",
                            "samples_used": int(best_len) if best_len is not None else np.nan,
                            "notes": "",
                        }
                    )
                    if debug:
                        print(f"[{user_id} {day.date()}] anytime_lowvar ✓ RHR={best_val:.1f}")
                    continue

                # fallback
                fb = float(day_df["hr5_mean"].quantile(fallback_quantile))
                rows.append(
                    {
                        "user_id": user_id,
                        "date": pd.Timestamp(day).date(),
                        "resting_hr": fb,
                        "method": f"anytime_p{int(fallback_quantile*100)}",
                        "samples_used": np.nan,
                        "notes": "No stable segment; used daily low-end percentile of HR5.",
                    }
                )
                if debug:
                    print(f"[{user_id} {day.date()}] fallback ✓ RHR={fb:.1f}")

        except Exception as e:
            if debug:
                print(f"[{user_id}] ERROR {type(e).__name__}: {e}")
            rows.append(
                {
                    "user_id": user_id,
                    "date": None,
                    "resting_hr": None,
                    "method": "error",
                    "samples_used": None,
                    "notes": f"{type(e).__name__}: {e}",
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.dropna(subset=["date"]).sort_values(["user_id", "date"]).reset_index(drop=True)
    return out


# ============================================================
# Merge pipeline helper
# ============================================================
def left_merge_on_user_date(base: pd.DataFrame, other: pd.DataFrame, *, other_date_col: str = "date") -> pd.DataFrame:
    """Standardize keys and left-merge on [user_id, date]."""
    base2 = standardize_merge_keys(base, date_col="date")
    other2 = standardize_merge_keys(other.rename(columns={other_date_col: "date"}), date_col="date")
    return base2.merge(other2, on=["user_id", "date"], how="left")


# ============================================================
# Main pipeline
# ============================================================
# ---- Load base tables
ds1 = pd.read_csv(CFG.data_dir / CFG.anthropometrics_csv)
pi = read_participant_info(CFG.data_dir / CFG.participant_info_csv)

# ---- Load per-user streams
dataframes_heartrate = load_csv_folder(CFG.data_dir / CFG.heartrate_dir)
dataframes_hrv = load_csv_folder(CFG.data_dir / CFG.hrv_dir)
dataframes_physical_activity = load_csv_folder(CFG.data_dir / CFG.physical_activity_dir)

# ---- Combine + clean
combined_hrv_df = clean_hrv(combine_user_frames(dataframes_hrv))  # timestamp -> date
combined_hrv_df = combined_hrv_df.rename(columns={"timestamp": "date"})
combined_hrv_df = standardize_merge_keys(combined_hrv_df, date_col="date")

combined_physical_activity_df = clean_physical_activity(combine_user_frames(dataframes_physical_activity))
combined_physical_activity_df = standardize_merge_keys(combined_physical_activity_df, date_col="date")

# ---- RHR daily
rhr_daily = compute_daily_rhr_anytime_only(
    dataframes_heartrate,
    ts_col="timestamp",
    hr_col="heart_rate",
    user_local_tz=CFG.user_tz,
    min_rest_minutes=5,
    lowvar_std_thresh=2.0,
    rhr_hr_cap=(35, 95),
    fallback_quantile=0.05,
    debug=True,
)
if not rhr_daily.empty:
    rhr_daily["resting_hr"] = pd.to_numeric(rhr_daily["resting_hr"], errors="coerce").round().astype("Int64")
rhr_daily = rhr_daily[["user_id", "date", "resting_hr"]]
rhr_daily = standardize_merge_keys(rhr_daily, date_col="date")

# ---- Merge all (assumes you already have: combined_sleep_df, features_daily)
merged = left_merge_on_user_date(rhr_daily, combined_hrv_df)                # RHR + HRV
merged = left_merge_on_user_date(merged, combined_sleep_df, other_date_col="date")  # + sleep
merged = left_merge_on_user_date(merged, features_daily, other_date_col="date")    # + features

# participant info is user-level only
pi["user_id"] = pi["user_id"].astype(str).str.strip()
merged = merged.merge(pi, on="user_id", how="left")

# + physical activity
merged = left_merge_on_user_date(merged, combined_physical_activity_df)

# quick checks
print("HRV:", combined_hrv_df.shape)
print("Physical activity:", combined_physical_activity_df.shape)
print("RHR daily:", rhr_daily.shape)
print("Merged:", merged.shape)

final_df = (
    merged_df[[
        "user_id",
        "date",
        "age",
        "sex",
        "resting_hr",
        "rmssd",
        "calories",
        "steps",
    ]]
    .sort_values(["user_id", "date"])
    .reset_index(drop=True)
)

final_df.to_csv("masters_df2.csv", index=False)