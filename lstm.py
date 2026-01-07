import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 0) REPRODUCIBILITY CONTROLS
# =====================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# =====================================================
# 1) Load data
# =====================================================
d = pd.read_csv("masters_df_imputed.csv")
cols = ["user_id", "date", "resting_hr", "rmssd", "age", "sex", "calories", "steps"]
df = d[cols].copy()


# =====================================================
# 2) Prep: cleaning + encoding (stable)
# =====================================================
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["user_id"] = out["user_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Encode sex if it's not numeric already
    if "sex" in out.columns and (out["sex"].dtype == "object" or str(out["sex"].dtype).startswith("string")):
        out["sex"] = (
            out["sex"].astype(str).str.strip().str.lower()
            .map({"m": 1, "male": 1, "f": 0, "female": 0})
            .fillna(0)
            .astype(int)
        )
    elif "sex" in out.columns:
        out["sex"] = pd.to_numeric(out["sex"], errors="coerce").fillna(0).astype(int)

    # Stable ordering
    out = out.sort_values(["user_id", "date"], kind="mergesort").reset_index(drop=True)

    # Drop rows with missing essentials
    needed = ["user_id", "date", "rmssd", "age", "sex"]
    out = out.dropna(subset=needed).reset_index(drop=True)
    return out


# =====================================================
# 3) Dataset
# =====================================================
class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, X_static: np.ndarray, y: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


# =====================================================
# 4) Time-aware split (deterministic)
# =====================================================
def time_split_indices(
    df: pd.DataFrame,
    history_days: int,
    seq_features: list,
    static_features: list,
    target_col: str = "rmssd",
):
    X_seq_tr, X_static_tr, y_tr = [], [], []
    X_seq_te, X_static_te, y_te = [], [], []

    required_cols = ["user_id", "date", target_col] + seq_features + static_features
    df_run = df.dropna(subset=required_cols).copy()

    # Deterministic user iteration order
    for uid, g in df_run.groupby("user_id", sort=True):
        g = g.sort_values("date", kind="mergesort").reset_index(drop=True)

        # static features from the first row (matches your original logic)
        s = g.loc[0, static_features].values.astype(float)

        valid_t = list(range(history_days - 1, len(g) - 1))
        if not valid_t:
            continue

        test_t = valid_t[-1]

        for t in valid_t:
            x_window = g.loc[t - history_days + 1 : t, seq_features].values.astype(float)
            y_next = float(g.loc[t + 1, target_col])

            if t == test_t:
                X_seq_te.append(x_window); X_static_te.append(s); y_te.append(y_next)
            else:
                X_seq_tr.append(x_window); X_static_tr.append(s); y_tr.append(y_next)

    if len(X_seq_tr) == 0 or len(X_seq_te) == 0:
        raise ValueError(
            f"No samples created. Check missingness or history_days={history_days} for chosen features={seq_features}."
        )

    return (
        np.stack(X_seq_tr, axis=0),
        np.stack(X_static_tr, axis=0),
        np.array(y_tr, dtype=float),
        np.stack(X_seq_te, axis=0),
        np.stack(X_static_te, axis=0),
        np.array(y_te, dtype=float),
    )


# =====================================================
# 5) Model: LSTM + static head
# =====================================================
class GlobalLSTM(nn.Module):
    def __init__(self, seq_input_dim: int, static_dim: int,
                 hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_static):
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]
        x = torch.cat([h_last, x_static], dim=1)
        return self.head(x)


# =====================================================
# 6) Train + predict utilities
# =====================================================
def train_model(model, train_loader, lr=1e-3, epochs=30, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for x_seq, x_static, y in train_loader:
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            y = y.to(device)

            pred = model(x_seq, x_static)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    model.to(device)
    preds, ys = [], []
    for x_seq, x_static, y in loader:
        x_seq = x_seq.to(device)
        x_static = x_static.to(device)
        pred = model(x_seq, x_static).cpu().numpy().ravel()
        preds.append(pred)
        ys.append(y.numpy().ravel())
    return np.concatenate(preds), np.concatenate(ys)


# =====================================================
# 7) One run
# =====================================================
def run_lstm_global(
    df: pd.DataFrame,
    history_days: int,
    seq_features: list,
    static_features: list = None,
    target_col="rmssd",
    hidden_dim=64,
    epochs=40,
    batch_size=64,
    lr=1e-3,
    device=None,
    seed=SEED,
):
    if static_features is None:
        static_features = ["age", "sex"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Re-seed at the start of every run (so each run is reproducible independent of loop order)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    df = prepare_df(df)

    X_seq_tr, X_static_tr, y_tr, X_seq_te, X_static_te, y_te = time_split_indices(
        df, history_days, seq_features, static_features, target_col=target_col
    )

    # ---- Scale train-only ----
    seq_scaler = StandardScaler()
    static_scaler = StandardScaler()
    y_scaler = StandardScaler()

    N, L, F = X_seq_tr.shape
    seq_scaler.fit(X_seq_tr.reshape(N * L, F))
    static_scaler.fit(X_static_tr)
    y_scaler.fit(y_tr.reshape(-1, 1))

    X_seq_tr_s = seq_scaler.transform(X_seq_tr.reshape(N * L, F)).reshape(N, L, F)
    X_seq_te_s = seq_scaler.transform(X_seq_te.reshape(X_seq_te.shape[0] * L, F)).reshape(X_seq_te.shape[0], L, F)

    X_static_tr_s = static_scaler.transform(X_static_tr)
    X_static_te_s = static_scaler.transform(X_static_te)

    y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()
    y_te_s = y_scaler.transform(y_te.reshape(-1, 1)).ravel()

    train_ds = SeqDataset(X_seq_tr_s, X_static_tr_s, y_tr_s)
    test_ds  = SeqDataset(X_seq_te_s, X_static_te_s, y_te_s)

    # Deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,   # important for determinism
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    model = GlobalLSTM(seq_input_dim=F, static_dim=X_static_tr_s.shape[1], hidden_dim=hidden_dim)

    train_model(model, train_loader, lr=lr, epochs=epochs, device=device)

    pred_s, y_s = predict(model, test_loader, device=device)

    # Inverse transform to original scale
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()

    # Metrics
    mae = float(mean_absolute_error(y_true, pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))  # FIXED: true RMSE
    r2 = float(r2_score(y_true, pred))

    # Pearson r (robust)
    if np.std(y_true) == 0 or np.std(pred) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(y_true, pred)[0, 1])

    return {
        "history_days": int(history_days),
        "feature_set": "+".join(seq_features),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": r,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


# =====================================================
# 8) Run grid
# =====================================================
FEATURE_SETS = [
    ["rmssd"],
    ["rmssd", "resting_hr"],
    ["rmssd", "resting_hr", "steps"],
    ["rmssd", "resting_hr", "steps", "calories"],
]

def run_all_windows_and_features(df: pd.DataFrame, history_days_list=(1, 7, 14), epochs=60, seed=SEED):
    results = []
    for hd in history_days_list:
        for seq_features in FEATURE_SETS:
            results.append(
                run_lstm_global(
                    df,
                    history_days=hd,
                    seq_features=seq_features,
                    epochs=epochs,
                    seed=seed
                )
            )

    out = (
        pd.DataFrame(results)
        .sort_values(["history_days", "feature_set"], kind="mergesort")
        .reset_index(drop=True)
    )

    print(out[["history_days", "feature_set", "n_train", "n_test", "mae", "rmse", "r2", "pearson_r"]].to_string(index=False))
    return out


# Example usage:
run_all_windows_and_features(df)
