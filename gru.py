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
# 0) GLOBAL REPRODUCIBILITY CONTROLS
# =====================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA determinism

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


# =====================================================
# 1) Load + prepare data
# =====================================================
d = pd.read_csv("masters_df_imputed.csv")
cols = ["user_id", "date", "resting_hr", "rmssd", "age", "sex", "calories", "steps"]
df = d[cols].copy()


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["user_id"] = out["user_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if out["sex"].dtype == "object":
        out["sex"] = (
            out["sex"].astype(str).str.lower()
            .map({"m": 1, "male": 1, "f": 0, "female": 0})
            .fillna(0)
            .astype(int)
        )

    # Stable ordering
    out = out.sort_values(["user_id", "date"], kind="mergesort").reset_index(drop=True)

    needed = ["user_id", "date", "rmssd", "age", "sex"]
    return out.dropna(subset=needed).reset_index(drop=True)


# =====================================================
# 2) Dataset
# =====================================================
class SeqDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


# =====================================================
# 3) Time-aware split (deterministic)
# =====================================================
def time_split_samples(df, history_days, seq_features, static_features, target_col="rmssd"):
    X_seq_tr, X_static_tr, y_tr = [], [], []
    X_seq_te, X_static_te, y_te = [], [], []

    required = ["user_id", "date", target_col] + seq_features + static_features
    df = df.dropna(subset=required)

    for uid, g in df.groupby("user_id", sort=True):
        g = g.sort_values("date", kind="mergesort").reset_index(drop=True)
        s = g.loc[0, static_features].values.astype(float)

        valid_t = list(range(history_days - 1, len(g) - 1))
        if not valid_t:
            continue

        test_t = valid_t[-1]

        for t in valid_t:
            x_window = g.loc[t - history_days + 1:t, seq_features].values.astype(float)
            y_next = float(g.loc[t + 1, target_col])

            if t == test_t:
                X_seq_te.append(x_window); X_static_te.append(s); y_te.append(y_next)
            else:
                X_seq_tr.append(x_window); X_static_tr.append(s); y_tr.append(y_next)

    return (
        np.stack(X_seq_tr), np.stack(X_static_tr), np.array(y_tr),
        np.stack(X_seq_te), np.stack(X_static_te), np.array(y_te),
    )


# =====================================================
# 4) Model
# =====================================================
class GlobalGRU(nn.Module):
    def __init__(self, seq_input_dim, static_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(seq_input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq, x_static):
        _, h_n = self.gru(x_seq)
        x = torch.cat([h_n[-1], x_static], dim=1)
        return self.head(x)


# =====================================================
# 5) Training / prediction
# =====================================================
def train_model(model, loader, lr, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.to(device)
    model.train()

    for _ in range(epochs):
        for x_seq, x_static, y in loader:
            x_seq, x_static, y = x_seq.to(device), x_static.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x_seq, x_static), y)
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys = [], []
    for x_seq, x_static, y in loader:
        x_seq, x_static = x_seq.to(device), x_static.to(device)
        preds.append(model(x_seq, x_static).cpu().numpy().ravel())
        ys.append(y.numpy().ravel())
    return np.concatenate(preds), np.concatenate(ys)


# =====================================================
# 6) Single run
# =====================================================
def run_gru_global(df, history_days, seq_features, epochs=60, batch_size=64):
    df = prepare_df(df)

    X_seq_tr, X_static_tr, y_tr, X_seq_te, X_static_te, y_te = time_split_samples(
        df, history_days, seq_features, ["age", "sex"]
    )

    # Scaling (train-only)
    seq_scaler = StandardScaler().fit(X_seq_tr.reshape(-1, X_seq_tr.shape[-1]))
    static_scaler = StandardScaler().fit(X_static_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))

    def scale_seq(X): 
        return seq_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    train_ds = SeqDataset(
        scale_seq(X_seq_tr),
        static_scaler.transform(X_static_tr),
        y_scaler.transform(y_tr.reshape(-1, 1)).ravel()
    )
    test_ds = SeqDataset(
        scale_seq(X_seq_te),
        static_scaler.transform(X_static_te),
        y_scaler.transform(y_te.reshape(-1, 1)).ravel()
    )

    g = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = GlobalGRU(seq_input_dim=len(seq_features), static_dim=2)
    train_model(model, train_loader, lr=1e-3, epochs=epochs, device="cuda" if torch.cuda.is_available() else "cpu")

    pred_s, y_s = predict(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu")
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()

    return {
        "history_days": history_days,
        "feature_set": "+".join(seq_features),
        "mae": mean_absolute_error(y_true, pred),
        "rmse": np.sqrt(mean_squared_error(y_true, pred)),  # FIXED
        "r2": r2_score(y_true, pred),
        "pearson_r": np.corrcoef(y_true, pred)[0, 1] if np.std(pred) > 0 else np.nan,
    }


# =====================================================
# 7) Run all
# =====================================================
FEATURE_SETS = [
    ["rmssd"],
    ["rmssd", "resting_hr"],
    ["rmssd", "resting_hr", "steps"],
    ["rmssd", "resting_hr", "steps", "calories"],
]

results = []
for hd in (1, 7, 14):
    for fs in FEATURE_SETS:
        results.append(run_gru_global(df, hd, fs))

out = pd.DataFrame(results).sort_values(["history_days", "feature_set"])
print(out.to_string(index=False))

