import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the imputed dataset
d = pd.read_csv('masters_df_imputed.csv')

cols = ["user_id", "date", "resting_hr", "rmssd", "age", "sex", "calories", "steps"]
df = d[cols]

# -----------------------------
# 1) Prep: cleaning + encoding
# -----------------------------
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)

    if df["sex"].dtype == "object":
        df["sex"] = (
            df["sex"].astype(str).str.lower()
            .map({"m": 1, "male": 1, "f": 0, "female": 0})
            .fillna(0).astype(int)
        )

    #if df["weekday"].dtype == "object":
    #    wd_map = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
    #    df["weekday"] = df["weekday"].astype(str).str.lower().map(wd_map)

    #df["weekday"] = df["weekday"].astype(int)

    # Only require core cols here; per-run required columns are enforced later
    needed = ["user_id","date","rmssd","age","sex"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


# -----------------------------------------
# 2) Build sequences + time-aware split
# -----------------------------------------
class SeqDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, X_static: np.ndarray, y: np.ndarray):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


def time_split_samples(df: pd.DataFrame, history_days: int,
                       seq_features: list, static_features: list,
                       target_col: str = "rmssd"):
    """
    For each user: put their LAST possible sample in test, earlier samples in train.
    Enforces required columns for this particular (seq_features/static_features) run.
    """
    X_seq_tr, X_static_tr, y_tr = [], [], []
    X_seq_te, X_static_te, y_te = [], [], []

    required_cols = ["user_id", "date", target_col] + seq_features + static_features
    df_run = df.dropna(subset=required_cols).copy()

    for uid, g in df_run.groupby("user_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
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


# -----------------------------
# 3) Model: GRU + static head
# -----------------------------
class GlobalGRU(nn.Module):
    def __init__(self, seq_input_dim: int, static_dim: int,
                 hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
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
        out, h_n = self.gru(x_seq)
        h_last = h_n[-1]
        x = torch.cat([h_last, x_static], dim=1)
        return self.head(x)


# -----------------------------
# 4) Train + predict utilities
# -----------------------------
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

            opt.zero_grad()
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


# -----------------------------
# 5) One run for a given window + feature set
# -----------------------------
def run_gru_global(df: pd.DataFrame, history_days: int,
                   seq_features: list,
                   static_features: list = None,
                   target_col="rmssd",
                   hidden_dim=64, epochs=60, batch_size=64, lr=1e-3,
                   device=None, seed=42):
    if static_features is None:
        static_features = ["age", "sex"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = prepare_df(df)

    X_seq_tr, X_static_tr, y_tr, X_seq_te, X_static_te, y_te = time_split_samples(
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

    train_loader = DataLoader(SeqDataset(X_seq_tr_s, X_static_tr_s, y_tr_s),
                              batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(SeqDataset(X_seq_te_s, X_static_te_s, y_te_s),
                             batch_size=batch_size, shuffle=False, drop_last=False)

    model = GlobalGRU(seq_input_dim=F, static_dim=X_static_tr_s.shape[1], hidden_dim=hidden_dim)

    train_model(model, train_loader, lr=lr, epochs=epochs, device=device)

    pred_s, y_s = predict(model, test_loader, device=device)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(y_s.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, pred)
    rmse = mean_squared_error(y_true, pred)  # true RMSE
    r2 = r2_score(y_true, pred)

    # Pearson r
    try:
        from scipy.stats import pearsonr
        pearson_r, _ = pearsonr(y_true, pred)
    except Exception:
        if np.std(y_true) == 0 or np.std(pred) == 0:
            pearson_r = np.nan
        else:
            pearson_r = float(np.corrcoef(y_true, pred)[0, 1])

    return {
        "history_days": history_days,
        "feature_set": "+".join(seq_features),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


# -----------------------------
# 6) Run for 1/7/14 × feature sets
# -----------------------------
FEATURE_SETS = [
    ["rmssd"],
    ["rmssd", "resting_hr"],
    ["rmssd", "resting_hr", "steps"],
    ["rmssd", "resting_hr", "steps", "calories"],
]

def run_all_windows_gru(df: pd.DataFrame, history_days_list=(1, 7, 14), epochs=60):
    results = []
    for hd in history_days_list:
        for seq_features in FEATURE_SETS:
            results.append(
                run_gru_global(df, history_days=hd, seq_features=seq_features, epochs=epochs)
            )

    out = pd.DataFrame(results).sort_values(["history_days", "feature_set"])
    print(out[["history_days", "feature_set", "n_train", "n_test", "mae", "rmse", "r2", "pearson_r"]].to_string(index=False))


# Example:
run_all_windows_gru(df)
