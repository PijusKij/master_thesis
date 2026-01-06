import pandas as pd
from tabpfn import TabPFNRegressor
import torch
from huggingface_hub import login
login("hf_x")

df = pd.read_csv('masters_df2.csv')

# Split
df_train   = df[df["rmssd"].notna()].copy()
df_missing = df[df["rmssd"].isna()].copy()

X_train   = df_train[feature_cols].astype(float)
y_train   = df_train["rmssd"]
X_missing = df_missing[feature_cols].astype(float)

# Device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)

# Train TabPFN
model = TabPFNRegressor(device=device)
model.fit(X_train, y_train)

# Predict missing rmssd
preds = model.predict(X_missing)
df_missing.loc[:, "rmssd"] = preds

df_imputed = pd.concat([df_train, df_missing]).sort_index()

final_df.to_csv("masters_df_imputed.csv", index=False)