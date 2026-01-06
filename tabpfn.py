from huggingface_hub import login
login("hf_xxx")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

df = pd.read_csv('masters_df2.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the regressor
regressor = TabPFNRegressor()  # Uses TabPFN-2.5 weights, trained on synthetic data only.
# To use TabPFN v2:
# regressor = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
mae = mean_absolute_error(y_test, predictions)
pearson_r, p_value = pearsonr(y_test, predictions)

print("R² Score:", r2)
print(f"RMSE      : {rmse:.4f}")
print(f"Pearson r : {pearson_r:.4f}")
print(f"p-value   : {p_value:.4e}")
