import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/cleaned_laptop_data.csv")
X = df.drop("Price", axis=1)
y = df["Price"]

# Categorical and numerical features
categorical = ["Company", "TypeName", "Cpu brand", "Gpu brand", "os"]
numerical = ["Ram", "Weight", "TouchScreen", "IPS", "ppi", "HDD", "SSD"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first", sparse_output=False), categorical)],
    remainder="passthrough"
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=30, random_state=42, verbosity=0),
}

# Directories
model_dir = "model"
plot_dir = os.path.join(model_dir, "plots")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Step 1: Train and collect results
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Save model
    model_path = os.path.join(model_dir, f"{name.replace(' ', '_')}.joblib")
    joblib.dump(pipeline, model_path)

    # Save prediction and error for plotting
    results[name] = {
        "y_test": y_test,
        "y_pred": y_pred,
        "error": np.abs(y_test - y_pred)
    }

# Step 2: Plot all in one image
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # shared colorbar space

for idx, (name, res) in enumerate(results.items()):
    sc = axs[idx].scatter(res["y_test"], res["y_pred"], c=res["error"], cmap="coolwarm", alpha=0.7)
    axs[idx].plot([res["y_test"].min(), res["y_test"].max()],
                  [res["y_test"].min(), res["y_test"].max()], 'k--')
    axs[idx].set_title(name)
    axs[idx].set_xlabel("Actual")
    axs[idx].set_ylabel("Predicted")

# Hide unused subplot
for j in range(len(results), len(axs)):
    fig.delaxes(axs[j])

# Shared colorbar
fig.colorbar(sc, cax=cbar_ax).set_label("Absolute Error")

# Save combined plot
plt.tight_layout(rect=[0, 0, 0.9, 1])
combined_path = os.path.join(plot_dir, "all_models_scatter.png")
plt.savefig(combined_path, dpi=300)
plt.close()

print("✅ All models trained and combined plot saved.")
