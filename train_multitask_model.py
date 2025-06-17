import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

# Load and balance dataset
df = pd.read_csv("features_customer_churn_ltv.csv")

df_majority = df[df["is_churned"] == 1]
df_minority = df[df["is_churned"] == 0]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

# Preprocess
features = ["order_count", "lifetime_value", "days_since_last_order"]
X = df_balanced[features].fillna(0).values
y_churn = df_balanced["is_churned"].values
y_ltv = df_balanced["lifetime_value"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_churn_train, y_churn_val, y_ltv_train, y_ltv_val = train_test_split(
    X_scaled, y_churn, y_ltv, test_size=0.2, random_state=42
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_churn_train_tensor = torch.tensor(y_churn_train.reshape(-1, 1), dtype=torch.float32)
y_ltv_train_tensor = torch.tensor(y_ltv_train.reshape(-1, 1), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_churn_val_tensor = torch.tensor(y_churn_val.reshape(-1, 1), dtype=torch.float32)
y_ltv_val_tensor = torch.tensor(y_ltv_val.reshape(-1, 1), dtype=torch.float32)

# Multi-task Model
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.churn_head = nn.Linear(16, 1)
        self.ltv_head = nn.Linear(16, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.churn_head(shared), self.ltv_head(shared)

model = MultiTaskModel()

# Loss & Optimizer
pos_weight = torch.tensor([(len(y_churn_train) - sum(y_churn_train)) / sum(y_churn_train)], dtype=torch.float32)
loss_fn_churn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_fn_ltv = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train Loop
for epoch in range(40):
    model.train()
    optimizer.zero_grad()

    churn_pred, ltv_pred = model(X_train_tensor)
    loss = loss_fn_churn(churn_pred, y_churn_train_tensor) + 0.5 * loss_fn_ltv(ltv_pred, y_ltv_train_tensor)

    loss.backward()
    optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        val_churn_pred, val_ltv_pred = model(X_val_tensor)
        val_acc = ((torch.sigmoid(val_churn_pred) >= 0.5) == y_churn_val_tensor).float().mean().item()
        val_ltv_rmse = torch.sqrt(nn.functional.mse_loss(val_ltv_pred, y_ltv_val_tensor)).item()

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}, Val LTV RMSE={val_ltv_rmse:.2f}")

# Save model and scaler
torch.save(model.state_dict(), "multitask_model.pt")
joblib.dump(scaler, "scaler.pkl")
print("✅ Multi-task model and scaler saved.")
