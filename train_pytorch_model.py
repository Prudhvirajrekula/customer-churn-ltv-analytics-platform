import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

# Step 1: Load and balance dataset
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
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)  # shuffle

# Step 2: Preprocess
X = df_balanced[["order_count", "lifetime_value", "days_since_last_order"]].fillna(0).values
y = df_balanced["is_churned"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)

# Step 3: Define Model
class ChurnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)  # no sigmoid
        )

    def forward(self, x):
        return self.fc(x)  # output logits

model = ChurnModel()
pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 4: Train
for epoch in range(40):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor)
        val_probs = torch.sigmoid(val_logits)
        val_preds = (val_probs >= 0.5).float()
        val_acc = (val_preds == y_val_tensor).float().mean().item()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Val Acc = {val_acc:.4f}")

# Step 5: Save model and scaler
torch.save(model, "churn_model.pt")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully.")
