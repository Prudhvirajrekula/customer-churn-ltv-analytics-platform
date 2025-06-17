# test_multitask_model.py
import torch
import joblib
import numpy as np

class MultiTaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        self.churn_head = torch.nn.Linear(16, 1)
        self.ltv_head = torch.nn.Linear(16, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.churn_head(shared), self.ltv_head(shared)

# Load model
model = MultiTaskModel()
model.load_state_dict(torch.load("multitask_model.pt", map_location=torch.device("cpu")))
model.eval()

# Load scaler
scaler = joblib.load("scaler.pkl")

# Sample input
input_data = np.array([[4, 1240.0, 18]])  # [order_count, lifetime_value, days_since_last_order]
scaled_input = scaler.transform(input_data)
input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

# Predict
with torch.no_grad():
    churn_logit, ltv_pred = model(input_tensor)
    churn_prob = torch.sigmoid(churn_logit).item()
    ltv_value = ltv_pred.item()

churn_label = "Churned" if churn_prob >= 0.5 else "Not Churned"
print(f"Churn Prediction: {churn_label} (Confidence: {churn_prob:.2f})")
print(f"Predicted Lifetime Value: ${ltv_value:.2f}")
