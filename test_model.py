import torch
import joblib
import numpy as np

# Define the same model class
class ChurnModel(torch.nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)  # raw logits

# Load model and scaler
model = torch.load("churn_model.pt", map_location=torch.device("cpu"), weights_only=False)
model.eval()
scaler = joblib.load("scaler.pkl")

# Test input: all zeros
sample = np.array([[0, 0, 0]])
scaled = scaler.transform(sample)
tensor = torch.tensor(scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    logit = model(tensor)
    prob = torch.sigmoid(logit).item()

print(f"🧠 Final Confidence for all-zero input: {prob:.4f}")
