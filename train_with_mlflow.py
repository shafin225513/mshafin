import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient


import mlflow
import mlflow.pytorch

# ----------------------------
# 1. Setup MLflow experiment
# ----------------------------
mlflow.set_experiment("tiny_classification_example")

# ----------------------------
# 2. Simulated data
# ----------------------------
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ----------------------------
# 3. Define tiny model
# ----------------------------
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

model = TinyNet()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# 4. Start MLflow tracking
# ----------------------------
with mlflow.start_run() as run:
    

    mlflow.log_param("lr", 0.01)
    mlflow.log_param("epochs", 5)

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss = loss_fn(test_preds, y_test)

        # Log metrics for each epoch
        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        mlflow.log_metric("test_loss", test_loss.item(), step=epoch)

    # Log the model itself
    mlflow.pytorch.log_model(model, artifact_path="model")
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, "MentalHealthClassifier")

    # Optional: add description
    client = MlflowClient()
    client.update_registered_model(
        name="MentalHealthClassifier",
        description="Tiny demo model for mental health classification."
    )