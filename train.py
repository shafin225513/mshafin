import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Access them
lr = params["train"]["learning_rate"]
epochs = params["train"]["epochs"]
batch_size = params["train"]["batch_size"]

# Use these variables in your model code
print(f"Training with lr={lr}, epochs={epochs}, batch={batch_size}")


# Load CSV
df = pd.read_csv("data.csv",encoding='ISO-8859-1')

# Preprocess: text to bag-of-words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Dataset and Dataloader
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SimpleDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

# Simple Binary Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

model = SimpleClassifier(input_dim=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr)

# Training Loop
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.3f}")

# Save the model
torch.save(model.state_dict(), "model.pt")
print("✅ Model trained and saved as model.pt")
