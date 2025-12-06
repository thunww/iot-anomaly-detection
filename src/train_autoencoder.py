import torch
import torch.nn as nn
import numpy as np
from model import DeepAE

device = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load("data/processed/train_features.npy")
y = np.load("data/processed/train_labels.npy")

X_normal = X[y == 0]

X_t = torch.tensor(X_normal, dtype=torch.float32).to(device)

input_dim = X.shape[1]
model = DeepAE(input_dim=input_dim, latent_dim=32).to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("[INFO] Training Autoencoder...")

for epoch in range(120):
    out, _ = model(X_t)
    loss = loss_fn(out, X_t)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/120 | Loss={loss.item():.6f}")

torch.save(model.state_dict(), "models/autoencoder.pt")
print("[OK] Saved AE model")
