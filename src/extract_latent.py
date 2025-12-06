import torch
import numpy as np
from model import DeepAE

device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = np.load("data/processed/train_features.npy")
X_test  = np.load("data/processed/test_features.npy")

input_dim = X_train.shape[1]

model = DeepAE(input_dim=input_dim, latent_dim=64).to(device)

model.load_state_dict(torch.load("models/autoencoder.pt", map_location=device))
model.eval()

def get_latent(X):
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, z = model(X_t)
    return z.cpu().numpy()

z_train = get_latent(X_train)
z_test  = get_latent(X_test)

np.save("data/processed/z_train.npy", z_train)
np.save("data/processed/z_test.npy", z_test)

print("[OK] Saved latent vectors:", z_train.shape, z_test.shape)
