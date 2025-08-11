import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
import torch.nn as nn
from model import ConfidenceNet
from tcp_utils import compute_tcp_star_targets

data = np.load('/Users/anushka/Documents/Summer-Project-/dataset/labeled_embeddings_probs_and_labels_v2.npz')
embeddings = data['test_embeddings']
probs = data['probs']
y_true = data['y_train']

y_hat = np.argmax(probs, axis=1)
p_y_hat = np.max(probs, axis=1)
p_y_star = np.sum(probs * y_true, axis=1)
y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
tcp_star_targets = compute_tcp_star_targets(y_true_class, y_hat, p_y_star, p_y_hat)

from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(np.arange(len(embeddings)), test_size=0.2, random_state=42)

X_train, y_train = embeddings[train_idx], tcp_star_targets[train_idx]
X_val, y_val = embeddings[test_idx], tcp_star_targets[test_idx]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float()), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float()), batch_size=64)

model = ConfidenceNet(input_dim=embeddings.shape[1])
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.MSELoss()

for epoch in range(5000):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/len(train_loader):.6f}")

# Save model
torch.save(model.state_dict(), 'tcp_confidence_experiment/confidence_net.pth')