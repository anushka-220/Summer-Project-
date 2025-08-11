import numpy as np
import torch
from model import ConfidenceNet

# Load inputs and model
data = np.load('/Users/anushka/Documents/Summer-Project-/dataset/labeled_embeddings_probs_and_labels_v2.npz')
embeddings = data['test_embeddings']
input_dim = embeddings.shape[1]

model = ConfidenceNet(input_dim)
model.load_state_dict(torch.load('tcp_confidence_experiment/confidence_net.pth'))
model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(embeddings, dtype=torch.float32)
    confidence_scores = model(X_tensor).cpu().numpy()

np.save('tcp_confidence_experiment/confidence_scores.npy', confidence_scores)