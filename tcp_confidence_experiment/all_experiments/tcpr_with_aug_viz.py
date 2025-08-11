import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from Confidence_model import ConfidenceNet
import matplotlib.pyplot as plt
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For full reproducibility (slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

data = np.load('/Users/anushka/Documents/Summer-Project-/dataset/labeled_embeddings_probs_and_labels_v2.npz')
embeddings = data['test_embeddings']
probs = data['probs']
y_true = data['y_train']
y_hat = np.argmax(probs, axis=1)
p_y_hat = np.max(probs, axis=1)
y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
p_y_star = np.sum(probs * y_true, axis=1)

def compute_tcp_star_targets(y_true_class, y_hat, p_y_star, p_y_hat, alpha):
    indicator = (y_true_class != y_hat).astype(float)
    numerator = p_y_star
    denominator = p_y_hat + (indicator * (p_y_star + alpha))
    tcp_star_targets = numerator / denominator
    return tcp_star_targets

tcp_star_targets = compute_tcp_star_targets(y_true_class, y_hat, p_y_star, p_y_hat, alpha=0.3)
all_indices = np.arange(len(embeddings))  
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
confidence_scores_tcpr= np.load('/Users/anushka/Documents/Summer-Project-/tcp_confidence_experiment/confidence_scores_tcpr_augmented_test.npy')
#load confidence scores
confidence_scores_tcpr_train= confidence_scores_tcpr[train_indices]
confidence_scores_tcpr_test = np.load('/Users/anushka/Documents/Summer-Project-/tcp_confidence_experiment/confidence_scores_tcpr_augmented_test.npy')
#plot histogram of confidence scores with success and error
is_correct = (y_hat == y_true_class)
is_correct_full = is_correct[train_indices]

success_conf = confidence_scores_tcpr_train[is_correct_full]
error_conf = confidence_scores_tcpr_train[~is_correct_full]
print(success_conf.shape, error_conf.shape)
plt.hist(success_conf, bins=20, density=True, alpha=0.7, label='Correct', color='green')
plt.hist(error_conf, bins=20, density=True, alpha=0.7, label='Error', color='red')
plt.xlabel('TCP*R Confidence')
plt.ylabel('Density')
plt.title('TCP*R Confidence: Success vs Error')
plt.legend()
plt.grid(True)  
plt.show()