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

X= embeddings
y = tcp_star_targets
# all_indices = np.arange(len(embeddings))  

# train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
is_correct = (y_hat == y_true_class).astype(int)  # 1 = success, 0 = error

train_indices, test_indices = train_test_split(
    np.arange(len(embeddings)),
    test_size=0.2,
    random_state=42,
    stratify=is_correct  # keeps success/error ratio same
)
# Now use these to split all your arrays
X_train = embeddings[train_indices]
y_train = tcp_star_targets[train_indices]

X_test = embeddings[test_indices]
y_test = tcp_star_targets[test_indices]

# Also for classifier outputs
y_hat_train = y_hat[train_indices]
y_true_class_train = y_true_class[train_indices]
y_hat_test = y_hat[test_indices]
y_true_class_test = y_true_class[test_indices]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Determine which are errors/successes on the training split
is_correct_train = (y_hat_train == y_true_class_train)
X_train_errors = X_train_tensor[~is_correct_train]
y_train_errors = y_train_tensor[~is_correct_train]

X_train_success = X_train_tensor[is_correct_train]
y_train_success = y_train_tensor[is_correct_train]

print(f"Errors in train: {len(X_train_errors)}, Successes in train: {len(X_train_success)}")

X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
y_val_tensor = torch.tensor(y_test, dtype=torch.float32)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

from torch.utils.data import Dataset, DataLoader

class Custom_Loader(Dataset):
    def __init__(self, X_success, y_success, X_error, y_error,
                 success_ratio=0.75, batch_size=128, augment_fn=None):
        self.X_success = X_success
        self.y_success = y_success
        self.X_error = X_error
        self.y_error = y_error
        self.success_ratio = success_ratio
        self.batch_size = batch_size
        self.augment_fn = augment_fn

        self.n_success = int(self.batch_size * self.success_ratio)
        self.n_error = self.batch_size - self.n_success
        self.num_batches = min(
            len(self.X_success) // self.n_success,
            len(self.X_error) // self.n_error
        )

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        success_idx = torch.randperm(len(self.X_success))[:self.n_success]
        error_idx = torch.randperm(len(self.X_error))[:self.n_error]

        Xb = torch.cat([self.X_success[success_idx], self.X_error[error_idx]], dim=0)
        yb = torch.cat([self.y_success[success_idx], self.y_error[error_idx]], dim=0)

        perm = torch.randperm(len(Xb))
        Xb = Xb[perm]
        yb = yb[perm]

        if self.augment_fn:
            Xb = self.augment_fn(Xb)

        return Xb, yb

train_dataset = Custom_Loader(
    X_train_success, y_train_success,
    X_train_errors, y_train_errors,
    success_ratio=0.75,
    batch_size=128,
    #augment_fn=add_gaussian_noise  
)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=False)
model = ConfidenceNet(input_dim=X_train.shape[1])
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.MSELoss()

for epoch in range(500):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        #print number of success and error samples in the batch
        print(f"Batch size: {len(xb)}, Success samples: {len(yb[yb > 0.5])}, Error samples: {len(yb[yb <= 0.5])}")
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/len(train_loader):.6f}")


model.eval()
with torch.no_grad():
    Xall_tensor = torch.tensor(embeddings, dtype=torch.float32)
    confidence_scores_tcpr_augmented = model(Xall_tensor).cpu().numpy() 
#save these scores
np.save('/Users/anushka/Documents/Summer-Project-/tcp_confidence_experiment/confidence_scores_tcpr_augmented_forvisual.npy', confidence_scores_tcpr_augmented)   
# Save the confidence scores
is_correct = (y_hat == y_true_class)
is_correct_full = is_correct[test_indices]

success_conf = confidence_scores_tcpr_augmented[is_correct_full]
error_conf = confidence_scores_tcpr_augmented[~is_correct_full]
print(success_conf.shape, error_conf.shape)
plt.hist(success_conf, bins=20, density=True, alpha=0.7, label='Correct', color='green')
plt.hist(error_conf, bins=20, density=True, alpha=0.7, label='Error', color='red')
plt.xlabel('TCP*R Confidence')
plt.ylabel('Density')
plt.title('TCP*R Confidence: Success vs Error')
plt.legend()
plt.grid(True)  
plt.show()

