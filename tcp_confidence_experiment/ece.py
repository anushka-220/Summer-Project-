import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tcp_utils import compute_tcp_star_targets
from model import ConfidenceNet
import torch
from calibration import default_ECE, inverse_weights_ECE, no_weights_ECE
ece_dict = {
    'Default ECE': [],
    'Inverse-Weighted ECE': [],
    'Unweighted ECE': []
}


data = np.load('/Users/anushka/Documents/Summer-Project-/dataset/labeled_embeddings_probs_and_labels_v2.npz')
embeddings = data['test_embeddings']
probs = data['probs']
y_true = data['y_train']
y_hat = np.argmax(probs, axis=1)
p_y_hat = np.max(probs, axis=1)
p_y_star = np.sum(probs * y_true, axis=1)
y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

tcp_star_targets = compute_tcp_star_targets(y_true_class, y_hat, p_y_star, p_y_hat)
all_indices = np.arange(len(embeddings))  
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
X_train = embeddings[train_indices]
y_train = tcp_star_targets[train_indices]

X_test = embeddings[test_indices]
y_test = tcp_star_targets[test_indices]

# Also for classifier outputs
y_hat_train = y_hat[train_indices]
y_true_class_train = y_true_class[train_indices]
is_correct = (y_hat_train == y_true_class_train)

X_errors = X_train[~is_correct]
y_errors = y_train[~is_correct]

X_success = X_train[is_correct]
y_success = y_train[is_correct]



confidence_scores_tcpr= np.load('tcp_confidence_experiment/confidence_scores.npy')



is_correct_test = y_hat[test_indices] == y_true_class[test_indices]

test_conf = confidence_scores_tcpr[test_indices]
test_y_hat = y_hat[test_indices]
test_y_true = y_true_class[test_indices]

# Only get test set errors
conf_eval_errors = test_conf[~is_correct_test]
y_hat_eval_errors = test_y_hat[~is_correct_test]
y_true_eval_errors = test_y_true[~is_correct_test]

y_hat_test = y_hat[test_indices]  # predicted class from classifier
y_true_test = y_true_class[test_indices]  # true class label

is_correct = (y_hat_test == y_true_test)
conf_correct = test_conf[is_correct]
conf_error = test_conf[~is_correct]

y_hat_correct = y_hat_test[is_correct]
y_hat_error = y_hat_test[~is_correct]

y_true_correct = y_true_test[is_correct]
y_true_error = y_true_test[~is_correct]

is_correct_train = y_hat[train_indices] == y_true_class[train_indices]

train_conf = confidence_scores_tcpr[train_indices]
train_y_hat = y_hat[train_indices]
train_y_true = y_true_class[train_indices]

conf_train_errors = train_conf[~is_correct_train]
conf_train_success = train_conf[is_correct_train]

y_hat_train_errors = train_y_hat[~is_correct_train]
y_hat_train_success = train_y_hat[is_correct_train]

y_true_train_errors = train_y_true[~is_correct_train]
y_true_train_success = train_y_true[is_correct_train]



error_counts = []
ece_values = []
# Start config: 200 errors + 4500 successes from training set
max_steps = min(len(conf_eval_errors), 150)
print("max steps=", max_steps)  # control based on available eval errors
step = 10

for i in range(0, max_steps + 1, step):
    num_added_errors = i
    num_removed_successes = i

    # Select base set
    train_errors = np.arange(200)  # first 200 errors from training
    train_success = np.arange(len(conf_train_success))
    sample_size = min(4500 - num_removed_successes, len(train_success))
    print (f"Sample size: {sample_size}")
    selected_successes = np.random.choice(train_success, size=sample_size, replace=False)

    # Add i errors from evaluation set
    eval_errors = np.arange(num_added_errors)

    # Final confidence
    conf_subset = np.concatenate([
        conf_train_errors[train_errors],
        conf_train_success[selected_successes],
        conf_eval_errors[eval_errors]
    ])

    y_hat_subset = np.concatenate([
        y_hat_train_errors[train_errors],
        y_hat_train_success[selected_successes],
        y_hat_eval_errors[eval_errors]
    ])

    y_true_subset = np.concatenate([
        y_true_train_errors[train_errors],
        y_true_train_success[selected_successes],
        y_true_eval_errors[eval_errors]
    ])

    # Shuffle
    idx = np.random.permutation(len(conf_subset))
    conf_subset = conf_subset[idx]
    y_hat_subset = y_hat_subset[idx]
    y_true_subset = y_true_subset[idx]

   # Compute all three ECEs
    ece_default = default_ECE(conf_subset, y_hat_subset, y_true_subset)
    ece_inverse = inverse_weights_ECE(conf_subset, y_hat_subset, y_true_subset)
    ece_unweighted = no_weights_ECE(conf_subset, y_hat_subset, y_true_subset)

    ece_dict['Default ECE'].append(ece_default)
    ece_dict['Inverse-Weighted ECE'].append(ece_inverse)
    ece_dict['Unweighted ECE'].append(ece_unweighted)
    error_counts.append(i)

    print(f"Errors: {i} | Successes: {4500 - i} | ECE Default: {ece_default:.4f} | Inverse: {ece_inverse:.4f} | No Weights: {ece_unweighted:.4f}")


# scaler = MinMaxScaler()
# normalized_ece_dict = {
#     label: scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
#     for label, values in ece_dict.items()
# }

# plt.figure(figsize=(10, 6))
# for label, values in normalized_ece_dict.items():
#     plt.plot(error_counts, values, marker='o', label=f'{label} (normalized)')

# plt.xlabel('Number of Evaluation Errors Added')
# plt.ylabel('Normalized ECE')
# plt.title('Normalized ECE vs Evaluation Errors (Three Estimators)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt

# Example: replace this with your real data
# inverse_ece_values = [...]  # e.g. from ece_dict['Inverse-Weighted ECE']
# step = 10  # Step size used when adding eval errors

# If using stored dict:
# inverse_ece_values = ece_dict['Unweighted ECE']
# step = 10

# # Create x-axis: number of evaluation errors added
# error_steps = list(range(0, len(inverse_ece_values) * step, step))

# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(error_steps, inverse_ece_values, marker='o', color='orange', label='Inverse-Weighted ECE')

# plt.xlabel("Number of Evaluation Errors Added")
# plt.ylabel("Expected Calibration Error (ECE)")
# plt.title("UnWeighted ECE vs Evaluation Errors")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

#plot all 3 ece values in one plot

plt.figure(figsize=(14, 8), dpi=120)
for label, values in ece_dict.items():
    plt.plot(error_counts, values, marker='o', label=label, linewidth=2)
plt.xlabel('Number of Evaluation Errors Added', fontsize=16)
plt.ylabel('ECE', fontsize=16)
plt.title('ECE vs Evaluation Errors (Three Estimators)', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

is_correct_full = (y_hat == y_true_class)
conf_success= confidence_scores_tcpr[is_correct_full]
conf_errors = confidence_scores_tcpr[~is_correct_full]
#plot histogram of confidence scores for errors and successes
plt.figure(figsize=(10, 6))
plt.hist(conf_success, bins=30, alpha=0.7, label='Successes', color='green', density=True)
plt.hist(conf_errors, bins=30, alpha=0.7, label='Errors', color='red', density=True)
plt.xlabel('Confidence Score')
plt.ylabel('Density')
plt.title('Confidence Score Density: Success vs Error')
plt.legend()
plt.grid(True)
plt.show()

