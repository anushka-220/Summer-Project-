# plot_confidence_success_vs_error.py

import numpy as np
import matplotlib.pyplot as plt

# Load raw data
data = np.load('/Users/anushka/Documents/Summer-Project-/dataset/labeled_embeddings_probs_and_labels_v2.npz')
probs = data['probs']
y_true = data['y_train']
y_hat = np.argmax(probs, axis=1)
y_true_class = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

# Load saved confidence scores
confidence_scores = np.load('tcp_confidence_experiment/confidence_scores.npy')  # from eval_model.py

# Determine correctness
is_correct = (y_hat == y_true_class)
conf_correct = confidence_scores[is_correct]
conf_errors = confidence_scores[~is_correct]

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(conf_correct, bins=30, alpha=0.7, label='Correct Predictions', color='green')
plt.hist(conf_errors, bins=30, alpha=0.7, label='Errors', color='red')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.title('Confidence Score Distribution: Correct vs Errors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()