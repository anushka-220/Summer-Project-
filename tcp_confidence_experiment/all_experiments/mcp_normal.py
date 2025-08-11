import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

data = np.load('/Users/anushka/Documents/Summer-Project-/dataset/labeled_embeddings_probs_and_labels_v2.npz')
embeddings = data['test_embeddings']
probs = data['probs']
y_true = data['y_train']
y_hat = np.argmax(probs, axis=1)
p_y_hat = np.max(probs, axis=1)
y_true = np.argmax(data['y_train'], axis=1) 
mcp_confidence_scores = p_y_hat
for i in range(5):
    print(f"Sample {i} | True: {y_true[i]}, Pred: {y_hat[i]}, MCP: {mcp_confidence_scores[i]:.4f}")

# Separate successes and errors
success_conf = mcp_confidence_scores[y_hat == y_true]
error_conf = mcp_confidence_scores[y_hat != y_true]

# Plot histograms
plt.figure(figsize=(8, 5))
plt.hist(success_conf, bins=20, alpha=0.6,density=True, label='Successes', color='green')
plt.hist(error_conf, bins=20, alpha=0.6, density= True, label='Errors', color='red')

plt.title('MCP Confidence Histogram: Successes vs Errors')
plt.xlabel('Confidence Score (MCP)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

correct = (y_hat == y_true).astype(int)
prob_true, prob_pred = calibration_curve(correct, mcp_confidence_scores, n_bins=15, strategy='uniform')

# Plot reliability curve
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Reliability Curve (Calibration Curve)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()