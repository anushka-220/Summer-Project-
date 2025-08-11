import numpy as np

def compute_tcp_star_targets(y_true_class, y_hat, p_y_star, p_y_hat, alpha=0.3):
    indicator = (y_true_class != y_hat).astype(float)
    numerator = p_y_star
    denominator = p_y_hat + (indicator * (p_y_star + alpha))
    return numerator / denominator