import numpy as np

def tanimoto_similarity(x, y):
    """
    Calculates the Tanimoto distance between x and samples in y.
    Arguments:
    x: array-like, shape (n_features,). Single binary sample.
    y: array-like, shape (n_samples, n_features). Multiple binary samples.
    """
    # Convert to numpy arrays to ensure vector operations
    x = np.array(x)
    y = np.array(y)
    
    # Dot product (intersection for binary vectors) -> a.b
    xy_dot = np.dot(y, x)
    
    # Sum of squares (count of 1s for binary) -> ||a||^2 and ||b||^2
    # For binary vectors x*x is the same as sum(x)
    x_sum = np.sum(x) # a^2
    y_sum = np.sum(y, axis=1) # b^2
    
    # Formula: (a.b) / (a^2 + b^2 - a.b)
    denominator = x_sum + y_sum - xy_dot
    similarity = xy_dot / denominator
    
    return similarity