import numpy as np

def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """
    # Formula: sqrt( sum((y_true - y_pred)^2) / N ) 
    return np.sqrt(np.mean((y_true - y_pred) ** 2))