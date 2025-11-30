import numpy as np

def tanimoto_similarity(x, y):
    """
    Calculates the Tanimoto distance between x and samples in y.
    Arguments:
    x: array-like, shape (n_features,). Single binary sample.
    y: array-like, shape (n_samples, n_features). Multiple binary samples.
    """
    # Converter para numpy arrays para garantir operações vetoriais
    x = np.array(x)
    y = np.array(y)
    
    # Produto escalar (interseção para vetores binários) -> a.b
    xy_dot = np.dot(y, x)
    
    # Soma dos quadrados (contagem de 1s para binário) -> ||a||^2 e ||b||^2
    # Para vetores binários x*x é o mesmo que sum(x)
    x_sum = np.sum(x) # a^2
    y_sum = np.sum(y, axis=1) # b^2
    
    # Fórmula: (a.b) / (a^2 + b^2 - a.b)
    denominator = x_sum + y_sum - xy_dot
    
    # Evitar divisão por zero
    similarity = xy_dot / denominator
    
    return similarity