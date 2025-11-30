import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    def __init__(self, n_components):
        """
        PCA implementation using eigenvalue decomposition.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset):
        # 1. Centering the data
        self.mean = np.mean(dataset.X, axis=0)
        centered_data = dataset.X - self.mean 
        
        # 2. Covariance matrix and eigenvalue decomposition
        cov_matrix = np.cov(centered_data, rowvar=False) 
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 3. Infer Principal Components 
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Select the first n components (transpose to be (n_comp, n_feat))
        self.components = sorted_eigenvectors[:, :self.n_components].T
        
        # 4. Infer Explained Variance
        self.explained_variance = sorted_eigenvalues[:self.n_components] / np.sum(eigenvalues)
        
        return self

    def _transform(self, dataset: Dataset):
        # 1. Centering the data
        centered_data = dataset.X - self.mean
        # 2. Calculate reduced X
        X_reduced = np.dot(centered_data, self.components.T)
        
        return Dataset(X=X_reduced, y=dataset.y, features=[f"PC{i+1}" for i in range(self.n_components)], label=dataset.label)