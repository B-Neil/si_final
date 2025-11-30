import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse 

class RidgeRegressionLeastSquares(Model):
    def __init__(self, l2_penalty: float = 1, scale: bool = True):
        super().__init__()
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None 

    def _fit(self, dataset: Dataset):
        X = dataset.X
        
        # 1. Scale data 
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            X = (X - self.mean) / self.std
            
        # 2. Add intercept term (column of ones)
        X = np.c_[np.ones(X.shape[0]), X]
        
        # 3. Compute penalty matrix (lambda * I) 
        n_features = X.shape[1]
        penalty_matrix = self.l2_penalty * np.eye(n_features)
        
        # 4. Change first position to 0 to not penalize intercept
        penalty_matrix[0, 0] = 0
        
        # 5. Compute parameters: theta = (X.T * X + penalty)^-1 * X.T * y 
        # X.T dot X
        xt_x = np.dot(X.T, X)
        # Inverse of (XTX + Penalty)
        inv_matrix = np.linalg.inv(xt_x + penalty_matrix)
        # ... dot X.T
        inv_xt = np.dot(inv_matrix, X.T)
        # ... dot y
        thetas = np.dot(inv_xt, dataset.y)
        
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]
        
        return self

    def _predict(self, dataset: Dataset):
        X = dataset.X
        # 1. Scale 
        if self.scale:
            X = (X - self.mean) / self.std
            
        # 2. Add intercept
        X = np.c_[np.ones(X.shape[0]), X]
        
        # 3. Compute predicted Y 
        thetas = np.r_[self.theta_zero, self.theta]
        return np.dot(X, thetas)

    def _score(self, dataset: Dataset, predictions):
        return mse(dataset.y, predictions) 