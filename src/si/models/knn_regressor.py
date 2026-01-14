import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor(Model):
    def __init__(self, k: int = 5, distance=euclidean_distance):
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None 

    def _fit(self, dataset: Dataset):
        self.dataset = dataset 
        return self

    def _predict(self, dataset: Dataset):
        predictions = []
        n_samples = dataset.shape()[0]
        
        for i in range(n_samples): # estudar vem esta parte aqui 

            # 1. Calculate distances
            dists = self.distance(dataset.X[i], self.dataset.X)
            
            # 2. Get indices of the k nearest neighbors
            k_nearest_neighbors = np.argsort(dists)[:self.k]
            
            # 3. Get corresponding y values
            k_nearest_values = self.dataset.y[k_nearest_neighbors]
            
            # 4. Calculate the mean
            prediction = np.mean(k_nearest_values)
            predictions.append(prediction)
            
        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions):
        # Calculate RMSE
        return rmse(dataset.y, predictions)