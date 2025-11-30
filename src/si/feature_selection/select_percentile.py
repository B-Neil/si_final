import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.base.transformer import Transformer

class SelectPercentile(Transformer):
    def __init__(self, score_func=f_classification, percentile=50):
        """
        Select features according to a percentile of the highest scores.
        Parameters:
        score_func: Function taking a dataset and returning (F-values, p-values).
        percentile: int, percent of features to keep.
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset):
        # Estima F e p para cada feature
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset):
        # Calcula o limiar baseado no percentil
        len_features = len(dataset.features)
        k = int(len_features * (self.percentile / 100))
        
        # Ordena os índices dos F-values em ordem decrescente
        idxs = np.argsort(self.F)[::-1]
        # Seleciona as k melhores features
        best_idxs = idxs[:k]
        
        # Lógica para lidar com empates
        
        dataset.X = dataset.X[:, best_idxs]
        dataset.features = [dataset.features[i] for i in best_idxs]
        
        return dataset