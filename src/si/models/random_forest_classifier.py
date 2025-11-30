import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier(Model):
    def __init__(self, n_estimators: int = 100, max_features: int = None, min_samples_split: int = 2, 
                 max_depth: int = 10, mode: str = 'gini', seed: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def _fit(self, dataset: Dataset):
        # 1. Set random seed 
        np.random.seed(self.seed)
        
        # 2. Define max_features
        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.shape()[1]))
            
        n_samples = dataset.shape()[0]
        
        for i in range(self.n_estimators):
            # 3. Create bootstrap dataset
            # Amostras com reposição
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            # Features sem reposição
            feature_indices = np.random.choice(dataset.shape()[1], self.max_features, replace=False)
            
            bootstrap_X = dataset.X[bootstrap_indices][:, feature_indices]
            bootstrap_y = dataset.y[bootstrap_indices]
            
            bootstrap_dataset = Dataset(bootstrap_X, bootstrap_y)
            
            # 4. Create and train decision tree 
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, 
                                          max_depth=self.max_depth, mode=self.mode)
            tree.fit(bootstrap_dataset)
            
            # 5. Append tuple (features used, tree)
            self.trees.append((feature_indices, tree))
            
        return self

    def _predict(self, dataset: Dataset):
        predictions = []
        n_samples = dataset.shape()[0]
        
        # 1. Get predictions for each tree
        all_tree_preds = []
        
        for feature_indices, tree in self.trees:
            tree_dataset = Dataset(dataset.X[:, feature_indices], dataset.y)
            all_tree_preds.append(tree.predict(tree_dataset))
            
        all_tree_preds = np.array(all_tree_preds)
        
        # 2. Get most common predicted class for each sample (Votação majoritária)
        for sample_preds in all_tree_preds.T:
            values, counts = np.unique(sample_preds, return_counts=True)
            most_common = values[np.argmax(counts)]
            predictions.append(most_common)
            
        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions):
        # Accuracy
        return np.sum(dataset.y == predictions) / len(dataset.y)