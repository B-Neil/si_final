from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training and testing sets keeping class proportions.
    """
    # Definir a seed
    np.random.seed(random_state)
    
    # Get unique class labels and counts 
    labels, counts = np.unique(dataset.y, return_counts=True)
    
    train_indices = []
    test_indices = []
    
    # Loop through unique labels 
    for label in labels:
        # Obter índices correspondentes à classe atual
        idxs = np.where(dataset.y == label)[0]
        
        # Calcular número de amostras de teste para esta classe
        n_test = int(len(idxs) * test_size)
        
        # Shuffle 
        np.random.shuffle(idxs)
        
        # Selecionar índices
        test_indices.extend(idxs[:n_test])
        train_indices.extend(idxs[n_test:])
        
    # Criar datasets de treino e teste 
    train_dataset = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)
    
    return train_dataset, test_dataset