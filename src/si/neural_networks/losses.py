import numpy as np
from abc import ABC, abstractmethod

class LossFunction(ABC):
    """
    Base class for loss functions.
    """
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss based on the true and predicted labels.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns
        -------
        loss : float
            The loss value.
        """
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss based on the true and predicted labels.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns
        -------
        derivative : np.ndarray
            The derivative of the loss.
        """
        pass


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error loss function.
    Usually used for regression tasks.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the mean squared error loss.
        Formula: (1/n) * sum((y_true - y_pred)^2)
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the mean squared error loss.
        Formula: (2/n) * (y_pred - y_true)
        """
        return 2 * (y_pred - y_true) / len(y_true)


class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross Entropy loss function.
    Usually used for binary classification tasks.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the binary cross entropy loss.
        Formula: - (1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        """

        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross entropy loss.
        Formula: (y_pred - y_true) / (y_pred * (1 - y_pred))
        """

        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (p - y_true) / (p * (1 - p))


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical Cross Entropy loss function.
    Used for multi-class classification tasks.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the categorical cross entropy loss.
        Formula: - sum(y_true * log(y_pred))
        
        Note: The formula in the slides sums over i=1 to N (classes/samples). 
        Typically, we average over the batch to get a scalar.
        """

        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p)) / len(y_true)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the categorical cross entropy loss.
        Formula: - y_true / y_pred
        """

        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p)