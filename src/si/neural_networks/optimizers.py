from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    

class Adam(Optimizer):
    """
    Adam optimizer.
    """
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        learning_rate : float
            The learning rate.
        beta_1 : float
            The exponential decay rate for the 1st moment estimates.
        beta_2 : float
            The exponential decay rate for the 2nd moment estimates.
        epsilon : float
            A small constant for numerical stability.
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        # Estimated parameters
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Time step

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights using the Adam algorithm.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_loss_w : np.ndarray
            Gradient of the loss function w.r.t. the weights.
        """
        # 1. Initialize m and v as zeros if they don't exist
        if self.m is None:
            self.m = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))
        
        # 2. Update time step
        self.t += 1
        
        # 3. Compute and update m (1st moment vector)
        # m = beta1 * m + (1 - beta1) * gradient
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        
        # 4. Compute and update v (2nd moment vector)
        # v = beta2 * v + (1 - beta2) * gradient^2
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)
        
        # 5. Compute Bias-Corrected first moment estimate (m_hat) 
        # Correção necessária porque no início m e v tendem para 0
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        
        # 6. Compute Bias-Corrected second moment estimate (v_hat) 
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        
        # 7. Update weights 
        # w = w - learning_rate * (m_hat / (sqrt(v_hat) + epsilon))
        w_updated = w - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
        
        return w_updated