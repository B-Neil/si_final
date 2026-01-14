from unittest import TestCase
import os
import sys
import numpy as np

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
SRC_PATH = os.path.join(project_root, 'src')

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ==============================================================================
# IMPORTS
# ==============================================================================
from si.neural_networks.losses import CategoricalCrossEntropy

class TestCategoricalCrossEntropy(TestCase):

    def setUp(self):
        self.loss_function = CategoricalCrossEntropy()

    def test_loss(self):
        # Case 1: 3 samples, 3 classes
        # Sample 1: Class 0 (Correct)
        # Sample 2: Class 1 (Correct)
        # Sample 3: Class 2 (Correct)
        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        # “Imperfect” but confident predictions
        y_pred = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],  
            [0.2, 0.2, 0.6]   
        ])
        
        loss = self.loss_function.loss(y_true, y_pred)
        
        # Check if the value is close to the expected value (up to 3 decimal places)
        self.assertAlmostEqual(loss, 0.3635, places=3)
        
        # Case 2: Perfect Predictions
        y_pred_perfect = np.array([
            [0.99, 0.0, 0.01],
            [0.01, 0.99, 0.0],
            [0.0, 0.0, 0.99]
        ])
        loss_perfect = self.loss_function.loss(y_true, y_pred_perfect)
        self.assertLess(loss_perfect, 0.02)

    def test_derivative(self):
        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        y_pred = np.array([
            [0.5, 0.2, 0.3],
            [0.1, 0.8, 0.1]
        ])
        
        derivative = self.loss_function.derivative(y_true, y_pred)
        
        expected_derivative = np.array([
            [-2.0, -0.0, -0.0],
            [-0.0, -1.25, -0.0]
        ])
        
        np.testing.assert_array_almost_equal(derivative, expected_derivative)

    def test_numerical_stability(self):
        """
        Tests whether the code can handle predictions of absolute zero (which would cause log(0)).
        This validates whether np.clip is working.
        """
        y_true = np.array([[1, 0]])
        # The model says that the probability of the correct class is 0 (theoretically infinite error).
        y_pred_bad = np.array([[0.0, 1.0]]) 
        loss = self.loss_function.loss(y_true, y_pred_bad)
        derivative = self.loss_function.derivative(y_true, y_pred_bad)
        
        # Check if you obtained finite numbers
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(derivative)))

if __name__ == '__main__':
    import unittest
    unittest.main()