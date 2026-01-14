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
from si.neural_networks.optimizers import Adam, SGD

class TestOptimizers(TestCase):

    def setUp(self):
        # Setup for a simple optimization problem
        # Function: f(w) = (w - 10)^2
        # Objective: Find w = 10 (where the error is 0)
        # Derivative (Gradient): f'(w) = 2 * (w - 10)
        self.w_start = np.array([0.0])  # Começamos em 0
        self.target = 10.0
        
    def test_sgd(self):
        """
        Quick test to ensure that the SGD (basis for comparison) works.
        """
        optimizer = SGD(learning_rate=0.1)
        w = self.w_start.copy()
        for _ in range(100):
            grad = 2 * (w - self.target)
            w = optimizer.update(w, grad)
        self.assertAlmostEqual(w[0], self.target, places=2)

    def test_adam_convergence(self):
        """
        Check whether Adam converges to the minimum of the function.
        """
        optimizer = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        w = self.w_start.copy()
        for _ in range(500):
            grad = 2 * (w - self.target)
            w = optimizer.update(w, grad)
        
        print(f"\nAdam Final Weight: {w[0]:.4f} (Target: {self.target})")
        
        # Check if m and v have been initialized
        self.assertIsNotNone(optimizer.m)
        self.assertIsNotNone(optimizer.v)
        
        # Check convergence
        self.assertAlmostEqual(w[0], self.target, places=1)

    def test_adam_math_logic(self):
        """
        White-box testing to validate the exact mathematics of an iteration.
        """
        lr = 0.1
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        
        optimizer = Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps)
        
        w = np.array([2.0])
        grad = np.array([1.0])
        
        # --- Iteration 1 ---
        new_w = optimizer.update(w, grad)
        
        # 1. Update t
        self.assertEqual(optimizer.t, 1)
        
        # 2. Update m
        # m = b1 * 0 + (1-b1) * grad = 0.9*0 + 0.1*1 = 0.1
        expected_m = 0.1
        self.assertAlmostEqual(optimizer.m[0], expected_m)
        
        # 3. Update v
        # v = b2 * 0 + (1-b2) * grad^2 = 0.999*0 + 0.001*1 = 0.001
        expected_v = 0.001
        self.assertAlmostEqual(optimizer.v[0], expected_v)
        
        # 4. Bias Correction (m_hat, v_hat) 
        # m_hat = m / (1 - b1^1) = 0.1 / 0.1 = 1.0
        # v_hat = v / (1 - b2^1) = 0.001 / 0.001 = 1.0
        m_hat = expected_m / (1 - b1)
        v_hat = expected_v / (1 - b2)
        
        # 5. Update Weights
        # w = w - lr * (m_hat / (sqrt(v_hat) + eps))
        # w = 2.0 - 0.1 * (1.0 / (1.0 + 1e-8)) approx 1.9
        expected_w = w - lr * (m_hat / (np.sqrt(v_hat) + eps))
        
        self.assertAlmostEqual(new_w[0], expected_w[0])

if __name__ == '__main__':
    import unittest
    unittest.main()