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
from si.neural_networks.activation import TanhActivation, SoftmaxActivation

class TestActivations(TestCase):

    def setUp(self):
        # Sample data (3 samples, 2 features)
        self.input_data = np.array([
            [0.5, -0.5],
            [1.0, 0.0],
            [-2.0, 2.0]
        ])

    def test_tanh_activation(self):
        layer = TanhActivation()
        
        #1. Forward Testing
        # tanh “crushes” values to [-1, 1]
        output = layer.forward_propagation(self.input_data, training=True)
        
        # Check limits
        self.assertTrue(np.all(output >= -1) and np.all(output <= 1))
        
        # Verify exact calculation (compare with numpy)
        expected_output = np.tanh(self.input_data)
        np.testing.assert_array_almost_equal(output, expected_output)
        
        #2. Backward Testing (Derivative)
        # Formula: 1 - tanh(x)^2
        # To test backward, we simulate an output error of 1
        output_error = np.ones_like(self.input_data)
        input_error = layer.backward_propagation(output_error)
        
        expected_derivative = 1 - np.tanh(self.input_data) ** 2
        np.testing.assert_array_almost_equal(input_error, expected_derivative)

    def test_softmax_activation(self):
        layer = SoftmaxActivation()
        
        # 1. Test Forward
        output = layer.forward_propagation(self.input_data, training=True)
        
        # Fundamental Property: The sum of the probabilities for each row must be 1.
        # axis=1 sums the columns for each row.
        sums = np.sum(output, axis=1)
        expected_sums = np.ones(self.input_data.shape[0])
        np.testing.assert_array_almost_equal(sums, expected_sums)
        
        # Check values: All positive
        self.assertTrue(np.all(output >= 0))

        # Check manual calculation for the first row [0.5, -0.5]
        # exp(0.5) = 1.6487, exp(-0.5) = 0.6065
        # sum = 2.2552
        # soft[0] = 0.731, soft[1] = 0.268
        row0_exp = np.exp([0.5, -0.5])
        row0_expected = row0_exp / np.sum(row0_exp)
        np.testing.assert_array_almost_equal(output[0], row0_expected)

        # 2. Backward Testing
        # Simplified Formula: output * (1 - output) * error
        output_error = np.ones_like(self.input_data)
        input_error = layer.backward_propagation(output_error)
        
        # Since output_error is 1, input_error = derivative
        # Derivative = S * (1 - S)
        expected_derivative = output * (1 - output)
        np.testing.assert_array_almost_equal(input_error, expected_derivative)

if __name__ == '__main__':
    import unittest
    unittest.main()