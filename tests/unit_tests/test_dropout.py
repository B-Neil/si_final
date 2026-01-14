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
from si.neural_networks.layers import Dropout

class TestDropout(TestCase):

    def setUp(self):
        # Create deterministic input data for testing
        # 2x10 matrix (2 samples, 10 features) with values from 1 to 20
        self.input_data = np.arange(1, 21).reshape(2, 10).astype(float)
        self.probability = 0.5
        self.layer = Dropout(probability=self.probability)

    def test_forward_propagation_training(self):
        """
        Tests behavior in training mode (training=True).
        Checks:
        1. Whether the output has the same shape as the input.
        2. Whether some values have been set to zero (turned off).
        3. Whether the active values have been scaled (Inverted Dropout).
        """
        output = self.layer.forward_propagation(self.input_data, training=True)
        
        # ]1. Check Shape 
        self.assertEqual(output.shape, self.input_data.shape)
        
        #2. Check if the mask has been created
        self.assertIsNotNone(self.layer.mask)
        
        # 3. Check Scaling Factor
        scaling_factor = 1 / (1 - self.probability)
        expected_active = self.input_data * scaling_factor
        
        # Validate logic element by element
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if self.layer.mask[i, j] == 1:
                    self.assertAlmostEqual(output[i, j], expected_active[i, j])
                else:
                    self.assertEqual(output[i, j], 0.0)

        # Check if “something” has been turned off
        self.assertTrue(np.any(output == 0), "O Dropout não desligou nenhum neurónio (improvável mas possível).")

    def test_forward_propagation_inference(self):
        """
        Tests behavior in inference mode (training=False).
        """
        output = self.layer.forward_propagation(self.input_data, training=False)
        
        #The input should not be changed.
        np.testing.assert_array_equal(output, self.input_data)

    def test_backward_propagation(self):
        """
        Tests whether the error only passes through active neurons.
        """
        self.layer.forward_propagation(self.input_data, training=True)
        output_error = np.ones_like(self.input_data)
        input_error = self.layer.backward_propagation(output_error)
        # Check that the input error respects the mask
        np.testing.assert_array_equal(input_error, self.layer.mask)

    def test_parameters_and_shape(self):
        """
        Auxiliary metadata tests.
        """
        self.assertEqual(self.layer.parameters(), 0)
        self.layer.input = self.input_data
        self.assertEqual(self.layer.output_shape(), self.input_data.shape)

if __name__ == '__main__':
    import unittest
    unittest.main()