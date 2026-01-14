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
DATASETS_PATH = os.path.join(project_root, 'datasets')

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ==============================================================================
# IMPORTS
# ==============================================================================
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.ensemble.stacking_classifier import StackingClassifier

class TestStackingClassifier(TestCase):
    
    def setUp(self):
        self.filename = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        
        if not os.path.exists(self.filename):
            self.skipTest(f"Dataset não encontrado: {self.filename}")
            
        self.dataset = read_csv(self.filename, features=True, label=True)

    def test_fit_predict_score(self):
        # 2. Split the data into train and test sets 
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)

        # 3. Create a KNNClassifier model 
        knn = KNNClassifier(k=3)

        # 4. Create a LogisticRegression model 
        lr = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

        # 5. Create a DecisionTree model 
        dt = DecisionTreeClassifier(min_samples_split=2, max_depth=5, mode='gini')

        # 6. Create a second KNNClassifier model (final model) 
        final_model = KNNClassifier(k=3)

        # 7. Create a StackingClassifier model 
        stacking = StackingClassifier(models=[knn, lr, dt], final_model=final_model)

        # 8. Train the StackingClassifier model 
        stacking.fit(train)

        # Verify if predictions are generated
        predictions = stacking.predict(test)
        self.assertEqual(predictions.shape[0], test.shape()[0])

        # What is the score of the model on the test set? 
        score = stacking.score(test)
        print(f"\nStackingClassifier Accuracy: {score:.4f}")
        
        # Validação básica: accuracy deve estar entre 0 e 1
        self.assertTrue(0 <= score <= 1)
        # Opcional: Verificar se é melhor que o acaso
        self.assertGreater(score, 0.5)

if __name__ == '__main__':
    import unittest
    unittest.main()