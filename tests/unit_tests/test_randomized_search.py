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
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import randomized_search_cv

class TestRandomizedSearch(TestCase):

    def setUp(self):
        # 1. Use the breast-bin.csv dataset
        self.filename = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        
        if not os.path.exists(self.filename):
            self.skipTest(f"Dataset não encontrado: {self.filename}")
            
        self.dataset = read_csv(self.filename, features=True, label=True)

    def test_randomized_search_protocol(self):
        # 2. Create a LogisticRegression model 
        model = LogisticRegression()

        # 3. Perform a randomized search with the following hyperparameter distributions:
        hyperparameter_grid = {
            # l2_penalty: distribution between 1 and 10 with 10 equal intervals
            'l2_penalty': np.linspace(1, 10, 10),
            
            # alpha: distribution between 0.001 and 0.0001 with 100 equal intervals
            'alpha': np.linspace(0.001, 0.0001, 100),
            
            # max_iter: distribution between 1000 and 2000 with 200 equal intervals
            'max_iter': np.linspace(1000, 2000, 200, dtype=int)
        }

        # 4. Use n_iter=10 and cv=3 folds for the cross validation 
        n_iter = 10
        cv = 3
        
        results = randomized_search_cv(model, 
                                       self.dataset, 
                                       hyperparameter_grid, 
                                       cv=cv, 
                                       n_iter=n_iter)

        # Validações do teste
        
        # Verificar se o output é um dicionário com as chaves corretas 585]
        self.assertIn('scores', results)
        self.assertIn('best_score', results)
        self.assertIn('best_hyperparameters', results)
        self.assertIn('hyperparameters', results)

        # Verificar se foram testadas exatamente 10 combinações
        self.assertEqual(len(results['scores']), n_iter)
        self.assertEqual(len(results['hyperparameters']), n_iter)

        # Verificar se o melhor score é válido (entre 0 e 1)
        self.assertTrue(0 <= results['best_score'] <= 1)

        # Imprimir resultados para análise (conforme pedido no slide)
        print("\n--- Resultados Randomized Search ---")
        print(f"Melhor Score: {results['best_score']:.4f}")
        print(f"Melhores Hiperparâmetros: {results['best_hyperparameters']}")
        print(f"Todos os scores: {results['scores']}")

if __name__ == '__main__':
    import unittest
    unittest.main()