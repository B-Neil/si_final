from unittest import TestCase
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
SRC_PATH = os.path.join(project_root, 'src')
DATASETS_PATH = os.path.join(project_root, 'datasets')

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
from si.decomposition.pca import PCA
from si.statistics.tanimoto_similarity import tanimoto_similarity
from si.model_selection.split import stratified_train_test_split, train_test_split
from si.models.knn_regressor import KNNRegressor
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.models.random_forest_classifier import RandomForestClassifier
from si.metrics.rmse import rmse
from si.metrics.mse import mse

class TestExercise1and2(TestCase):
    """Exercícios 1 e 2"""

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_numpy_slicing_ex1_variations(self):
        #Penúltima feature
        penultimate = self.dataset.X[:, -2]
        self.assertEqual(penultimate.shape[0], self.dataset.shape()[0])

        #Médias com slice diferente
        first_5 = self.dataset.X[:5]
        mean_first_5 = np.mean(first_5, axis=0)
        self.assertEqual(mean_first_5.shape[0], self.dataset.shape()[1])

        #Filtro que retorna vazio
        mask_empty = self.dataset.X[:, 0] < 0 
        empty_X = self.dataset.X[mask_empty]
        self.assertEqual(empty_X.shape[0], 0)

    def test_dataset_methods_ex2_variations(self):
        # Dataset Dummy com muitos NaNs
        # Coluna 0: valores [1, nan, 5, 7]. Válidos: 1, 5, 7. Soma=13. Média=4.333
        X = np.array([[1, 2], [np.nan, np.nan], [5, 6], [7, np.nan]])
        y = np.array([0, 1, 0, 1])
        features = ['f1', 'f2']
        ds = Dataset(X.copy(), y.copy(), features=features, label='y')

        # Dropna
        ds_drop = Dataset(X.copy(), y.copy(), features, label='y')
        ds_drop.dropna()
        self.assertEqual(ds_drop.shape()[0], 2)
        self.assertFalse(np.isnan(ds_drop.X).any())

        #Fillna com valor escalar (zero)
        ds_fill_zero = Dataset(X.copy(), y.copy(), features, label='y')
        ds_fill_zero.fillna(0)
        self.assertEqual(ds_fill_zero.X[1, 0], 0)
        self.assertEqual(ds_fill_zero.X[3, 1], 0)

        #Fillna com 'mean'
        ds_fill_mean = Dataset(X.copy(), y.copy(), features, label='y')
        ds_fill_mean.fillna('mean')
        self.assertAlmostEqual(ds_fill_mean.X[1, 0], 13/3)

        #Remove by index (remover o último)
        ds = Dataset(X.copy(), y.copy(), features, label='y')
        last_idx = ds.shape()[0] - 1
        ds.remove_by_index(last_idx)
        self.assertEqual(ds.shape()[0], 3)


class TestSelectPercentile(TestCase):
    """Exercício 3"""

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit_transform_variations(self):
        #50% (Iris tem 4 features -> sobram 2)
        selector_50 = SelectPercentile(percentile=50)
        selector_50.fit(self.dataset)
        transformed_50 = selector_50.transform(self.dataset)
        self.assertEqual(transformed_50.shape()[1], 2)

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        #100% (Deve manter todas as 4 features)
        selector_100 = SelectPercentile(percentile=100)
        selector_100.fit(self.dataset)
        transformed_100 = selector_100.transform(self.dataset)
        self.assertEqual(transformed_100.shape()[1], 4)

        #Verificar se calculou F e p
        self.assertIsNotNone(selector_50.F)
        self.assertIsNotNone(selector_50.p)


class TestStatistics(TestCase):
    """Exercício 4 (Tanimoto)"""

    def test_tanimoto_similarity_variations(self):
        a = np.array([1, 0, 1, 0])
        b = np.array([1, 0, 1, 0]) # Igual
        c = np.array([0, 1, 0, 1]) # Oposto
        d = np.array([1, 0, 0, 0]) # Parcialmente igual

        #Identicos
        self.assertEqual(tanimoto_similarity(a, b.reshape(1, -1))[0], 1.0)
        
        #Totalmente diferentes
        self.assertEqual(tanimoto_similarity(a, c.reshape(1, -1))[0], 0.0)

        #Parcial
        self.assertAlmostEqual(tanimoto_similarity(a, d.reshape(1, -1))[0], 0.5)


class TestPCA(TestCase):
    """Exercício 5"""

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit_transform_variations(self):
        #2 Componentes
        pca_2 = PCA(n_components=2)
        pca_2.fit(self.dataset)
        t_2 = pca_2.transform(self.dataset)
        self.assertEqual(t_2.shape()[1], 2)
        self.assertTrue(pca_2.explained_variance[0] >= pca_2.explained_variance[1])

        #1 Componente
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        pca_1 = PCA(n_components=1)
        pca_1.fit(self.dataset)
        t_1 = pca_1.transform(self.dataset)
        self.assertEqual(t_1.shape()[1], 1)

        self.assertIsNotNone(pca_2.mean)


class TestStratifiedSplit(TestCase):
    """Exercício 6"""

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_stratified_train_test_split_variations(self):
        #Split 20%
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.assertEqual(train.shape()[0] + test.shape()[0], 150)
        self.assertEqual(len(np.unique(test.y)), 3)

        #Split 50%
        train_50, test_50 = stratified_train_test_split(self.dataset, test_size=0.5, random_state=1)
        self.assertTrue(70 <= test_50.shape()[0] <= 80)
        self.assertEqual(train_50.shape()[0] + test_50.shape()[0], 150)


class TestKNNRegressor(TestCase):
    """Exercício 7"""

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        if not os.path.exists(self.csv_file):
            self.skipTest("cpu.csv não encontrado")
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_knn_variations(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)

        #k=1
        knn_1 = KNNRegressor(k=1)
        knn_1.fit(train)
        score_1 = knn_1.score(test)
        self.assertGreater(score_1, 0)

        #k=3
        knn_3 = KNNRegressor(k=3)
        knn_3.fit(train)
        score_3 = knn_3.score(test)
        self.assertIsInstance(score_3, float)
        
        preds = knn_3.predict(test)
        self.assertEqual(len(preds), len(test.y))


class TestRidgeRegressionLeastSquares(TestCase):
    """Exercício 8"""

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        if not os.path.exists(self.csv_file):
            self.skipTest("cpu.csv não encontrado")
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_ridge_variations(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)

        #Com Scale=True
        ridge_scaled = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge_scaled.fit(train)
        score_scaled = ridge_scaled.score(test)
        self.assertIsNotNone(ridge_scaled.theta)
        
        #Sem Scale
        ridge_no_scale = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=False)
        ridge_no_scale.fit(train)
        score_no_scale = ridge_no_scale.score(test)
        
        self.assertNotEqual(score_scaled, score_no_scale)
        
        #L2 Penalty Zero
        ridge_zero = RidgeRegressionLeastSquares(l2_penalty=0.0, scale=True)
        ridge_zero.fit(train)
        self.assertIsNotNone(ridge_zero.theta_zero)


class TestRandomForestClassifier(TestCase):
    """Exercício 9"""

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_rf_variations(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

        #Floresta pequena
        rf_small = RandomForestClassifier(n_estimators=5, max_depth=3, seed=42)
        rf_small.fit(train)
        self.assertEqual(len(rf_small.trees), 5)
        score_small = rf_small.score(test)
        self.assertTrue(0 <= score_small <= 1)

        #Floresta maior
        rf_large = RandomForestClassifier(n_estimators=10, max_depth=10, seed=42)
        rf_large.fit(train)
        self.assertEqual(len(rf_large.trees), 10)
        
        #Reproducibilidade
        rf_rep1 = RandomForestClassifier(n_estimators=5, seed=1)
        rf_rep1.fit(train)
        preds_1 = rf_rep1.predict(test)
        
        rf_rep2 = RandomForestClassifier(n_estimators=5, seed=1)
        rf_rep2.fit(train)
        preds_2 = rf_rep2.predict(test)
        
        self.assertTrue(np.array_equal(preds_1, preds_2))


if __name__ == '__main__':
    import unittest
    unittest.main()