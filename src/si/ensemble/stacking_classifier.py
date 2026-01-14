import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.base.model import Model

class StackingClassifier(Model):
    """
    Ensemble classifier that uses a set of models to generate predictions. These predictions are then used as input features for a final model to make the ultimate prediction.
    """
    def __init__(self, models, final_model):
        """
        Initialize the StackingClassifier.

        Parameters
        ----------
        models : list
            Initial set of models to be trained.
        final_model : Model
            The model to make the final predictions.
        """
        super().__init__()
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Trains the ensemble models and the final model.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self : StackingClassifier
            The trained model.
        """
        # 1. Train the initial set of models
        for model in self.models:
            model.fit(dataset)

        # 2. Get predictions from the initial set of models
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))
        
        # Transforms the list of arrays (n_models, n_samples) into (n_samples, n_models)
        # np.column_stack
        predictions = np.array(predictions).T

        # 3. Train the final model with the predictions of the initial set of models
        # The X of the new dataset is the predictions, the Y remains the original
        dataset_predictions = Dataset(predictions, dataset.y)
        self.final_model.fit(dataset_predictions)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models and final model.

        Parameters
        ----------
        dataset : Dataset
            The validation dataset.

        Returns
        -------
        final_predictions : np.ndarray
            The predictions of the final model.
        """
        # 1. Get predictions from the initial set of models
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))
        
        # Organize predictions in the correct format (n_samples, n_models)
        predictions = np.array(predictions).T

        # 2. Get the final predictions using the final model
        dataset_predictions = Dataset(predictions, dataset.y)
        return self.final_model.predict(dataset_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray = None) -> float:
        """
        Computes the accuracy of the model.

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate.
        predictions : np.ndarray
            The predictions (optional).

        Returns
        -------
        score : float
            The accuracy of the model.
        """
        if predictions is None:
            predictions = self.predict(dataset)
            
        return accuracy(dataset.y, predictions)

