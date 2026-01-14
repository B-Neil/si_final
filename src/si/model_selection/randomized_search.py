import numpy as np
import itertools
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model, dataset: Dataset, hyperparameter_grid: dict, scoring=None, cv: int = 5, n_iter: int = 10):
    """
    Performs a randomized search cross-validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset : Dataset
        The dataset to cross validate on.
    hyperparameter_grid : dict
        The hyperparameter grid to search. Key is the hyperparameter name and value is the list of values to search.
    scoring : Callable
        The scoring function to use.
    cv : int
        The number of folds to use.
    n_iter : int
        The number of random combinations to test.

    Returns
    -------
    results : dict
        The results of the randomized search cross-validation. Includes the scores, hyperparameters, best hyperparameters and best score.
    """
    #1. Verify that hyperparameters exist in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}")

    #2. Obtain all possible combinations (Cartesian product)
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()
    all_combinations = list(itertools.product(*values))

    #3. Select n_iter random combinations
    num_combinations = len(all_combinations)
    if n_iter > num_combinations:
        n_iter = num_combinations
        random_indices = np.arange(num_combinations)
    else:
        random_indices = np.random.choice(num_combinations, size=n_iter, replace=False)
    
    # List only with the selected combinations
    random_combinations = [all_combinations[i] for i in random_indices]

    # Structure for storing results
    results = {
        'hyperparameters': [],
        'scores': [],
        'best_hyperparameters': None,
        'best_score': -np.inf
    }

    #4. Iterate over the chosen combinations
    for combination in random_combinations:
        # Create dictionary of current combination {param: value}
        parameters = dict(zip(keys, combination))
        
        # Configure the model with these hyperparameters (setattr is the trick here)        
        for parameter, value in parameters.items():
            setattr(model, parameter, value)

        #5. Validate with Cross-Validation
        scores = k_fold_cross_validation(model, dataset, scoring=scoring, cv=cv)
        
        # Calculate the average of the scores
        score = np.mean(scores)

        #6. Save the results
        results['hyperparameters'].append(parameters)
        results['scores'].append(score)

        #7. Check if it's the best so far
        if score > results['best_score']:
            results['best_score'] = score
            results['best_hyperparameters'] = parameters

    return results
