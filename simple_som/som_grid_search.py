from itertools import product
from joblib import Parallel, delayed
from simple_som.som import SimpleSOM
from sklearn.metrics import accuracy_score

class SOMGridSearch:
    def __init__(self, param_grid, n_jobs=1):
        """Initializes the SOMGridSearch object.

        Args:
            param_grid (dict): Dictionary containing the SOM parameters to be tuned.
            n_jobs (int): Number of parallel jobs for training SOM models.
                          Default is 1.
        """
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = 0

    def _train_som(self, params, X_train, y_train, X_test, y_test):
        """Trains a SOM model with given parameters and
        calculates accuracy on test data.

        Args:
            params (tuple): Tuple of parameter values.
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            X_test (numpy.ndarray): Test data features.
            y_test (numpy.ndarray): Test data labels.
            param_combinations (list): List of parameter combinations.

        Returns:
            accuracy (float): Accuracy score on test data.
            params (tuple): The same input parameters tuple.
        """
        som = SimpleSOM(X_train=X_train, y_train=y_train,
                        **dict(zip(self.param_grid.keys(), params)))
        # Train the SOM
        X_train = X_train.copy()
        _ = som.fit(X_data=X_train)
        predicted_1 = som.predict(X_train=X_train, y_train=y_train, X_test=X_test, K=None)
        predicted_2 = som.predict_kw(X_train, y_train, X_test, K=None)
        predicted_3 = som.predict_kcw(X_train, y_train, X_test, K=None)
        predicted_4 = som.predict_knn(X_test, K=None)
        accuracy_1 = accuracy_score(y_test, predicted_1)
        accuracy_2 = accuracy_score(y_test, predicted_2)
        accuracy_3 = accuracy_score(y_test, predicted_3)
        accuracy_4 = accuracy_score(y_test, predicted_4)
        accuracy = max(accuracy_1, accuracy_2, accuracy_3, accuracy_4)
        return accuracy, params

    def fit(self, X_train, y_train, X_test, y_test):
        """Fits the SOM models with different parameter
        combinations and finds the best model.

        Args:
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            X_test (numpy.ndarray): Test data features.
            y_test (numpy.ndarray): Test data labels.

        Returns:
            best_params (dict): Dictionary of the best SOM parameters.
            best_score (float): Accuracy score of the best model.
        """
        param_combinations = list(product(*self.param_grid.values()))
        folds = 1
        print(f"Fitting {folds} folds for each of {len(param_combinations)}"
            f" candidates, totalling {folds*len(param_combinations)} fits")
        # joblib to parallelize the training process
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_som)(params,
                                     X_train, y_train, X_test, 
                                     y_test) for params in param_combinations
        )
        # Find the best results
        for result in results:
            accuracy, params = result
            if accuracy > self.best_score_:
                self.best_score_ = accuracy
                self.best_params_ = params
        self.best_params_ = {param_name: param_value
                             for param_name, param_value
                             in zip(self.param_grid, self.best_params_)}
        return self.best_params_, self.best_score_