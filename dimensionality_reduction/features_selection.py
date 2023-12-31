from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

def search_best_param(X_train, y_train, n_jobs=-1):
    """Find the best hyperparameters using grid search for a RandomForestClassifier.

    Args:
        X_train (DataFrame): Observations.
        y_train (Series): Labels.

    Returns:
        tuple: (best_params, best_estimator)
        - best_params (dict): Best hyperparameters for RandomForestClassifier.
        - best_estimator (RandomForestClassifier): Best RandomForestClassifier model.
    """
    # Hyperparameters to explore
    param_grid = {
        'n_estimators': [500, 600, 800, 1000, 1200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    # RandomForestClassifier
    rf_model = RandomForestClassifier()
    # Metric to use for evaluation
    scoring = {'Accuracy': make_scorer(accuracy_score)}
    # Perform grid search
    grid_search = GridSearchCV(rf_model, param_grid, scoring=scoring, refit='Accuracy', cv=5, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    return best_params, best_estimator


def compute_feature_importance(X_train, y_train, n_jobs=-1):
    """Calculate feature importance using a Random Forest classifier.

    Args:
        X_train (DataFrame): Observations.
        y_train (Series): Labels.

    Returns:
        ndarray: Importances of features.
    """
    # Find the best hyperparameters using grid search
    # best_params, _ = search_best_param(X_train=X_train, y_train=y_train)
    N_ESTIMATORS = 1000 # The number of trees in the forest.
    CRITERION = 'entropy' # The function to measure the quality of a split
    MAX_DEPTH = 20 # The maximum depth of the tree
    MAX_FEATURE = 'log2' # The number of features to consider when looking for the best split
    RANDOM_STATE = 42
    # Train the RF model
    best_model = RandomForestClassifier(n_estimators=N_ESTIMATORS, criterion=CRITERION, 
                               max_depth=MAX_DEPTH, max_features=MAX_FEATURE,
                               random_state=RANDOM_STATE, n_jobs=n_jobs)
    best_model.fit(X_train, y_train)
    # Return feature importance
    return best_model.feature_importances_

def compute_threshold(min_value, max_value, percentage):
    """Calculate the threshold T based on the percentage within the range.

    Args:
        min_value (float): Minimum value.
        max_value (float): Maximum value.
        percentage (float): Percentage of difference to consider.

    Returns:
        float: Threshold T.
    """
    return ((max_value - min_value) * percentage) + min_value

def feature_selection_rfc(X_train, feature_importances, percentage=None):
    """Removes less important features based on a RandomForestClassifier.

    Args:
        X_train (DataFrame): Observations.
        feature_importances (ndarray): Importances of features.
        percentage (float): Percentage of difference to consider.

    Returns:
        DataFrame: Modified X_train by eliminating features with importance <= T.
        List: list of the name columns that are dropped
    """
    # Calculate the threshold T
    if percentage != None:
        treshold = compute_threshold(min_value=min(feature_importances),
                                    max_value=max(feature_importances),
                                    percentage=percentage)
    else:
        treshold = 0
    print('Threshold computed: ', treshold)

    # Create a list of indexes to delete
    indices_to_delete = [index for index, score in enumerate(feature_importances) if score <= treshold]
    # Save name columns
    columns_to_drop = X_train.columns[indices_to_delete]
    # Drop columns with specified indexes
    X_train = X_train.drop(X_train.columns[indices_to_delete], axis=1)
    return X_train, columns_to_drop