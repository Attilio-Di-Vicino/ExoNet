from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

def grid_search_param_optimization(X_train, y_train, n_jobs=-1):
    """Performs hyperparameter tuning using Grid Search.

    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_test (DataFrame): Test features.
        y_test (Series): Test labels.

    Returns:
        tuple: Best parameters, best cross-validation score, and the best estimator.
    """
    rf_classifier = RandomForestClassifier(n_jobs=n_jobs)
    # hyperparameters
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_features': [2, 3, 4, 5],
        'max_depth': [2, 5, 8, 11, None],
        'min_samples_split': [2, 3, 5, 6, 7, 9]
    }
    # Metric to use for evaluation
    scoring = {'Accuracy': make_scorer(accuracy_score)}
    # Grid Search
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                               scoring=scoring, refit='Accuracy', cv=5, verbose=2, n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_