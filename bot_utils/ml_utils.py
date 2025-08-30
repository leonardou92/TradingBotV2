import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def entrenar_modelo_rf(X, y):
    best_score = 0
    clf = None
    if len(np.unique(y)) > 1:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 8, None],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid.fit(X, y)
        clf = grid.best_estimator_
        best_score = grid.best_score_
    return clf, best_score
