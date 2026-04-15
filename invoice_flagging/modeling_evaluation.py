from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV # Changed this
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    # We keep the same grid, but RandomizedSearchCV will sample from it
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)

    # Changed GridSearchCV to RandomizedSearchCV
    # n_iter=20 means it picks 20 random combinations instead of doing all 216
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,          
        scoring=scorer,
        cv=3,               # Reduced folds to 3 for even more speed
        n_jobs=-1,
        verbose=1,          # Set to 1 so you can see progress
        random_state=42
    )

    random_search.fit(X_train, y_train)
    return random_search

def evaluate_classifier(model, X_test, y_test, model_name):
    # This remains the same, but ensure you pass 'best_estimator_' when calling it
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)

# --- How to call it properly in your main script ---
# search_result = train_random_forest(X_train_scaled, y_train)
# evaluate_classifier(search_result.best_estimator_, X_test_scaled, y_test, "Random Forest (Random Search)")