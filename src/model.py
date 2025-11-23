from sklearn.ensemble import RandomForestClassifier

def create_model():
    """Return a simple RandomForest model."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
