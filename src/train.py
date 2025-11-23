from src.preprocess import load_data
from src.model import create_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train():
    """Train model and return accuracy"""
    df = load_data()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = create_model()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy:", acc)

    return acc


if __name__ == "__main__":
    train()
