# train.py простое обучение breast cancer модели

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train():
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)

    # evaluate
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc:.4f}")

    return acc


if __name__ == "__main__":
    train()
