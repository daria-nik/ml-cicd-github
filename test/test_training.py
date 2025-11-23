from src.train import train

def test_train_runs():
    """Basic test: model should train and produce accuracy > 0."""
    acc = train()
    assert acc > 0.5
