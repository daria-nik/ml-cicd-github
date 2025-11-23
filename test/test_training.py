from src.train import train

def test_train_runs():
    acc = train()
    assert acc > 0.5
