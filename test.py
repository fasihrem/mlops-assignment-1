import pytest
from main import model, X_test, y_test
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8, f"Expected accuracy > 0.8 but got {accuracy:.2f}"

if __name__ == "__main__":
    pytest.main()
