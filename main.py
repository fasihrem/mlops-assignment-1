import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Loading the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
# Converting to binary classification [Setosa vs other flowers]
y = (iris.target == 0).astype(int)

# Train/test data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=64)

# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating results for the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

