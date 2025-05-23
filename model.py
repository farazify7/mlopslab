from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load Iris dataset
X, y = load_iris(return_X_y=True)

# Initialize Logistic Regression model with max_iter=200
clf = LogisticRegression(max_iter=200)

# Fit the model on the dataset
clf.fit(X, y)

# Print confirmation
print("Model trained")
