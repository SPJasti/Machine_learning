import numpy as np


def gradient_descent(x, y, learning_rate, num_iterations):
    num_samples = len(x)
    num_features = x.shape[1]
    theta = np.zeros(num_features + 1)  # Initialize parameters with zeros
    x = np.concatenate((np.ones((num_samples, 1)), x), axis=1)  # Add a column of ones for the bias term

    for _ in range(num_iterations):
        predictions = np.dot(x, theta)
        errors = predictions - y
        gradient = (1/num_samples) * np.dot(x.T, errors)
        theta -= learning_rate * gradient

    return theta


def predict(x, theta):
    num_samples = len(x)
    num_features = x.shape[1]
    x = np.concatenate((np.ones((num_samples, 1)), x), axis=1)  # Add a column of ones for the bias term
    predictions = np.dot(x, theta)
    return predictions


# we are going to use some random data only for today from next class I will need you get a dataset from Kaggle
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 3 + np.random.randn(100, 1)

# Split the data into training and test sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Perform gradient descent to learn the parameters
learning_rate = 0.01
num_iterations = 1000
theta = gradient_descent(X_train, y_train, learning_rate, num_iterations)

# Make predictions on the test set
predictions = predict(X_test, theta)

# Print the learned parameters
print("Learned parameters:")
print(f"theta0 = {theta[0]}, theta1 = {theta[1]}")

# Print some predictions and their corresponding true values
print("\nPredictions:")
for i in range(5):
    print(f"Prediction: {predictions[i]}, True value: {y_test[i]}")
