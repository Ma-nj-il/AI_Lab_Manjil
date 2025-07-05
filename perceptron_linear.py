import numpy as np

np.random.seed(42)

X = np.random.rand(10, 3)
# Compute target y = 2x₁ + 3x₂ - x₃ + 5
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate

    def predict(self, X):
        # Linear activation: output = w·x + b
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict(X)
            # Compute mean squared error
            mse = np.mean((y_pred - y) ** 2)
            print(f"Epoch {epoch + 1}, MSE: {mse:.6f}")

            # Compute gradients
            error = y_pred - y
            dw = np.dot(X.T, error) / len(y)
            db = np.mean(error)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

perceptron = Perceptron(input_size=3, learning_rate=0.01)
perceptron.train(X, y, epochs=100)

print("\nFinal Weights:", perceptron.weights)
print("Final Bias:", perceptron.bias)
print("Target Weights: [2, 3, -1]")
print("Target Bias: 5")