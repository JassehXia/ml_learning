"""
Lesson Three: Neural Networks with Forward and Back Propagation
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target.reshape(-1, 1)

X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
X_scaled = (X-X_mean) / X_std

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class NeutalNetwork:
    def __init__(self, input_size=30, hidden_size=10, output_size=1, learning_rate=0.01):
        self.eta = learning_rate
        self.lambd = 0.1

        # Layer 1: Input -> Hidden (30 x 10)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: Hidden -> Output
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        # Pass through Layer 1
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        # Pass through Layer 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2
    
    def backward(self, X, y):
        N = X.shape[0]

        # 1) Error at Output Layer (dZ2)
        # Difference between the guess and the truth
        dZ2 = self.A2 - y

        # 2) Gradients for W2 and b2
        # error * input from prev. layer
        dW2 = (1 / N) * (self.A1.T @ dZ2)
        db2 = (1/ N) * np.sum(dZ2, axis=0, keepdims=True)

        # 3) Error at the Hidden Layer (dZ1)
        # pass the error back to W2
        # multiply by the derivative of the ReLU
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0) # Derivative of ReLU: 1 if Z > 0, else 0

        # 4) Gradients for W1 and b1
        dW = (1/N) * (X.T @ dZ1)
        db1 = (1/N) * np.sum(dZ1, axis=0, keepdims=True)
        dW2 = (1/N) * (self.A1.T @ dZ2) + (self.lambd/N) * self.W2
        dw1 = (1/N) *(X.T @ dZ1) + (self.lambd / N) * self.W1

        # 5. Update Weights (Gradient Descent)
        self.W2 -= self.eta * dW2
        self.b2 -= self.eta * db2
        self.W1 -= self.eta * dW
        self.b1 -= self.eta * db1

    def fit(self, X, y, iterations=1000):
        self.loss_history=[]
        for i in range(iterations):
            # forward pass
            y_hat = self.forward(X)

            # backward pass to update weights
            self.backward(X, y)

            # calculate inary cross-entropy loss
            # clip to avoid log(0) erorrs
            y_hat = np.clip(y_hat, 1e-15, 1-1e-15)
            loss = -np.mean(y*np.log(y_hat) + (1-y) * np.log(1-y_hat))
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
            
nn = NeutalNetwork(input_size=30, hidden_size=10, output_size=1, learning_rate=0.01)

nn.fit(X_train, y_train, iterations=2000)

test_preds = (nn.forward(X_test) > 0.5).astype(int)
accuracy = np.mean(test_preds == y_test)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")