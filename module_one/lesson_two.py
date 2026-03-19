"""
Module 2: Logistic Regression for Clincal Classificaton
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
data = load_breast_cancer()
# Let's start with just one feature: 'mean radius'
X = data.data
y = data.target

# Find the mean and std for each column
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Normalize each feature by its own mean and std
X_scaled = (X-X_mean)/X_std

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

class LogisticRegressor:
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.eta = learning_rate
        self.n_iterations = iterations
        self.b0 = 0
        self.loss_history = []
    
    def sigmoid(self, z):
        # Math: σ(z) = 1 / (1 + e^-z)
        # Why: Squashes any real number into a probability [0, 1]
        return 1 / (1 + np.exp(-z))
    
    def predict_prob(self, X):
        # Linear part: z = β0 + β1*x
        z = X @ self.W + self.b0
        # Apply activation: ŷ = σ(z)
        return self.sigmoid(z)
    
    def predict_class(self, X, threshold =0.2):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int)
    
    def fit(self, X, y):
        N, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b0 = 0

        for _ in range(self.n_iterations):
            # 1. Get probabilities (0 to 1)
            y_prob = self.predict_prob(X)

           

            # 2. Calculate Gradients
            # Formula: 1/N * Σ(y_prob - y) * x
            db0 = np.mean(y_prob - y)
            dW = (1 / N) * (X.T @ (y_prob - y))

            # 3. Update weights
            self.b0 -= self.eta * db0
            self.W -= self.eta * dW

            # 4. Calculate the log-loss for history
            # Prevent log(0) error
            y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
            loss = -np.mean(y*np.log(y_prob) + (1 - y) * np.log(1 - y_prob))
            self.loss_history.append(loss)

model = LogisticRegressor(learning_rate=0.01, iterations = 500)

# 1. Train the model on training data
model.fit(X_train, y_train)

# 2. Predict on the "unseen" test data
predictions = model.predict_class(X_test)
fp = np.sum((predictions == 1) & (y_test == 0))
fn = np.sum((predictions == 0) & (y_test == 1))
tn = np.sum((predictions == 0) & (y_test == 0))
tp = np.sum((predictions == 1) & (y_test == 1))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")
print(f"True Positives: {tp}")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1_score * 100:.2f}%")

# 3. Calculate Accuracy
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

