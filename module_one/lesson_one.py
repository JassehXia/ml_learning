"""
Lesson One: Basic Linear Regression For BMI and Blood Pressure
We will be building a simple linear regression model using NumPy
to understand the math behind machine learning.
"""

import numpy as np

# Clinical Data
bmi = np.array([18.5, 22.0, 24.5, 26.1, 30.2, 32.5, 35.0, 40.1])
bp  = np.array([110, 115, 122, 125, 135, 140, 148, 160])

# Standardization: z = (x - mu) / sigma
# Why: This transforms the features to have a mean of 0 and std of 1.
# It prevents features with larger scales from dominating the gradient update.
bmi_scaled = (bmi - np.mean(bmi)) / np.std(bmi)

class LinearRegression:
    def __init__(self, learning_rate = 0.01, iterations = 1000):
        self.eta = learning_rate
        self.n_iterations = iterations
        self.b0 = 0 # Intercept
        self.b1 = 0 # Slope
        self.loss_history = []

    def predict(self, X):
        # Linear hypothesis: ŷ = β0 + β1 * x
        return self.b0 + self.b1 * X
    
    def fit(self, X, y):
        N = len(y)
        for _ in range(self.n_iterations):
            y_hat = self.predict(X)

            # --- THE CALCULUS: CALCULATING GRADIENTS ---
            # We calculate the partial derivative of the Mean Squared Error (MSE)
            # Loss L = (1/2N) * sum((y - y_hat)^2)
            
            # db0 (Intercept Gradient): ∂L/∂β0 = -1/N * Σ(y - ŷ)
            db0 = -np.mean(y - y_hat)
            
            # db1 (Slope Gradient): ∂L/∂β1 = -1/N * Σ((y - ŷ) * x)
            db1 = -np.mean(X * (y - y_hat))

            # --- THE OPTIMIZATION: GRADIENT DESCENT ---
            # Update Rule: β = β - (η * gradient)
            # We move in the opposite direction of the gradient to minimize error.
            self.b0 -= self.eta * db0
            self.b1 -= self.eta * db1

            

            # Calculate the Mean Squared Error (MSE)
            # 1/N * Σ(y - ŷ)^2
            MSE = np.mean((y-y_hat)**2)

            print(f"b0: {self.b0}, b1: {self.b1}, MSE: {MSE}")  

            # the model is at its most optimal once loss platues 
            if len((self.loss_history)) > 0:
                if abs(self.loss_history[-1] - MSE) < .0001:
                    break

            self.loss_history.append(MSE)

# Start the training
model = LinearRegression(learning_rate=0.1,iterations=500)

# training step
model.fit(bmi_scaled, bp)

# Checking the results
print(f"Final Intercept (b0): {model.b0}")
print(f"Final Slope (b1): {model.b1}")
