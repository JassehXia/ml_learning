# Machine Learning Journey: Notes & Concepts

This document tracks my progress and key concepts learned throughout the machine learning modules.

## Module One: Foundations of Machine Learning

### Lesson 1: Basic Linear Regression
* **Goal**: Predict a continuous value (e.g., Blood Pressure) based on a continuous feature (e.g., BMI).
* **Core Concepts**:
  * **Standardization (Z-score normalization)**: Transforming features to have a mean of 0 and standard deviation of 1. Prevents features with inherently larger scales from unfairly dominating the learning process.
  * **Linear Hypothesis**: $\hat{y} = \beta_0 + \beta_1x$ (Intercept + Slope * Input).
  * **Mean Squared Error (MSE)**: The loss function used to measure how far the predictions are from the true values.
  * **Gradients (Calculus)**: Calculating the partial derivatives of the MSE with respect to the intercept ($\beta_0$) and slope ($\beta_1$). This tells us which direction to adjust the weights.
  * **Gradient Descent**: The optimization step where we update our parameters: $\beta = \beta - (\eta \cdot \text{gradient})$, where $\eta$ is the learning rate.

### Lesson 2: Logistic Regression
* **Goal**: Binary classification (e.g., predicting Malignant vs. Benign in clinical data).
* **Core Concepts**:
  * **Sigmoid Activation Function**: $\sigma(z) = \frac{1}{1 + e^{-z}}$. Squashes any linear output (from $-\infty$ to $\infty$) into a probability between 0 and 1.
  * **Log-Loss (Binary Cross-Entropy)**: The loss function used for binary probabilities instead of MSE.
  * **Decision Threshold**: Converting the continuous probability into a discrete class (e.g., if probability $\ge 0.5$, predict Class 1).
  * **Evaluation Metrics**:
    * **Confusion Matrix Elements**: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).
    * **Precision**: Out of all predicted positives, how many were actually positive?
    * **Recall**: Out of all actual positives, how many did we correctly find?
    * **F1 Score**: The harmonic mean of precision and recall (useful for imbalanced datasets).
    * **Accuracy**: Overall correctness of the model.

### Lesson 3: Neural Networks (Forward & Back Propagation)
* **Goal**: Building a Multi-Layer Perceptron (MLP) from scratch to handle more complex, non-linear relationships.
* **Core Concepts**:
  * **Network Architecture**: 
    * Input Layer (features).
    * Hidden Layer (e.g., 10 nodes).
    * Output Layer (1 node for binary classification).
  * **Forward Pass**: Pushing the data through the layers.
    * Uses weight matrices (`W1`, `W2`) and bias vectors (`b1`, `b2`) instead of single scalar values.
    * **ReLU (Rectified Linear Unit)** Activation: Used in the hidden layer (`max(0, z)`). Allows the network to learn non-linear patterns.
    * **Sigmoid** Activation: Used on the final output layer to get a probability.
  * **Backward Pass (Backpropagation)**: 
    * Using the Chain Rule from calculus to propagate the error backward from the output layer to the hidden layers.
    * Calculating how much *each individual weight and bias* in every layer contributed to the final error.
  * **L2 Regularization ($\lambda$)**: Adding a penalty to the weight updates to prevent the weights from growing too large, which helps prevent overfitting.

### Lesson 4: Decision Trees and Random Forests
* **Goal**: Using tree-based models for classification and understanding feature importance.
* **Core Concepts**:
  * **Decision Trees (CART)**: Algorithms that split data by asking a series of "Yes/No" questions. They are highly interpretable but prone to overfitting if the tree grows too deep (controlled by `max_depth`).
  * **Random Forests**: An "ensemble" method that creates hundreds of different Decision Trees (e.g., `n_estimators=100`) and has them vote on the outcome. This vastly improves accuracy and prevents overfitting compared to a single tree.
  * **Evaluation - F-beta Score**: Using the `fbeta_score` metric (like the $F_2$ score, where $\beta > 1$) to place more weight on Recall than Precision. This is crucial in medical diagnoses where false negatives (missing cancer) are far worse than false positives.
  * **Interpretability with SHAP (SHapley Additive exPlanations)**: A powerful tool to "explain" how complex models make decisions. 
    * `TreeExplainer` specifically calculates the exact mathematical contribution of each feature to the model's output for tree-based models.
    * `summary_plot` provides a global view of feature importance (e.g., cell radius or perimeter) and how their values push the model toward a specific prediction across the entire dataset.
