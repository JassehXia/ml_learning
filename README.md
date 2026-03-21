# Machine Learning Journey

This repository contains my step-by-step progress as I learn the mathematical and practical foundations of Machine Learning. The goal is to build an intuition for how algorithms learn from data by building them from scratch before moving on to powerful, modern libraries.

## 📁 Project Structure

Currently, the project is organized into learning modules.

### Module One: Foundations of Machine Learning
This module covers the progression from simple linear algebra to deep neural networks and tree-based ensemble models, primarily using clinical data (e.g., Breast Cancer dataset).

*   **`lesson_one.py`**: Basic Linear Regression using NumPy. Implements Gradient Descent manually to calculate the Mean Squared Error (MSE) gradients for a single variable.
*   **`lesson_two.py`**: Logistic Regression. Introduces the Sigmoid activation function and Binary Cross-Entropy (Log-Loss) to predict binary outcomes. Covers essential clinical metrics like Precision, Recall, and the F1 Score.
*   **`lesson_three.py`**: A 2-Layer Neural Network from scratch. Introduces Hidden Layers, ReLU activation, Forward Propagation, and the Chain Rule calculus required for Backpropagation. Also implements L2 Regularization.
*   **`lesson_four.py`**: Tree-Based Models. Transitions to using `scikit-learn` to build Decision Trees and Random Forests. Introduces the $F_2$ Score to prioritize Recall, and utilizes SHAP (`TreeExplainer`) to interpret global feature importance and understand the model's inner workings.

## 📝 Notes
For a detailed breakdown of the mathematical formulas, concepts, and evaluation metrics learned in each lesson, please see the [ml_notes.md](ml_notes.md) file.
