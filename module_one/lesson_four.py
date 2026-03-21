"""
Lesson Four: Decision Trees, Random Forests, and SHAP
"""

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
import shap
import matplotlib.pyplot as plt
import numpy as np


# Load the medical data
data = load_breast_cancer()
print(data.target_names)
X, y = data.data, data.target

# Split to train and test groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

# Random Forest of 100 Trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print(f"Single Tree Accuracy: {accuracy_score(y_test, tree_preds):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")

print(f"Single Tree F2 Score: {fbeta_score(y_test, tree_preds, beta=1.414):.4f}")
print(f"Random Forest F2 Score: {fbeta_score(y_test, rf_preds, beta=1.414):.4f}")

# 1. Create the explainer
explainer = shap.TreeExplainer(rf_model)

# 2. Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# 3. Handle the 'Malignant' class and first patient
# index [1] is Malignant, [0, :] is the first patient in the test set

# 1. Create the explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# 1. Set index to 0 for Malignant
target_class_index = 0 

# 2. Extract SHAP values for all patients for the 'Malignant' class
if isinstance(shap_values, list):
    # Older SHAP/sklearn format
    shap_values_to_plot = shap_values[target_class_index]
else:
    # Modern SHAP format: [all_patients, all_features, malignant_class]
    shap_values_to_plot = shap_values[:, :, target_class_index]

# 3. Generate the Global Summary Plot
# Note: This will replace the previous window
shap.summary_plot(shap_values_to_plot, X_test, feature_names=data.feature_names)

# 4. Display the plot
plt.show()