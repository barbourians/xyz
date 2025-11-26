#
# Model Solution - Python Version
# Converted from R script for logistic regression analysis
#

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('PassFail.csv')
print(data.info())
print(data.head())

# Convert Pass variable to integer (already 0/1 in CSV)
# In Python/sklearn, this is handled automatically
print("\nData types:")
print(data.dtypes)

# Build the model
X = data[['Hours']]  # Features (needs to be 2D array)
y = data['Pass']     # Target variable

model = LogisticRegression()
model.fit(X, y)

# Display model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Hours coefficient: {model.coef_[0][0]:.4f}")

# What is the probability of passing with 2 hours of study?
hours_test = np.array([[2.0]])
prob = model.predict_proba(hours_test)[:, 1]
print(f"\nProbability of passing with 2 hours: {prob[0]:.4f} ({prob[0]*100:.2f}%)")

# Predicting for several values at the same time
hours_multiple = np.array([[1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5]])
prob_pass = model.predict_proba(hours_multiple)[:, 1]
print("\nProbabilities for multiple hour values:")
for hours, prob in zip(hours_multiple.flatten(), prob_pass):
    print(f"{hours} hours: {prob:.4f} ({prob*100:.1f}%)")

# Classification means converting probabilities into classes
# Important parameter is the threshold - let's say it's 0.5
print("\nClassification with threshold 0.5:")
classifications_binary = np.where(prob_pass > 0.5, 1, 0)
print("Binary (1/0):", classifications_binary)

classifications_bool = np.where(prob_pass > 0.5, True, False)
print("Boolean:", classifications_bool)

classifications_text = np.where(prob_pass > 0.5, 'PASS', 'FAIL')
print("Text:", classifications_text)

# Probabilities to pass the exam for in-sample values
prob_pass_insample = model.predict_proba(X)[:, 1]
print("\nIn-sample probabilities:")
print(prob_pass_insample)

# Plot them
plt.figure(figsize=(10, 6))
plt.scatter(data['Hours'], prob_pass_insample, color='blue', alpha=0.6)
plt.plot(data['Hours'], prob_pass_insample, 'b-', alpha=0.3)
plt.xlabel('Hours of Study')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Hours vs Probability of Passing')
plt.grid(True, alpha=0.3)
plt.savefig('logistic_regression_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as 'logistic_regression_plot.png'")

# Classification of in-sample values
classes = np.where(prob_pass_insample > 0.5, 1, 0)
print("\nPredicted classes for in-sample data:")
print(classes)

# Add predicted classes into the original dataset
data['PredictedPass'] = classes
print("\nDataset with predictions:")
print(data)

# Calculate the accuracy of the model
accuracy = accuracy_score(data['Pass'], data['PredictedPass'])
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Alternative way to calculate accuracy
accuracy_alt = np.sum(data['PredictedPass'] == data['Pass']) / len(data)
print(f"Alternative calculation: {accuracy_alt:.4f} ({accuracy_alt*100:.2f}%)")

# Build confusion matrix
cm = confusion_matrix(data['Pass'], data['PredictedPass'])
print("\nConfusion Matrix:")
print("                Predicted")
print("              0 (Fail)  1 (Pass)")
print(f"Actual 0  [{cm[0,0]:>8}  {cm[0,1]:>8}]")
print(f"Actual 1  [{cm[1,0]:>8}  {cm[1,1]:>8}]")

# Calculate performance metrics manually
TP = np.sum((data['PredictedPass'] == data['Pass']) & (data['Pass'] == 1))
TN = np.sum((data['PredictedPass'] == data['Pass']) & (data['Pass'] == 0))
FP = np.sum((data['PredictedPass'] != data['Pass']) & (data['Pass'] == 0))
FN = np.sum((data['PredictedPass'] != data['Pass']) & (data['Pass'] == 1))

print("\nPerformance Metrics:")
print(f"True Positives (TP):  {TP}")
print(f"True Negatives (TN):  {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Calculate rates
TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f"\nTrue Positive Rate (Sensitivity): {TPR:.4f}")
print(f"False Positive Rate: {FPR:.4f}")

print("\n--- Analysis Complete ---")
