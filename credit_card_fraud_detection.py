import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load dataset
dataset = pd.read_csv('creditcard.csv')

# Display basic info and first few rows
print(dataset.head(5))
dataset.info()

# Check class distribution
print(dataset['Class'].value_counts())

# Prepare features (X) and target (y)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Handle missing values in the dataset
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# Create and train Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# Make a prediction on a sample input (add appropriate number of features)
sample_input = [[0.0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, 0.0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053, 149.62, 0]]
print(classifier.predict(sample_input))  # Predict using the trained model

# Predict on test data
y_pred = classifier.predict(x_test)

# Convert probabilities to binary predictions (adjust threshold as needed)
y_pred_proba = classifier.predict_proba(x_test)  # Get probabilities for each class
y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)  # Convert probabilities to binary predictions

# Round the imputed values in y_test to the nearest integer (0 or 1)
y_test = np.round(y_test).astype(int)

# Calculate and display confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the trained model using joblib
joblib.dump(classifier, 'credit_card_model.pkl')
print("Model saved as 'credit_card_model.pkl'.")
