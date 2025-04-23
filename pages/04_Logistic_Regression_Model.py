import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import numpy as np
from utils.logistic_regression import LogisticRegressionModel
from utils.feature_selection import FeatureSelector
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/synthetic_student_data.csv')

# Target column
target_column = 'track'  # Example target column, replace with your actual target

# Initialize FeatureSelector class
feature_selector = FeatureSelector(df, target_column)

# Feature selection
selected_features = feature_selector.select_features(grade="7")

# Train Logistic Regression model
logistic_model = LogisticRegressionModel()
model, y_test, y_pred, y_prob = logistic_model.train(df, selected_features, target_column, grade="7")

# Display model metrics
st.title("Logistic Regression Model")

st.write("Model trained successfully!")

# Show the model coefficients to understand feature importance
st.write("Model Coefficients (Feature Importance):")
coefficients = model.coef_[0]
features = selected_features
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
})
feature_importance['Importance'] = np.abs(feature_importance['Coefficient'])  # Absolute value of coefficients
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

st.write(feature_importance)

# Visualize feature importance as a bar chart
st.write("Feature Importance Visualization:")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Logistic Regression')
st.pyplot(plt)

# Show the grades/subject combinations that contribute to prediction
st.write("Grades/Subject Combinations Contributing to Prediction:")

# Example: you can display how grades and subjects (e.g., Math, Science) contribute to the model's prediction
# For simplicity, let's display the top important features and explain which subjects/grades they correspond to.
# This can be customized based on your dataset.

for feature in feature_importance['Feature'].head(10):  # Show top 10 features
    st.write(f"### {feature}")
    # Assuming the feature names are related to subjects, you can include logic to display more info about each feature
    st.write(f"Grade/Subject Combination: {feature}")  # Customize this according to your actual features
