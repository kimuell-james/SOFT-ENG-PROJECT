import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from utils.logistic_regression import LogisticRegressionModel
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/student_data.csv')

# Target column
target_column = 'track'

# Initialize LogisticRegressionModel
logistic_model = LogisticRegressionModel()

# Use grade 7 for evaluation
model, y_test, y_pred, y_prob = logistic_model.train(df, ['g7_math', 'g7_science'], target_column, grade="7")

# Model Evaluation Page
st.title("Model Evaluation")

# Display classification report
st.subheader("Classification Report")
st.write(classification_report(y_test, y_pred))

# Show confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# ROC-AUC Curve
st.subheader("ROC-AUC Curve")
fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_prob[:, 1]))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
st.pyplot(plt)
