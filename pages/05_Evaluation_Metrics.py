# pages/05_Evaluation_Metrics.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from utils.dataloader import load_data
from utils.feature_selection import FeatureSelector
from utils.logistic_regression import LogisticModel
from utils.evaluation import evaluate_model

st.title("🎯 Model Evaluation Metrics")

# Setup
df = load_data()
target = "track"
grades = [7, 8, 9, 10]
grade = st.selectbox("Select Grade Level", grades)

selector = FeatureSelector(df, target)
selected_features = selector.select_features(grade)

if not selected_features:
    st.warning("No significant features found for this grade.")
    st.stop()

model = LogisticModel(df, target)
model.train_model(grade, selected_features)

# Get predicted probabilities
y_proba = model.model.predict_proba(model.X_test)

# Evaluate
eval_results = evaluate_model(model.y_test, model.y_pred, y_proba, model)

# Display
st.title("Evaluation Metrics")
st.subheader("Accuracy")
st.write(f"{eval_results['accuracy'] * 100:.2f}%")

st.subheader("Confusion Matrix")
st.dataframe(eval_results["confusion_matrix_df"])

st.subheader("Classification Report")
st.dataframe(eval_results["classification_report"])

# Plot ROC Curve if binary
if eval_results["roc_curve"][0] is not None:
    fpr, tpr = eval_results["roc_curve"]
    auc_score = eval_results["roc_auc"]

    st.subheader("ROC Curve")

    # Create Plotly figure
    fig = go.Figure()

    # Add ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {auc_score:.2f}", line=dict(color='blue')))

    # Add diagonal line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='white', dash='dash')))

    # Layout
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        width=700,
        height=500
    )

    st.plotly_chart(fig)