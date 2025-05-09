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
from utils.balanced_logreg import BalancedLogisticModel
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

model = BalancedLogisticModel(df, target)
model.train_model(grade, selected_features)

# Get predicted probabilities
y_proba = model.model.predict_proba(model.X_test)

# Evaluate
eval_results = evaluate_model(model.y_test, model.y_pred, y_proba, model)

# Display
st.title("Evaluation Metrics")
st.subheader("Model Overall Accuracy")
st.subheader(f"{eval_results['accuracy'] * 100:.2f}%")
st.write(f"The model achieved an overall accuracy of {eval_results['accuracy'] * 100:.2f}%, meaning it accurately identified the appropriate Senior High School track for the majority of students.")

st.subheader("Confusion Matrix")
st.dataframe(eval_results["confusion_matrix_df"])

# Extract values from the confusion matrix
cm = eval_results["confusion_matrix_df"].values
true_academic, false_academic = cm[0]
false_tvl, true_tvl = cm[1]

# Add descriptive summary
st.markdown(f"""
**Interpretation:**

- Out of **{true_academic + false_academic}** students who actually chose **Academic**, the model correctly predicted **{true_academic}** and misclassified **{false_academic}** as TVL.
- Out of **{true_tvl + false_tvl}** students who actually chose **TVL**, the model correctly predicted **{true_tvl}** and misclassified **{false_tvl}** as Academic.
""")

st.subheader("Classification Report")
st.dataframe(eval_results["classification_report"])

# Extract classification report DataFrame
report_df = eval_results["classification_report"]

# Round for cleaner display if needed
report_df_rounded = report_df.round(2)

# Get values
academic = report_df_rounded.loc["Academic"]
tvl = report_df_rounded.loc["TVL"]
accuracy = report_df_rounded.loc["accuracy", "f1-score"] if "accuracy" in report_df_rounded.index else None

# Display dynamic summary
st.markdown(f"""
**Interpretation:**

- **Academic Track**: The model achieved a precision of **{academic['precision']}**, recall of **{academic['recall']}**, and F1-score of **{academic['f1-score']}**.
- **TVL Track**: The model achieved a precision of **{tvl['precision']}**, recall of **{tvl['recall']}**, and F1-score of **{tvl['f1-score']}**.
- **Overall Accuracy**: **{accuracy}** (based on total predictions across both tracks).
""")


# Plot ROC Curve
if eval_results["roc_curve"][0] is not None:
    fpr, tpr = eval_results["roc_curve"]
    auc_score = eval_results["roc_auc"]

    st.subheader("ROC Curve")

    # Create Plotly figure
    fig = go.Figure()
    # ROC curve
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {auc_score:.2f}", line=dict(color='blue')))
    # Add diagonal line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(color='red', dash='dash')))
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

st.markdown(f"""
The **ROC (Receiver Operating Characteristic) Curve** illustrates the model's ability to distinguish between the Academic and TVL tracks. The **AUC (Area Under the Curve)** value is **{auc_score:.2f}**, which indicates the model's overall performance:
- An AUC of **1.0** represents perfect classification,
- An AUC of **0.5** suggests no discriminatory power (equivalent to random guessing).

With an AUC of **{auc_score:.2f}**, the model demonstrates {"excellent" if auc_score >= 0.9 else "good" if auc_score >= 0.8 else "fair" if auc_score >= 0.7 else "poor"} capability in correctly classifying the student track.
""")
