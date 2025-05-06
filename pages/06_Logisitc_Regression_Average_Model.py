import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from utils.average_logreg import AverageLogisticModel
from utils.dataloader import load_data
from utils.evaluation import evaluate_model

st.set_page_config(page_title="Average Logistic Model", layout="wide")
st.title("Average Logistic Regression Model per Grade Level")

# Load your data
df = load_data()

# Initialize the model class
model = AverageLogisticModel(df, target_column="track")

# Train and collect results
result_rows = []
evaluation_results = {}  # Store evaluations per grade

for grade in range(7, 11):
    avg_features, track, y_test, y_pred, y_proba = model.train_model(grade, return_eval=True)
    
    result_rows.append({
        "Grade Level": f"G{grade}",
        **{subject.replace(f"_g{grade}", "").replace("avg_", "").capitalize(): round(model.df[subject].mean(), 2)
           for subject in avg_features if subject.startswith("avg_")},
        "Leaning Track": track
    })

    evaluation_results[grade] = evaluate_model(y_test, y_pred, y_proba, model)

# Display as DataFrame
results = pd.DataFrame(result_rows)
st.write("### Subject Averages and Track Leaning per Grade Level")
st.dataframe(results)

# Prepare the dataframe: melt to long format
plot_df = results.melt(
    id_vars=["Grade Level", "Leaning Track"],
    var_name="Subject",
    value_name="Average Grade"
)

# Evaluation Section
st.write("### Model Evaluation per Grade Level")
grades = [7, 8, 9, 10]
grade = st.selectbox("Select Grade Level", grades)

if grade in evaluation_results:
    ev = evaluation_results[grade]

    # st.write(f"**Accuracy (G{grade}):**", round(ev['accuracy'], 4))
    st.subheader("Accuracy")
    st.write(f"{ev['accuracy'] * 100:.2f}%")

    st.write("**Confusion Matrix:**")
    st.dataframe(ev['confusion_matrix_df'])

    st.write("**Classification Report:**")
    st.dataframe(ev['classification_report'])

    # Plot ROC Curve
    if ev["roc_curve"][0] is not None:
        fpr, tpr = ev["roc_curve"]
        auc_score = ev["roc_auc"]

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

# # Optional: Export CSV
# st.download_button(
#     label="Download Result as CSV",
#     data=results.to_csv(index=False).encode('utf-8'),
#     file_name='average_grade_track_prediction.csv',
#     mime='text/csv'
# )
