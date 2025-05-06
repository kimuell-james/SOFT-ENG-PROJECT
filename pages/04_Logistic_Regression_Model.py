import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.dataloader import load_data
from utils.feature_selection import FeatureSelector
from utils.logistic_regression import LogisticModel

st.title("🤖 Logistic Regression Model")

df = load_data()
target_column = "track"
grades = [7, 8, 9, 10]
grade = st.selectbox("Select Grade Level", grades)

selector = FeatureSelector(df, target_column)

# --- CUMULATIVE FEATURE SELECTION LOGIC ---
included_grades = list(range(7, grade + 1))
final_features = []

for g in included_grades:
    feats = selector.select_features(g)
    prefix = f"g{g}_"
    grade_feats = [f for f in feats if f.startswith(prefix)]
    final_features.extend(grade_feats)

# Add age and gender only once
if 'age' in df.columns:
    final_features.append('age')
if 'gender' in df.columns:
    final_features.append('gender')

# Remove duplicates while preserving order
final_features = list(dict.fromkeys(final_features))

# --- Train Model ---
model = LogisticModel(df, target_column)
model.train_model(grade, final_features)

# Filter out non-subject features when calculating stats
filtered_features = [feat for feat in final_features if feat not in ['gender', 'age']]

subject_stats = model.calculate_grade_statistics(filtered_features)
subject_grade_total = subject_stats['Academic (Count)'].sum() + subject_stats['TVL (Count)'].sum()

# --- Display Outputs ---
st.title(" Grade Statistics")

st.subheader("📌 Grade Statistics by Predicted Track")
st.subheader("Track Prediction Distribution")
for track, count in model.prediction_counts.items():
    st.write(f"**{track}:** {count} ({model.track_distribution_data[track]:.2f}%)")
st.write(f"**Total Predictions:** {model.total_predictions}")
st.write(f"**Total Subject Grade Entries:** {int(subject_grade_total)}")

# st.write("### Subject Grades (Grouped by Predicted Track)")
# st.write(subject_stats)

st.write("### Subject Grades (Grouped by Predicted Track)")

# Split subject_stats into Academic and TVL
# Separate Academic and TVL statistics
academic_cols = [col for col in subject_stats.columns if "Academic" in col]
tvl_cols = [col for col in subject_stats.columns if "TVL" in col]

# Keep the index (subject names) and include only relevant columns
academic_table = subject_stats[academic_cols].copy()
academic_table.index = subject_stats.index

tvl_table = subject_stats[tvl_cols].copy()
tvl_table.index = subject_stats.index

st.markdown("#### Academic")
st.dataframe(academic_table)

st.markdown("#### TVL")
st.dataframe(tvl_table)

# Show features used
st.markdown(f"##### ✅ Features Used for Grade {grade}")
if final_features:
    st.markdown(f"**Grade {grade}:** " + ", ".join(final_features))
else:
    st.info(f"No significant features found for Grade {grade}.")

# Get coefficients
coeff_df = model.get_coefficients()

if not coeff_df.empty:
    st.subheader("Logistic Regression Coefficients")
    st.markdown(
        "- **Positive coefficient**: Subject leans more toward Academic track\n"
        "- **Negative coefficient**: Subject pushes prediction toward TVL\n"
    )

    # Create a Plotly bar chart
    fig = px.bar(coeff_df, 
                 x="Feature", 
                 y="Coefficient", 
                 color="Coefficient",  # Color by the coefficient values
                 color_continuous_scale="RdYlGn",  # Use a diverging color scale (red to green)
                 labels={"Coefficient": "Coefficient Value", "Feature": "Subject"},
                 title="Logistic Regression Coefficients by Subject")

    # Show the chart
    fig.update_layout(
        xaxis_title="Subject",  # Set the x-axis label
        yaxis_title="Coefficient Value",  # Set the y-axis label
        xaxis_tickangle=-45,  # Rotate the x-axis labels for better readability
        # plot_bgcolor='white'  # Set background color to white for better contrast
    )
    
    st.plotly_chart(fig)
else:
    st.info("No coefficients available. Make sure the model has been trained.")

# GENDER DISTRIBUTION VISUALIZATION
st.subheader("Gender Distribution by Predicted Track")
gender_counts = model.get_gender_distribution_by_track()
st.write("#### Gender Count Table")
st.dataframe(gender_counts)
st.write("#### Bar Chart of Gender Distribution")
st.bar_chart(gender_counts)

# st.subheader("📌 Overall Grade Statistics (All Students)")
# overall_stats = df[final_features].agg(['min', 'max', 'mean', 'median', 'count']).T
# overall_stats.columns = ['Min', 'Max', 'Mean', 'Median', 'Count']
# all_subject_count = overall_stats['Count'].sum()
# # st.write(f"**Total Subject Grades:** {int(all_subject_count)}")

# st.write("### Overall Subject Grades (All Students)")
# st.write(overall_stats)