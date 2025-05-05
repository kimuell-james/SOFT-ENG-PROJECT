import streamlit as st
import pandas as pd

from utils.dataloader import load_data
from utils.feature_selection import FeatureSelector
from utils.logistic_regression import LogisticModel

st.title("📊 Logistic Regression Model by Grade Level")

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
st.title("📚 Grade Statistics")

st.subheader("📌 Grade Statistics by Predicted Track")
st.subheader("Track Prediction Distribution")
for track, count in model.prediction_counts.items():
    st.write(f"**{track}:** {count} ({model.track_distribution_data[track]:.2f}%)")
st.write(f"**Total Predictions:** {model.total_predictions}")
st.write(f"**Total Subject Grade Entries:** {int(subject_grade_total)}")

st.write("### Subject Grades (Grouped by Predicted Track)")
st.write(subject_stats)

# Show features used
st.markdown(f"### ✅ Features Used for Grade {grade}")
if final_features:
    st.markdown(f"**Grade {grade}:** " + ", ".join(final_features))
else:
    st.info(f"No significant features found for Grade {grade}.")

# GENDER DISTRIBUTION VISUALIZATION
st.subheader("👥 Gender Distribution by Predicted Track")
gender_counts = model.get_gender_distribution_by_track()
st.write("### Gender Count Table")
st.dataframe(gender_counts)
st.write("### Bar Chart of Gender Distribution")
st.bar_chart(gender_counts)


# st.subheader("📌 Overall Grade Statistics (All Students)")
# overall_stats = df[final_features].agg(['min', 'max', 'mean', 'median', 'count']).T
# overall_stats.columns = ['Min', 'Max', 'Mean', 'Median', 'Count']
# all_subject_count = overall_stats['Count'].sum()
# # st.write(f"**Total Subject Grades:** {int(all_subject_count)}")

# st.write("### Overall Subject Grades (All Students)")
# st.write(overall_stats)
