import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.feature_selection import FeatureSelector
from utils.dataloader import load_data

st.title("🧠 Feature Selection by Grade Level")

# Load your dataset
df = load_data()
target_column = "track"  # adjust if needed
grades = [7, 8, 9, 10]

# Select grade level
grade = st.selectbox("Select Grade Level", grades)

# Run feature selection
selector = FeatureSelector(df, target_column)
selector.select_features(grade)

# Get raw scores (F-scores) and p-values
scores_df = selector.scores_per_grade[grade]
significant = selector.significant_features[grade]

# Clean scores for plotting (rename "F Score" for display)
plot_scores = scores_df["F Score"].sort_values(ascending=True)

# Plot F-scores of all features
st.subheader(f"F-Scores of All Features for Grade {grade}")
st.bar_chart(plot_scores)

# Show significant features
st.markdown("### ✅ Statistically Significant Features")
if significant:
    st.markdown(f"**Grade {grade}:** " + ", ".join(significant))
else:
    st.info(f"No significant features found for Grade {grade}.")

# Show full score table if desired
if st.checkbox("Show full score table"):
    st.dataframe(scores_df.sort_values("F Score", ascending=False))
