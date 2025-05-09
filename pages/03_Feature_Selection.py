import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import chi2_contingency
from utils.feature_selection import FeatureSelector
from utils.dataloader import load_data

st.title("🧠 Feature Selection")

# Load dataset
df = load_data()
target_column = "track"
grades = [7, 8, 9, 10]

# Select grade level
grade = st.selectbox("Select Grade Level", grades)

selector = FeatureSelector(df, target_column)

# Select features for the chosen grade
significant_features = selector.select_features(grade)

# Get raw scores (F-scores for numerical features and p-values for categorical features)
scores_df = selector.scores_per_grade[grade]

# Clean scores for plotting (plot both F-scores and Chi-square p-values together)
plot_scores = scores_df["F-Score"].sort_values(ascending=True)

# Plot F-scores of all features
st.subheader("Numerical Feature Significance (ANOVA F-Scores)")
st.markdown(f"- **F-score**: Measures how much the feature discriminates between different tracks (higher is better).")
st.bar_chart(plot_scores)

st.subheader("Categorical Feature Significance (Chi-Square p-value)")
if "gender" in df.columns:
    # Compute Chi-Square p-value
    contingency = pd.crosstab(df['gender'], df[target_column])
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    st.markdown(f"**Chi-Square Test P-Value:** `{p}`")
    st.markdown(f"""
    **How to Decide Significance:**
    - If **p < 0.05**, the feature is **significant**.
    - If **p ≥ 0.05**, the feature is **not significant**.
    """)

    # Show significance
    if p < 0.05:
        st.success("Gender is statistically significant in predicting the track (p < 0.05).")
    else:
        st.warning("Gender is not statistically significant (p ≥ 0.05).")

# Show significant features
st.subheader("✅ Statistically Significant Features")
if significant_features:
    st.markdown(f"**Grade {grade}:** " + ", ".join(significant_features))
else:
    st.info(f"No significant features found for Grade {grade}.")

# Show full score table
with st.expander("📄 Show Full Score Table"):
    st.dataframe(scores_df.sort_values("F-Score", ascending=False))


