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

# Clean scores for plotting (we'll plot both F-scores and Chi-square p-values together)
plot_scores = scores_df["Score"].sort_values(ascending=True)

# Plot F-scores of all features
st.subheader(f"Scores of All Features for Grade {grade}")
st.bar_chart(plot_scores)

# Show significant features
st.markdown("### ✅ Statistically Significant Features")
if significant_features:
    st.markdown(f"**Grade {grade}:** " + ", ".join(significant_features))
else:
    st.info(f"No significant features found for Grade {grade}.")

if "gender" in df.columns:
    st.subheader("Gender Distribution Across Tracks")

    # Prepare the crosstab data for plotting
    crosstab_df = pd.crosstab(df['gender'], df[target_column])
    crosstab_df = crosstab_df.reset_index().melt(id_vars='gender', var_name='Track', value_name='Count')

    # Plot using Plotly
    fig = px.bar(crosstab_df, x='gender', y='Count', color='Track', barmode='stack',
                 title="Gender Distribution Across Tracks")

    st.plotly_chart(fig)

    # Compute Chi-Square p-value
    contingency = pd.crosstab(df['gender'], df[target_column])
    chi2, p, dof, expected = chi2_contingency(contingency)

    # Show significance
    st.markdown(f"**Chi-Square Test P-Value:** `{p}`")
    if p < 0.05:
        st.success("Gender is statistically significant in predicting the track (p < 0.05).")
    else:
        st.warning("Gender is not statistically significant (p ≥ 0.05).")


# Show full score table
st.dataframe(scores_df.sort_values("Score", ascending=False))

