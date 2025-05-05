import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from utils.dataloader import load_data

# Load dataset
df = load_data()

# -----------------------------------------------
st.title("📥 Data Overview")

# -----------------------------------------------
# Missing Values
st.subheader("Missing Values Summary")

missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    st.write("✅ No missing values found.")
else:
    st.write(missing)

# -----------------------------------------------
# Data Statistics
st.subheader("Data Statistics")

data_desc = df.describe()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Add median
data_desc.loc['median'] = df[numeric_columns].median()

st.write(data_desc)

# -----------------------------------------------
# Track Distribution
st.subheader("Track Distribution")

track_counts = df['track'].value_counts()
track_percent = df['track'].value_counts(normalize=True)

track_distribution = pd.DataFrame({
    'Count': track_counts,
    'Percentage': (track_percent * 100).round(2).astype(str) + '%'
})

st.write(track_distribution)

# Pie chart
fig = px.pie(df, names='track', title='Track Distribution (Pie Chart)', hole=0.3)
st.plotly_chart(fig)

# -----------------------------------------------
# Average Grades per Track
st.subheader("Average Grades per Track")

grade_cols = [col for col in df.columns if 'g' in col and any(sub in col for sub in ['filipino', 'english', 'math', 'science', 'ap', 'tle', 'mapeh', 'esp'])]
track_avg_grades = df.groupby('track')[grade_cols].mean().round(2)

st.write(track_avg_grades)

# # -----------------------------------------------
# # Correlation Matrix
# st.subheader("Correlation Matrix")

# corr_matrix = df.corr(numeric_only=True)

# plt.figure(figsize=(12, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
# st.pyplot(plt)

# # Unstack and filter pairs with high correlation (excluding self-correlations)
# high_corr_pairs = (
#     corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     .stack()
#     .reset_index()
# )

# high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
# high_corr_pairs = high_corr_pairs[
#     (high_corr_pairs['Correlation'] >= 0.8) | (high_corr_pairs['Correlation'] <= -0.8)
# ]

# st.subheader("Highly Correlated Features (|r| ≥ 0.8)")
# st.write(high_corr_pairs)

# -----------------------------------------------
# Age Distribution - Overall
st.subheader("Age Distribution (Overall)")

# Create the histogram with Plotly
fig = px.histogram(df, x='age', nbins=15, title="Age Distribution (Overall)", 
                   labels={"age": "Age"}, opacity=0.7)
fig.update_traces(marker_color='skyblue')
st.plotly_chart(fig)

# # -----------------------------------------------
# # Age Distribution - Per Track
# st.subheader("Age Distribution (Per Track)")

# plt.figure(figsize=(10, 6))
# sns.boxplot(x='track', y='age', data=df, palette='Set2')
# plt.title('Age Distribution per Track')
# plt.xlabel('Track')
# plt.ylabel('Age')
# st.pyplot(plt)

# -----------------------------------------------
# Gender Distribution - Overall
st.subheader("Gender Distribution (Overall)")

# Gender counts
gender_counts = df['gender'].value_counts()

# Create bar chart using Plotly
fig = px.bar(gender_counts, x=gender_counts.index, y=gender_counts.values, 
             title="Gender Distribution (Overall)", labels={"x": "Gender", "y": "Count"})

# You can specify colors here using a list of colors if you want
fig.update_traces(marker=dict(color=['#ff7f0e', '#1f77b4']))  # Example colors: orange and blue

st.plotly_chart(fig)

# # -----------------------------------------------
# # Gender Distribution - Per Track
# st.subheader("Gender Distribution (Per Track)")

# plt.figure(figsize=(10, 6))
# gender_per_track = df.groupby(['track', 'gender']).size().unstack().fillna(0)
# gender_per_track.plot(kind='bar', stacked=False, color=['lightblue', 'lightcoral'], width=0.8)
# plt.title('Gender Distribution per Track')
# plt.xlabel('Track')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# st.pyplot(plt)

# # -----------------------------------------------
# # Demographic Breakdown
# st.subheader("Demographic Breakdown")

# df['age_group'] = pd.cut(df['age'], bins=[0, 13, 15, 17, 20], labels=['<14', '14-15', '16-17', '18+'])
# age_gender = df.groupby(['age_group', 'gender']).size().unstack(fill_value=0)

# st.write(age_gender)

# -----------------------------------------------
# Raw Data
st.subheader("Raw Data")
st.write(df)
