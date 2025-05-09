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

st.markdown(f"**📊 Description of Summary Statistics:**")
st.markdown(f"""
* **Count**: Total number of valid (non-missing) values in the column.
* **Mean**: The average value of the column.
* **Standard Deviation (std)**: A measure of how much the values deviate from the mean.
* **Min**: The smallest value in the column.
* **25% (1st Quartile)**: 25% of the values fall below this point.
* **50% (Median)**: The middle value; 50% of the data lies below this point.
* **75% (3rd Quartile)**: 75% of the values fall below this point.
* **Max**: The largest value in the column.
* **Median**: Same as the 50%, represents the central value of the dataset.
""")

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
st.subheader("Average Grades per Track (Per Grade Level)")
st.write("The tables below show the average subject grades per track (Academic and TVL) for each grade level from Grade 7 to Grade 10. This breakdown provides a clearer view of student performance trends across subjects and grade levels, helping identify which subjects may influence track predictions the most.")
grade_cols = [col for col in df.columns if 'g' in col and any(sub in col for sub in ['filipino', 'english', 'math', 'science', 'ap', 'tle', 'mapeh', 'esp'])]

# Define grade levels and corresponding prefixes
grade_levels = {
    "Grade 7": "g7_",
    "Grade 8": "g8_",
    "Grade 9": "g9_",
    "Grade 10": "g10_"
}

for grade_name, prefix in grade_levels.items():
    st.markdown(f"**{grade_name}**")
    
    # Filter columns for the specific grade
    grade_subjects = [col for col in grade_cols if col.startswith(prefix)]
    
    # Compute average per track
    avg_table = df.groupby('track')[grade_subjects].mean().round(2)
    
    # Display the table
    st.write(avg_table)


# -----------------------------------------------
# Age Distribution
st.subheader("Age Distribution")
st.write("The age distribution charts provide insights into the student population's age range. The overall distribution shows how frequently each age appears across all students, while the grouped histogram compares age frequencies between the Academic and TVL tracks, highlighting any age-related trends or differences in track enrollment.")

# Group by track and get age counts
age_per_track = df.groupby('track')['age'].value_counts().unstack(fill_value=0)

# Calculate the total (overall) counts for each age
total_age_count = df['age'].value_counts().sort_index()

# Create a DataFrame for total counts and label it as 'Total'
total_age_count_df = pd.DataFrame([total_age_count])
total_age_count_df.index = ['Total']

# Combine Academic, TVL, and Total
age_per_track_with_total = pd.concat([age_per_track, total_age_count_df], axis=0)

# Rename the index to 'Track' so it appears as a column
age_per_track_with_total.index.name = 'Track'
age_per_track_with_total.reset_index(inplace=True)

# Display in Streamlit
st.write(age_per_track_with_total)

# Age Distribution - per Track
fig = px.histogram(
    df,
    x='age',
    color='track',
    nbins=15,
    opacity=0.6,
    barmode='stack',
    title="Age Distribution per Track (Grouped)",
    labels={"age": "Age", "track": "Track"},
)

fig.update_layout(bargap=0.1)
st.plotly_chart(fig)

# # -----------------------------------------------
# Gender Distribution - Overall
st.subheader("Gender Distribution (Overall)")
st.write("The gender distribution charts offer a view of the student population by gender. The overall chart shows the total number of male and female students in the dataset, while the grouped bar chart breaks this down by track, allowing for a comparison of gender representation between the Academic and TVL tracks.")

# Group by track and get gender counts
gender_per_track = df.groupby('track')['gender'].value_counts().unstack(fill_value=0)

# Calculate the total counts for gender
total_gender_counts = df['gender'].value_counts().sort_index()

# Create a DataFrame for total and label it as 'Total'
total_gender_df = pd.DataFrame([total_gender_counts])
total_gender_df.index = ['Total']

# Combine Academic, TVL, and Total
gender_per_track_with_total = pd.concat([gender_per_track, total_gender_df], axis=0)

# Rename index to 'Track' so it becomes the first column
gender_per_track_with_total.index.name = 'Track'
gender_per_track_with_total.reset_index(inplace=True)

# Display in Streamlit
st.write(gender_per_track_with_total)

# Prepare grouped data
gender_track_counts = df.groupby(['track', 'gender']).size().reset_index(name='count')

# Plot with Plotly
fig = px.bar(
    gender_track_counts,
    x='track',
    y='count',
    color='gender',
    barmode='group',
    title="Gender Distribution per Track",
    labels={"track": "Track", "count": "Count", "gender": "Gender"},
    color_discrete_sequence=['#ff7f0e', '#1f77b4']  # Blue for Male, Orange for Female
)

st.plotly_chart(fig)

# -----------------------------------------------
# Raw Data
with st.expander("📄 Show Raw Data"):
    st.write(df)

