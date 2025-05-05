import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from utils.dataloader import load_data

# Load dataset
df = load_data()

# Data Overview Page
st.title("Data Overview")

# Show statistics
st.subheader("Data Statistics")

# Get the description of the dataset
data_desc = df.describe()

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Compute the median for each numeric column
median_values = df[numeric_columns].median()

# Add the median values to the summary statistics
data_desc.loc['median'] = median_values

# Display the updated statistics table with the median
st.write(data_desc)

# -----------------------------------------------
# Age Distribution - Overall
st.subheader("Age Distribution (Overall)")

# Plot the age distribution as a histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], kde=True, color='skyblue', bins=15)
plt.title('Age Distribution (Overall)')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Display the histogram in Streamlit
st.pyplot(plt)

# -----------------------------------------------
# Age Distribution - Per Track
st.subheader("Age Distribution (Per Track)")

# Plot age distribution for each track using boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='track', y='age', data=df, palette='Set2')
plt.title('Age Distribution per Track')
plt.xlabel('Track')
plt.ylabel('Age')

# Display the boxplot in Streamlit
st.pyplot(plt)

# -----------------------------------------------
# Gender Distribution - Overall
st.subheader("Gender Distribution (Overall)")

# Plot the gender distribution as a bar chart
gender_counts = df['gender'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='viridis')
plt.title('Gender Distribution (Overall)')
plt.xlabel('Gender')
plt.ylabel('Count')

# Display the bar chart in Streamlit
st.pyplot(plt)

# -----------------------------------------------
# Gender Distribution - Per Track
st.subheader("Gender Distribution (Per Track)")

# Plot gender distribution for each track using a grouped bar chart
plt.figure(figsize=(10, 6))
gender_per_track = df.groupby(['track', 'gender']).size().unstack().fillna(0)
gender_per_track.plot(kind='bar', stacked=False, color=['lightblue', 'lightcoral'], width=0.8)

plt.title('Gender Distribution per Track')
plt.xlabel('Track')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Display the grouped bar chart in Streamlit
st.pyplot(plt)

# Raw Data
st.subheader("Raw Data")
st.write(df)
