import streamlit as st
from dataloader import DataLoader
from visualizations import plot_age_distribution, plot_gender_distribution, plot_track_distribution, plot_grade_distribution_g7, plot_grade_distribution_g8, plot_grade_distribution_g9, plot_grade_distribution_g10

st.set_page_config(
    page_title="Student Insights"
)

st.title("Student Insights")

st.header("Visualized Data")

file_path = "./data/synthetic_student_data.csv"

data_loader = DataLoader(file_path)
df = data_loader.load_data()

st.subheader("Age Distribution")
plot_age_distribution(df)
st.subheader("Gender Distribution")
plot_gender_distribution(df)
st.subheader("Track Distribution")
plot_track_distribution(df)
st.subheader("Grade Distribution (G7)")
plot_grade_distribution_g7(df)
st.subheader("Grade Distribution (G8)")
plot_grade_distribution_g8(df)
st.subheader("Grade Distribution (G9)")
plot_grade_distribution_g9(df)
st.subheader("Grade Distribution (G10)")
plot_grade_distribution_g10(df)