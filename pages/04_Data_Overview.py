import streamlit as st
from dataloader import DataLoader

st.set_page_config(
    page_title="Data Overview"
)

st.title("Data Overview")

file_path = "./data/synthetic_student_data.csv"

loader = DataLoader(file_path)
df = loader.load_data()

st.dataframe(df)