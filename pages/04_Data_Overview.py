import streamlit as st
from dataloader import load_csv

st.set_page_config(
    page_title="Data Overview"
)

st.title("Data Overview")

file_path = "./data/synthetic_student_data.csv"

df = load_csv(file_path)

st.dataframe(df)