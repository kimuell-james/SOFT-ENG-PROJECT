import pandas as pd
import streamlit as st

@st.cache_data
def load_csv(path: str):
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the CSV file: {e}")

