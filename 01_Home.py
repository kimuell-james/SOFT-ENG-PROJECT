import streamlit as st
from dataloader import load_csv

file_path = "./data/synthetic_student_data.csv"

df = load_csv(file_path)

st.set_page_config(
    page_title="Predictive Analytics"
)

st.title("Homepage")

st.header("Research Overview")
st.write("""In this research, the goal is to develop a predictive analytics model that will 
be implemented using the Logistic Regression algorithm to predict the most 
suitable Senior High School track for Junior High School students based on their 
academic performance and other relevant factors. Logistic regression is a data 
analysis technique that calculates the likelihood of an event occurring given a set 
of independent factors. It employs mathematics to discover links between two data 
factors, then uses these relationships to forecast the value of one factor based on 
the other. A logistic regression is best to use when predicting dichotomous 
categorical outcomes, and in this research, the categories are Academic 
Track and TVL Track. The data to be evaluated are the students' profiles and 
academic records from Grade 7 to Grade 10. Then, the data collected will be used 
to train the model and test its performance. This allows for the tracking of changes 
in forecast at each grade level. In this manner, students will know which track is 
best for them based on their academic achievement. """)