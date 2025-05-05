import streamlit as st

st.set_page_config(
    page_title="Predictive Analytics"
)

st.title("🎓 Predictive Analytics for SHS Track Selection")

st.caption("A Predictive Analytics Tool for Junior High Students to their Senior High School Track")

# --- Project Overview ---
st.markdown("### 🧾 Project Overview")
st.write("""
This system uses Logistic Regression to predict the most suitable Senior High School (SHS) track—Academic or TVL—for Junior High School students based on their academic performance and profile from Grades 7 to 10. By analyzing grade-level data, the model helps identify which track aligns best with a student's strengths, offering insights at each stage of their academic journey.
""")

# --- How It Works ---
st.markdown("### ⚙️ How It Works")
st.markdown("""
1. 📥 **Data Collection** – Gathers student profiles and academic records from Grades 7 to 10.  
2. 🧹 **Preprocessing** – Cleans and prepares the data for analysis.  
3. 🧠 **Feature Selection** – Identifies the most relevant subjects and profile factors influencing track selection.  
4. 🤖 **Model Training** – Uses logistic regression to train a predictive model on the selected features.  
5. 🎯 **Evaluation** – Predicts the SHS track (Academic or TVL) and evaluates model accuracy across grade levels.
""")

# --- Flowchart ---
# st.markdown("### 🗺️ Process Flow")
# st.image("A_flowchart_in_a_digital_vector_graphic_illustrate.png", caption="Process Flow of the System", use_column_width=True)

# --- How to Use the App ---
st.markdown("### 📌 How to Use This App")
st.markdown("""
- Go to the **Data Overview** tab to explore the student dataset.
- Visit **Evaluation Metrics** to review model performance.
- See **Student Insights** to view individual predictions and breakdowns.
""")

# --- About Section ---
st.markdown("### 👨‍💻 About This Project")
st.write("""
This project was developed as part of a software engineering research initiative to help students make informed SHS track decisions using predictive analytics based on academic performance.
""")

# --- GitHub Link (optional) ---
# st.markdown("[🔗 View on GitHub](https://github.com/your-username/your-repo)")
