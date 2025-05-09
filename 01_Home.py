import streamlit as st

st.set_page_config(
    page_title="SHS Track Insight"
)

st.title("🎓 SHS Track Insight")

st.caption("A predictive analytics web application designed to guide Junior High School students in selecting the most suitable Senior High School track.")

# --- Project Overview ---
st.markdown("### 🧾 Project Overview")
st.write("""
This system uses Logistic Regression to predict the most suitable Senior High School (SHS) track—Academic or TVL—for Junior High School students based on their academic performance and profile from Grades 7 to 10. By analyzing grade-level data, the model helps identify which track aligns best with a student's strengths, offering insights at each stage of their academic journey.
""")

# --- Flowchart ---
# st.markdown("### 🗺️ Process Flow")
# st.image("A_flowchart_in_a_digital_vector_graphic_illustrate.png", caption="Process Flow of the System", use_column_width=True)

# --- How to Use the App ---
st.markdown("### 📌 How to Use This App")
st.markdown("""
- 📥 **Data Overview** tab to explore and understand the structure, features, and distribution of the student dataset.
- 🧠 **Feature Selection** tab to identify the most relevant academic and demographic factors that influence Senior High School track predictions.
- 🤖 **Logistic Regression Model** page to analyze predictions based on the selected features and visualize the model’s decision outcomes.
- 🎯 **Evaluation Metrics** section to assess the model’s performance using accuracy, confusion matrix, classification report, and ROC curve.
-    **Balanced Logistic Regression Model**
-    **Evaluation Metrics** for Balanced Model
""")


# --- About Section ---
st.markdown("### 👨‍💻 About This Project")
st.write("""
This project was developed as part of a software engineering research initiative to help JHS students make informed SHS track decisions using predictive analytics based on academic performance.
""")

# --- GitHub Link (optional) ---
# st.markdown("[🔗 View on GitHub](https://github.com/your-username/your-repo)")
