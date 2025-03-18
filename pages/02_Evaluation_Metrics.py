import streamlit as st
from dataloader import DataLoader
from feature_selection import FeatureSelection
from logistic_regression import LogisticModel
from evaluation import ModelEvaluation
from visualizations import Visualization

st.set_page_config(page_title="Evaluation Metrics")

st.title("Evaluation Metrics")
st.header("Results")

# Load data
file_path = "./data/synthetic_student_data.csv"
data_loader = DataLoader(file_path)
df = data_loader.load_data()
st.write("✅ Data Loaded Successfully!")

# **Feature Selection**
feature_selector = FeatureSelection(df, "track")
selected_features = feature_selector.select_features()
df_selected = df[selected_features + ["track"]]
st.write(f"🔹 Selected Features: {selected_features}")

# **Train Model**
if st.sidebar.button("Train Model"):
    lm = LogisticModel(df_selected, "track")
    model, X_test, y_test = lm.train_model()
    st.sidebar.success("✅ Model Trained Successfully!")

    # **Generate Predictions**
    y_pred = model.predict(X_test)

    # **Evaluate Model**
    evaluator = ModelEvaluation(y_test, y_pred)  # ✅ Pass only y_test and y_pred
    metrics = evaluator.evaluate()

    st.subheader("📊 Model Evaluation Metrics")
    for metric, value in metrics.items():
        st.write(f"🔹 {metric}: {value:.4f}")

    # **Show Confusion Matrix**
    st.subheader("📉 Confusion Matrix")
    st.write(evaluator.show_confusion_matrix())

    # **Track Prediction Trends**
    vis = Visualization(df_selected)
    vis.plot_track_prediction_trends(df_selected)
