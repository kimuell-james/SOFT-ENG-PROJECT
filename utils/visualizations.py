import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

class Visualizations:
    def __init__(self, df):
        self.df = df

    def plot_feature_importance(self, feature_importance):
        """Visualizes feature importance."""
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=list(feature_importance.keys()), y=list(feature_importance.values()), ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    def plot_track_prediction_trends(self):
        """Visualizes track prediction trends from G7 to G10."""
        st.subheader("📈 Track Prediction Trends from G7 to G10")

        track_counts = {"G7": 0, "G8": 0, "G9": 0, "G10": 0}

        for grade in ["G7", "G8", "G9", "G10"]:
            relevant_cols = [col for col in self.df.columns if col.startswith(grade.lower())]
            avg_grade = self.df[relevant_cols].mean(axis=1)
            track_counts[grade] = (avg_grade > 85).sum()

        trend_df = pd.DataFrame.from_dict(track_counts, orient='index', columns=['Academic Track'])
        trend_df["TVL Track"] = len(self.df) - trend_df["Academic Track"]

        fig, ax = plt.subplots(figsize=(10, 5))
        trend_df.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
        plt.xlabel("Grade Level")
        plt.ylabel("Number of Students")
        plt.title("Track Prediction Trends from G7 to G10")
        plt.legend(["Academic Track", "TVL Track"])
        st.pyplot(fig)

    @staticmethod
    def plot_age_distribution():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(self.df['age'], bins=10, kde=True, color='skyblue', ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    @staticmethod
    def plot_gender_distribution():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=self.df['gender'], palette="pastel", ax=ax)
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    @staticmethod
    def plot_track_distribution():
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=self.df['track'], palette="Set2", ax=ax)
        ax.set_xlabel("Track")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    @staticmethod
    def plot_grade_distribution_g7():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.df[['g7_filipino', 'g7_english', 'g7_math', 'g7_science', 'g7_ap', 'g7_tle', 'g7_mapeh', 'g7_esp']], palette="coolwarm", ax=ax)
        ax.set_xlabel("Subjects")
        ax.set_ylabel("Grades")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    @staticmethod
    def plot_grade_distribution_g8():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.df[['g8_filipino', 'g8_english', 'g8_math', 'g8_science', 'g8_ap', 'g8_tle', 'g8_mapeh', 'g8_esp']], palette="coolwarm", ax=ax)
        ax.set_xlabel("Subjects")
        ax.set_ylabel("Grades")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    @staticmethod
    def plot_grade_distribution_g9():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.df[['g9_filipino', 'g9_english', 'g9_math', 'g9_science', 'g9_ap', 'g7_tle', 'g9_mapeh', 'g9_esp']], palette="coolwarm", ax=ax)
        ax.set_xlabel("Subjects")
        ax.set_ylabel("Grades")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    @staticmethod
    def plot_grade_distribution_g10():
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.df[['g10_filipino', 'g10_english', 'g10_math', 'g10_science', 'g10_ap', 'g10_tle', 'g10_mapeh', 'g10_esp']], palette="coolwarm", ax=ax)
        ax.set_xlabel("Subjects")
        ax.set_ylabel("Grades")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    @staticmethod
    def track_distribution(df, target_column):
        sns.countplot(x=target_column, data=df)
        plt.title("Track Rate Distribution")
        st.pyplot(plt)

    @staticmethod
    def correlation_matrix(df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
        plt.title("Correlation Matrix")
        st.pyplot(plt)

    @staticmethod
    def roc_curve_plot(y_test, y_prob):
        y_test_bin = label_binarize(y_test, classes=list(set(y_test)))
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        st.pyplot(plt)

    @staticmethod
    def feature_importance_plot(model, features):
        coef = pd.Series(model.coef_[0], index=features)
        coef = coef.sort_values()
        coef.plot(kind='barh', title="Feature Importance (Logistic Regression Coefficients)")
        plt.xlabel("Coefficient Value")
        plt.tight_layout()
        st.pyplot(plt)
