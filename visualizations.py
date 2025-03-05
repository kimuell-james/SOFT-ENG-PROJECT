import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Visualization functions
def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df['age'], bins=10, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_gender_distribution(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=df['gender'], palette="pastel", ax=ax)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_track_distribution(df):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=df['track'], palette="Set2", ax=ax)
    ax.set_xlabel("Track")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_grade_distribution_g7(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df[['g7_filipino', 'g7_english', 'g7_math', 'g7_science', 'g7_ap', 'g7_tle', 'g7_mapeh', 'g7_esp']], palette="coolwarm", ax=ax)
    ax.set_xlabel("Subjects")
    ax.set_ylabel("Grades")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_grade_distribution_g8(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df[['g8_filipino', 'g8_english', 'g8_math', 'g8_science', 'g8_ap', 'g8_tle', 'g8_mapeh', 'g8_esp']], palette="coolwarm", ax=ax)
    ax.set_xlabel("Subjects")
    ax.set_ylabel("Grades")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_grade_distribution_g9(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df[['g9_filipino', 'g9_english', 'g9_math', 'g9_science', 'g9_ap', 'g7_tle', 'g9_mapeh', 'g9_esp']], palette="coolwarm", ax=ax)
    ax.set_xlabel("Subjects")
    ax.set_ylabel("Grades")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_grade_distribution_g10(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(data=df[['g10_filipino', 'g10_english', 'g10_math', 'g10_science', 'g10_ap', 'g10_tle', 'g10_mapeh', 'g10_esp']], palette="coolwarm", ax=ax)
    ax.set_xlabel("Subjects")
    ax.set_ylabel("Grades")
    plt.xticks(rotation=45)
    st.pyplot(fig)
