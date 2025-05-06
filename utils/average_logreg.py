import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class AverageLogisticModel:
    def __init__(self, df, target_column):
        self.df = df.copy()  # Safer: avoid modifying the original DataFrame
        self.target_column = target_column
        self.model = LogisticRegression(max_iter=1000)
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.predicted_tracks = {}

    def encode_categorical(self):
        if 'gender' in self.df.columns:
            self.df['gender'] = self.label_encoder.fit_transform(self.df['gender'])
        return self.df

    def calculate_average_grades(self, grade_level):
        included_grades = list(range(7, grade_level + 1))
        subjects = ['filipino', 'english', 'math', 'science', 'ap', 'tle', 'mapeh', 'esp']
        avg_columns = []

        for subject in subjects:
            subject_cols = [f"g{g}_{subject}" for g in included_grades if f"g{g}_{subject}" in self.df.columns]
            if subject_cols:
                avg_col_name = f"avg_{subject}_g{grade_level}"
                self.df[avg_col_name] = self.df[subject_cols].mean(axis=1)
                avg_columns.append(avg_col_name)

        return avg_columns

    def train_model(self, grade_level, return_eval=False):
        # Recalculate average grade features
        features = self.calculate_average_grades(grade_level)
        
        # Drop any rows with missing values (optional based on your dataset)
        data = self.df[features + [self.target_column]].dropna()

        X = data[features]
        y = data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        # Fit logistic regression
        self.model.fit(X_train, y_train)

        # Predict and store results
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Determine most predicted track
        track_prediction = pd.Series(y_pred).value_counts().idxmax()
        self.predicted_tracks[grade_level] = track_prediction

        if return_eval:
            return features, track_prediction, y_test, y_pred, y_proba
        else:
            return features, track_prediction

    def get_track_by_grade_level(self):
        return self.predicted_tracks
