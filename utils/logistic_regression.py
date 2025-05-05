import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class LogisticModel:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.model = LogisticRegression(max_iter=1000)
        self.feature_importances = None
        self.track_distribution_data = None
        self.grade_stats = None
        self.label_encoder = LabelEncoder()

    def encode_categorical(self):
        if 'gender' in self.df.columns:
            self.df['gender'] = self.label_encoder.fit_transform(self.df['gender'])
        return self.df

    def train_model(self, grade, features):
        self.df = self.encode_categorical()

        included_grades = list(range(7, grade + 1))
        grade_columns = [col for col in self.df.columns if any(f"g{g}_" in col for g in included_grades)]
        grade_columns.extend(['age', 'gender'])

        filtered_df = self.df.dropna(subset=grade_columns)

        X = filtered_df[features]
        y = filtered_df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)

        self.total_predictions = len(self.y_pred)
        self.prediction_counts = dict(pd.Series(self.y_pred).value_counts())
        self.track_distribution_data = pd.Series(self.y_pred).value_counts(normalize=True) * 100

        self.df_with_predictions = filtered_df.copy()
        self.df_with_predictions['predicted_track'] = self.model.predict(filtered_df[features])

    def calculate_grade_statistics(self, features):
        df_with_track = self.X_test.copy()
        df_with_track[self.target_column] = self.y_pred

        # Calculate stats per predicted track
        stats = df_with_track.groupby(self.target_column).agg(
            {feature: ['min', 'max', 'mean', 'median', 'count'] for feature in features}
        )

        # Build result DataFrame
        subjects = []
        for subject in features:
            academic_min = stats[subject]['min'].get('Academic', np.nan)
            academic_max = stats[subject]['max'].get('Academic', np.nan)
            academic_mean = stats[subject]['mean'].get('Academic', np.nan)
            academic_median = stats[subject]['median'].get('Academic', np.nan)
            academic_count = stats[subject]['count'].get('Academic', 0)

            tvl_min = stats[subject]['min'].get('TVL', np.nan)
            tvl_max = stats[subject]['max'].get('TVL', np.nan)
            tvl_mean = stats[subject]['mean'].get('TVL', np.nan)
            tvl_median = stats[subject]['median'].get('TVL', np.nan)
            tvl_count = stats[subject]['count'].get('TVL', 0)

            subjects.append([
                academic_min, academic_max, academic_mean, academic_median, academic_count,
                tvl_min, tvl_max, tvl_mean, tvl_median, tvl_count
            ])

        result_df = pd.DataFrame(
            subjects,
            columns=[
                'Academic (Min)', 'Academic (Max)', 'Academic (Mean)', 'Academic (Median)', 'Academic (Count)',
                'TVL (Min)', 'TVL (Max)', 'TVL (Mean)', 'TVL (Median)', 'TVL (Count)'
            ],
            index=features
        )

        return result_df

    def get_track_distribution(self):
        return self.track_distribution_data

    def grade_statistics_per_track(self):
        return self.grade_stats
    
    def get_gender_distribution_by_track(self):
        df_with_track = self.X_test.copy()
        df_with_track[self.target_column] = self.y_pred

        # Map encoded gender back to labels if needed
        if hasattr(self.label_encoder, 'inverse_transform'):
            df_with_track['gender'] = self.label_encoder.inverse_transform(df_with_track['gender'])

        # Count gender per predicted track
        gender_counts = df_with_track.groupby([self.target_column, 'gender']).size().unstack(fill_value=0)

        return gender_counts
