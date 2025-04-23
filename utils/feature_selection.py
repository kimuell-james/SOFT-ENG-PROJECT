# Let's define the corrected FeatureSelector class to include scores_per_grade and significant_features.

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd

class FeatureSelector:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.scores_per_grade = {}
        self.significant_features = {}

    def select_features(self, grade_level):
        grade_prefix = f"g{grade_level}_"
        grade_columns = [col for col in self.df.columns if col.startswith(grade_prefix)]
        additional_features = ["age", "gender"]
        features = grade_columns + [f for f in additional_features if f in self.df.columns]

        X = self.df[features]
        y = self.df[self.target_column]

        # Ensure gender is treated as categorical (if not already encoded)
        if "gender" in X.columns and X["gender"].dtype == object:
            X["gender"] = X["gender"].astype("category").cat.codes

        # Scale numerical features (grades and age) for ANOVA
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
        X_scaled = X.copy()
        X_scaled[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

        # Compute ANOVA F-scores and p-values
        f_values, p_values = f_classif(X_scaled, y)

        # Store results in DataFrame
        scores_df = pd.DataFrame({
            "Feature": X.columns,
            "F Score": f_values,
            "P-Value": p_values
        }).set_index("Feature")

        # Store all scores
        self.scores_per_grade[grade_level] = scores_df

        # Identify statistically significant features (p < 0.05)
        sig_features = scores_df[scores_df["P-Value"] < 0.05].index.tolist()
        self.significant_features[grade_level] = sig_features

        return sig_features

