from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency
import pandas as pd

class FeatureSelector:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column
        self.scores_per_grade = {}
        self.significant_features = {}

    def chi_square_test(self, feature_col, target_col):
        if feature_col not in self.df.columns or target_col not in self.df.columns:
            raise KeyError(f"Columns '{feature_col}' or '{target_col}' not found in the dataframe.")
        
        # Compute the contingency table for chi-square
        contingency_table = pd.crosstab(self.df[feature_col], self.df[target_col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return p

    def select_features(self, grade_level):
        # Select the grade columns up to the selected grade level
        grade_columns = []
        for g in range(7, grade_level + 1):
            grade_columns.extend([col for col in self.df.columns if col.startswith(f"g{g}_")])
        
        # Add age and gender columns to the list of features
        additional_features = ["age", "gender"]
        features = grade_columns + [f for f in additional_features if f in self.df.columns]

        print("Available grade columns:", grade_columns)
        print("Available columns:", self.df.columns)

        # Prepare data for feature selection
        X = self.df[features]
        y = self.df[self.target_column]

        # Store the feature scores
        scores_df = pd.DataFrame(columns=["Feature", "F-Score", "P-Value"])

        for feature in X.columns:
            if X[feature].dtype in ["float64", "int64"]:  # Numerical feature
                # Scale numerical features before ANOVA
                X_scaled = X[feature].values.reshape(-1, 1)
                X_scaled = StandardScaler().fit_transform(X_scaled)
                
                # Perform ANOVA (f_classif) for numerical features
                f_values, p_values = f_classif(X_scaled, y)
                temp_df = pd.DataFrame({"Feature": [feature], "F-Score": [f_values[0]], "P-Value": [p_values[0]]})
                scores_df = pd.concat([scores_df, temp_df], ignore_index=True)
                
            else:  # Categorical feature (e.g., gender)
                # Perform Chi-Square test for categorical features
                print(f"Chi-Square test for: {feature} with target: {self.target_column}")
                p_value = self.chi_square_test(feature, self.target_column)
                temp_df = pd.DataFrame({"Feature": [feature], "F-Score": [None], "P-Value": [p_value]})
                scores_df = pd.concat([scores_df, temp_df], ignore_index=True)

        # Set the feature as the index
        scores_df.set_index("Feature", inplace=True)

        # Store all scores for this grade level
        self.scores_per_grade[grade_level] = scores_df

        # Identify significant features (p < 0.05)
        sig_features = scores_df[scores_df["P-Value"] < 0.05].index.tolist()
        self.significant_features[grade_level] = sig_features

        return sig_features
