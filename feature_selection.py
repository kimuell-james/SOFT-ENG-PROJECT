import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureSelection:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def encode_categorical_features(self, categorical_features):
        """Convert categorical features to numerical using Label Encoding."""
        encoder = LabelEncoder()
        df_encoded = self.df.copy()
        for col in categorical_features:
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
        return df_encoded

    def chi_square_test(self):
        """Apply Chi-Square test on categorical features"""
        categorical_features = ['gender']  # Only gender is categorical in your dataset
        df_encoded = self.encode_categorical_features(categorical_features)

        chi_scores, _ = chi2(df_encoded[categorical_features], df_encoded[self.target])
        return dict(zip(categorical_features, chi_scores))

    def numerical_feature_importance(self):
        """Use SelectKBest for numerical feature selection (grades, age)"""
        numerical_features = self.df.select_dtypes(include=['number']).columns.tolist()

        # Ensure the target column is not mistakenly included
        if self.target in numerical_features:
            numerical_features.remove(self.target)

        if not numerical_features:  # Check if list is empty
            raise ValueError("No numerical features available for selection.")

        X = self.df[numerical_features]
        y = self.df[self.target]

        # Standardize data
        X_scaled = StandardScaler().fit_transform(X)

        # Apply SelectKBest with ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k="all")  # Use all features for analysis
        selector.fit(X_scaled, y)

        scores = dict(zip(numerical_features, selector.scores_))
        return scores

    def select_features(self):
        """Combine categorical and numerical feature selection results"""
        chi_results = self.chi_square_test()
        numerical_results = self.numerical_feature_importance()

        # Merge both results and sort by importance
        all_scores = {**chi_results, **numerical_results}
        selected_features = sorted(all_scores, key=all_scores.get, reverse=True)[:10]  # Keep top 10

        return selected_features
