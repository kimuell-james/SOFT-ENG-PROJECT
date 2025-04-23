from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class LogisticRegressionModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.predictions = {}

    def train(self, df, selected_features, target_column, grade):
        X = df[selected_features]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        max_iter = max(1000, df.shape[0] // 2)
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=max_iter)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        self.models[grade] = model
        self.scalers[grade] = scaler
        self.predictions[grade] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        return model, y_test, y_pred, y_prob
