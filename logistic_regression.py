from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class LogisticModel:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def train_model(self):
        """Splits data, trains logistic regression, and returns model & test data."""
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)

        return model, X_test, y_test
