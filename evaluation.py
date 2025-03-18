from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

class ModelEvaluation:
    def __init__(self, y_test, y_pred):
        self.y_test = np.array(y_test)
        self.y_pred = np.array(y_pred)

    def evaluate(self):
        """Evaluate the model using multiple classification metrics."""
        
        # Convert labels: 'Academic' → 0, 'TVL' → 1
        label_map = {'Academic': 0, 'TVL': 1}
        y_test_numeric = np.array([label_map[label] for label in self.y_test])
        y_pred_numeric = np.array([label_map[label] for label in self.y_pred])

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test_numeric, y_pred_numeric)
        precision = precision_score(y_test_numeric, y_pred_numeric, average="binary", pos_label=1)
        recall = recall_score(y_test_numeric, y_pred_numeric, average="binary", pos_label=1)
        auc = roc_auc_score(y_test_numeric, y_pred_numeric)
        error_rate = 1 - accuracy

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "AUC": auc,
            "Error Rate": error_rate
        }

    def show_confusion_matrix(self):
        """Returns confusion matrix."""
        return confusion_matrix(self.y_test, self.y_pred)
