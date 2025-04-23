import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

class ModelEvaluation:
    def __init__(self, predictions):
        """
        Initialize with the predictions for each grade.
        predictions: dictionary of predictions per grade.
        Example: {
            'G7': {'y_test': y_test_g7, 'y_pred': y_pred_g7, 'y_prob': y_prob_g7},
            'G8': {'y_test': y_test_g8, 'y_pred': y_pred_g8, 'y_prob': y_prob_g8},
            ...
        }
        """
        self.predictions = predictions

    def evaluate(self, grade):
        """
        Evaluate the model using multiple classification metrics for a specific grade.
        grade: grade level for which metrics will be evaluated (e.g., 'G7', 'G8', etc.)
        """
        y_test = np.array(self.predictions[grade]['y_test'])
        y_pred = np.array(self.predictions[grade]['y_pred'])
        
        # Convert labels: 'Academic' → 0, 'TVL' → 1
        label_map = {'Academic': 0, 'TVL': 1}
        y_test_numeric = np.array([label_map[label] for label in y_test])
        y_pred_numeric = np.array([label_map[label] for label in y_pred])

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

    def show_confusion_matrix(self, grade):
        """
        Returns confusion matrix for a specific grade.
        grade: grade level for which confusion matrix is computed.
        """
        y_test = np.array(self.predictions[grade]['y_test'])
        y_pred = np.array(self.predictions[grade]['y_pred'])
        return confusion_matrix(y_test, y_pred)

    def classification_report(self, grade):
        """
        Returns classification report for a specific grade.
        grade: grade level for which classification report is generated.
        """
        y_test = np.array(self.predictions[grade]['y_test'])
        y_pred = np.array(self.predictions[grade]['y_pred'])
        return classification_report(y_test, y_pred)

    def roc_auc(self, grade):
        """
        Returns the ROC AUC score for a specific grade.
        grade: grade level for which ROC AUC score is computed.
        """
        y_test = np.array(self.predictions[grade]['y_test'])
        y_prob = np.array(self.predictions[grade]['y_prob'])
        return roc_auc_score(pd.get_dummies(y_test), y_prob)

    def roc_curve_plot(self, grade):
        """
        Plots ROC curve for a specific grade.
        grade: grade level for which ROC curve is plotted.
        """
        y_test = np.array(self.predictions[grade]['y_test'])
        y_prob = np.array(self.predictions[grade]['y_prob'])

        # Binarize the labels
        y_test_bin = pd.get_dummies(y_test)
        
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {grade}')
        plt.legend()
        plt.show()
