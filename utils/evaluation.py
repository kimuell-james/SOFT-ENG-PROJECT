import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

def evaluate_model(y_test, y_pred, y_proba, model):
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)
    labels = model.model.classes_  # ['Academic', 'TVL']
    cm_df = pd.DataFrame(cm, index=[f"Actual: {label}" for label in labels],
                            columns=[f"Predicted: {label}" for label in labels])

    # ROC/AUC
    unique_classes = np.unique(y_test)
    if len(unique_classes) == 2:
        pos_class_index = list(model.model.classes_).index(unique_classes[1])
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, pos_class_index], pos_label=unique_classes[1])
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None

    return {
        'accuracy': accuracy,
        'classification_report': pd.DataFrame(class_report).transpose(),
        'confusion_matrix_df': cm_df,
        'roc_curve': (fpr, tpr),
        'roc_auc': roc_auc
    }
