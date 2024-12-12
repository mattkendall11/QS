# adapted from https://github.com/matteoprata/LOBCAST

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
)

from . import constants as cst


def compute_metrics(truth, prediction):
    truth = torch.Tensor(truth)
    prediction = torch.Tensor(prediction)

    cr = classification_report(truth, prediction, output_dict=True, zero_division=0)
    accuracy = cr["accuracy"]  # MICRO-F1

    f1score = cr["macro avg"]["f1-score"]  # MACRO-F1
    precision = cr["macro avg"]["precision"]  # MACRO-PRECISION
    recall = cr["macro avg"]["recall"]  # MACRO-RECALL

    f1score_w = cr["weighted avg"]["f1-score"]  # WEIGHTED-F1
    precision_w = cr["weighted avg"]["precision"]  # WEIGHTED-PRECISION
    recall_w = cr["weighted avg"]["recall"]  # WEIGHTED-RECALL

    mcc = matthews_corrcoef(truth, prediction)
    cok = cohen_kappa_score(truth, prediction)

    mat_confusion = confusion_matrix(truth, prediction)

    val_dict = {
        cst.Metrics.F1.value: float(f1score),
        cst.Metrics.F1_W.value: float(f1score_w),
        cst.Metrics.PRECISION.value: float(precision),
        cst.Metrics.PRECISION_W.value: float(precision_w),
        cst.Metrics.RECALL.value: float(recall),
        cst.Metrics.RECALL_W.value: float(recall_w),
        cst.Metrics.ACCURACY.value: float(accuracy),
        cst.Metrics.MCC.value: float(mcc),
        cst.Metrics.COK.value: float(cok),
        cst.Metrics.CM.value: mat_confusion.tolist(),
    }
    return val_dict


def compute_score(truth, prediction) -> float:
    """Compute the percentage difference between the scores and the benchmark."""
    benchmark = np.array([88.7, 80.6, 80.1, 88.2, 91.6])
    # Compute scores for each horizon
    scores = [
        compute_metrics(truth[:, h], prediction[:, h])[cst.Metrics.F1.value]
        for h in range(5)
    ]
    # Calculate percentage differences between scores and the benchmark
    percentage_diffs = [
        100 * (scores[h] - benchmark[h]) / benchmark[h] for h in range(5)
    ]
    # Compute the average percentage difference
    avg_percentage_diff = np.mean(percentage_diffs)
    return avg_percentage_diff
