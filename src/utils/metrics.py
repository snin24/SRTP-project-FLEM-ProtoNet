import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_loss,
)


def to_numpy(array):
    if array is None:
        return None
    if "sparse" in str(type(array)):
        return array.toarray()
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def binarize_predictions(y_score, threshold=0.5):
    y_score = to_numpy(y_score)
    if np.min(y_score) < 0:
        return (y_score > 0).astype(np.int64)
    return (y_score > threshold).astype(np.int64)


def multilabel_metrics(y_true, y_score, y_pred=None, threshold=0.5):
    y_true = to_numpy(y_true)
    y_score = to_numpy(y_score)
    y_pred = binarize_predictions(y_score, threshold) if y_pred is None else to_numpy(y_pred)

    class_freq = np.sum(y_true, axis=0)
    valid_classes = class_freq != 0

    result = {
        "Hamming": hamming_loss(y_true, y_pred),
        "Ranking": label_ranking_loss(y_true, y_score),
        "Coverage": (coverage_error(y_true, y_score) - 1) / y_score.shape[1],
        "Micro-F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if np.any(valid_classes):
        result["mAP"] = average_precision_score(
            y_true[:, valid_classes],
            y_score[:, valid_classes],
            average="macro",
        )
    else:
        result["mAP"] = 0.0

    return result


def prediction_statistics(y_true, y_score, threshold=0.5):
    y_true = to_numpy(y_true)
    y_score = to_numpy(y_score)
    y_pred = binarize_predictions(y_score, threshold)

    return {
        "prob_min": float(np.min(y_score)),
        "prob_mean": float(np.mean(y_score)),
        "prob_max": float(np.max(y_score)),
        "pred_positive_rate": float(np.mean(y_pred)),
        "label_positive_rate": float(np.mean(y_true)),
    }


def sweep_thresholds(
    y_true,
    y_score,
    thresholds=None,
    target_metric="Micro-F1",
):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    best_threshold = None
    best_metrics = None
    best_value = -np.inf

    for threshold in thresholds:
        metrics = multilabel_metrics(y_true, y_score, threshold=float(threshold))
        value = metrics[target_metric]
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = metrics

    result = {
        f"Best-{target_metric}": best_value,
        "Best-threshold": best_threshold,
    }
    for name, value in best_metrics.items():
        result[f"Tuned-{name}"] = value
    return result


def evaluate_multilabel_predictions(
    y_true,
    y_score,
    threshold=0.5,
    threshold_search=False,
    thresholds=None,
):
    metrics = multilabel_metrics(y_true, y_score, threshold=threshold)
    metrics.update(prediction_statistics(y_true, y_score, threshold=threshold))

    if threshold_search:
        metrics.update(
            sweep_thresholds(
                y_true,
                y_score,
                thresholds=thresholds,
                target_metric="Micro-F1",
            )
        )

    return metrics


EPISODE_PROTOCOL = "episode_mean_v2_fixed_val"


def evaluate_episodic_predictions(labels, probabilities, threshold=0.5, threshold_search=False):
    """Equal-weight episode means; local columns must never be pooled as classes.

    AP excludes classes with no positive query in that episode, matching the
    existing AP convention. Report that coverage explicitly. Threshold search
    is for validation only and optimizes mean episode Micro-F1.
    """
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Expected matching nonempty episode lists")
    targets = [to_numpy(y) for y in labels]
    scores = [to_numpy(p) for p in probabilities]
    def compute(cutoff):
        rows = [multilabel_metrics(y, p, threshold=cutoff) for y, p in zip(targets, scores)]
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    metrics = compute(threshold)
    metrics.update(prediction_statistics(np.concatenate(targets), np.concatenate(scores), threshold))
    metrics["AP-valid-label-fraction"] = float(np.mean([np.mean(y.sum(axis=0) > 0) for y in targets]))
    if threshold_search:
        best_value, best_threshold = -1.0, None
        for cutoff in np.arange(0.05, 0.96, 0.05):
            f1_values = []
            for y, p in zip(targets, scores):
                predicted = p > cutoff
                denominator = predicted.sum() + y.sum()
                f1_values.append(float(2 * (predicted * y).sum() / denominator) if denominator else 0.0)
            value = float(np.mean(f1_values))
            if value > best_value:
                best_value, best_threshold = value, float(cutoff)
        metrics["Best-threshold"] = best_threshold
        metrics["Best-Micro-F1"] = best_value
        metrics.update({f"Tuned-{k}": v for k, v in compute(best_threshold).items()})
    return metrics
