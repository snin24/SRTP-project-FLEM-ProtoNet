import torch
import torch.nn.functional as F
from torch import nn


def _validate_binary_targets(logits, targets, logits_name, targets_name):
    if logits.shape != targets.shape:
        raise ValueError(f"{logits_name} and {targets_name} must have the same shape.")
    if not torch.is_floating_point(targets):
        targets = targets.to(dtype=logits.dtype)
    return targets


def multilabel_prototype_loss(logits, query_labels, reduction="mean"):
    query_labels = _validate_binary_targets(
        logits,
        query_labels,
        "logits",
        "query_labels",
    )
    return F.binary_cross_entropy_with_logits(
        logits,
        query_labels,
        reduction=reduction,
    )


def soft_cross_entropy(logits, soft_targets, reduction="mean"):
    """Softmax-based soft cross entropy. 假设标签互斥，适用于单标签场景。"""
    if logits.shape != soft_targets.shape:
        raise ValueError("logits and soft_targets must have the same shape.")

    loss = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of: mean, sum, none.")


def sigmoid_soft_cross_entropy(logits, soft_targets, reduction="mean"):
    """Sigmoid-based soft cross entropy. 逐标签独立计算，适用于多标签场景。

    loss = -[t * log(sigmoid(x)) + (1-t) * log(1-sigmoid(x))]
    """
    if logits.shape != soft_targets.shape:
        raise ValueError("logits and soft_targets must have the same shape.")

    loss = F.binary_cross_entropy_with_logits(
        logits,
        soft_targets,
        reduction=reduction,
    )
    return loss


def pairwise_cosine_similarity(features):
    normalized = F.normalize(features, dim=1)
    return normalized @ normalized.transpose(0, 1)


def flem_loss_enhanced(
    pred,
    teach,
    y,
    gamma_pos=0.0,
    gamma_neg=1.0,
    eps=1e-7,
):
    """原论文 loss_enhanced(pred, teach, y):
    pred  = LE logits (student)
    teach = classifier logits (teacher)
    y     = multi-hot labels
    """
    y = _validate_binary_targets(
        pred,
        y,
        "pred (LE)",
        "y (labels)",
    )
    if teach.shape != pred.shape:
        raise ValueError("teach (classifier) and pred (LE) must have the same shape.")

    pred_probs = torch.sigmoid(pred)
    positive_loss = y * torch.log(pred_probs.clamp(min=eps, max=1 - eps))
    negative_loss = (1 - y) * torch.log(
        (1 - pred_probs).clamp(min=eps, max=1 - eps)
    )
    loss = positive_loss + negative_loss

    with torch.no_grad():
        teach_probs = torch.sigmoid(teach)
        pt = teach_probs * y + (1 - teach_probs) * (1 - y)
        one_sided_gamma = gamma_pos * y + gamma_neg * (1 - y)
        one_sided_weight = torch.pow(1 - pt, one_sided_gamma)

    return -(loss * one_sided_weight).mean()


def flem_label_specific_loss(
    le,
    pred,
    y,
    x=None,
    method="ld",
    threshold=0.0,
    reduction="mean",
):
    """原论文 label-specific loss: le (LE logits), pred (classifier logits), y (labels), x (features)."""
    y = _validate_binary_targets(
        le,
        y,
        "le (LE logits)",
        "y (labels)",
    )
    if pred.shape != le.shape:
        raise ValueError("pred (classifier) and le (LE) must have the same shape.")

    if method == "ld":
        pred_probs = torch.sigmoid(pred.detach())
        return nn.BCEWithLogitsLoss(reduction=reduction)(
            le,
            pred_probs,
        )

    if method == "threshold":
        sample_losses = []
        for sample_idx in range(pred.shape[0]):
            neg_index = y[sample_idx] == 0
            pos_index = y[sample_idx] == 1
            if torch.sum(pos_index) == 0 or torch.sum(neg_index) == 0:
                continue
            hardest_negative = le[sample_idx][neg_index].max()
            easiest_positive = le[sample_idx][pos_index].min()
            sample_losses.append(
                torch.clamp(hardest_negative - easiest_positive + threshold, min=0.0)
            )
        if not sample_losses:
            return le.new_tensor(0.0)
        losses = torch.stack(sample_losses)
        if reduction == "sum":
            return losses.sum()
        if reduction in {"mean", "none"}:
            return losses.mean() if reduction == "mean" else losses
        raise ValueError("reduction must be one of: mean, sum, none.")

    if method == "sim":
        if x is None:
            raise ValueError("x (features) is required when method='sim'.")
        with torch.no_grad():
            feature_similarity = pairwise_cosine_similarity(x).detach()
        label_similarity = pairwise_cosine_similarity(le)
        return F.mse_loss(label_similarity, feature_similarity, reduction=reduction)

    raise ValueError("method must be one of: ld, threshold, sim.")


def flem_original_style_loss(
    pred,
    le,
    y,
    x=None,
    alpha=0.001,
    beta=0.001,
    method="ld",
    threshold=0.0,
    reduction="mean",
):
    y = _validate_binary_targets(
        pred,
        y,
        "pred (classifier)",
        "y (labels)",
    )
    if le is None:
        raise ValueError("le (LE logits) is required for FLEM loss.")
    if pred.shape != le.shape:
        raise ValueError("pred (classifier) and le (LE) must have the same shape.")

    loss_pred_cls = nn.BCEWithLogitsLoss(reduction=reduction)(
        pred,
        y,
    )
    loss_pred_ld = nn.BCEWithLogitsLoss(reduction=reduction)(
        pred,
        torch.sigmoid(le.detach()),
    )
    loss_le_cls = flem_loss_enhanced(
        pred=le,
        teach=pred,
        y=y,
    )
    loss_le_spec = flem_label_specific_loss(
        le=le,
        pred=pred,
        y=y,
        x=x,
        method=method,
        threshold=threshold,
        reduction=reduction,
    )

    loss_pred = alpha * loss_pred_ld + (1 - alpha) * loss_pred_cls
    loss_le = beta * loss_le_spec + (1 - beta) * loss_le_cls
    total = loss_pred + loss_le

    return total, {
        "flem_total": total,
        "flem_prediction": loss_pred,
        "flem_prediction_cls": loss_pred_cls,
        "flem_prediction_ld": loss_pred_ld,
        "flem_label_enhancement": loss_le,
        "flem_label_cls": loss_le_cls,
        "flem_label_specific": loss_le_spec,
    }


def flem_protonet_loss(
    logits,
    query_labels,
    support_label_logits=None,
    support_classifier_logits=None,
    support_features=None,
    support_labels=None,
    support_loss_weight=0.0,
    support_loss_type="flem",
    flem_alpha=0.001,
    flem_beta=0.001,
    flem_method="ld",
    flem_threshold=0.0,
    reduction="mean",
):
    prediction_loss = multilabel_prototype_loss(
        logits,
        query_labels,
        reduction=reduction,
    )
    total_loss = prediction_loss
    losses = {
        "total": total_loss,
        "prediction": prediction_loss,
    }

    if support_loss_weight > 0.0:
        if support_label_logits is None or support_labels is None:
            raise ValueError(
                "support_label_logits and support_labels are required when "
                "support_loss_weight > 0."
            )

        if support_loss_type == "flem":
            if support_classifier_logits is None:
                raise ValueError(
                    "support_classifier_logits is required when "
                    "support_loss_type='flem'."
                )
            support_loss, support_parts = flem_original_style_loss(
                pred=support_classifier_logits,
                le=support_label_logits,
                y=support_labels,
                x=support_features,
                alpha=flem_alpha,
                beta=flem_beta,
                method=flem_method,
                threshold=flem_threshold,
                reduction=reduction,
            )
            losses.update(support_parts)
        elif support_loss_type == "bce":
            support_labels = _validate_binary_targets(
                support_label_logits,
                support_labels,
                "support_label_logits",
                "support_labels",
            )
            support_loss = F.binary_cross_entropy_with_logits(
                support_label_logits,
                support_labels,
                reduction=reduction,
            )
            losses["support_label_enhancement"] = support_loss
        else:
            raise ValueError("support_loss_type must be either 'flem' or 'bce'.")

        total_loss = prediction_loss + support_loss_weight * support_loss
        losses["support"] = support_loss
        losses["total"] = total_loss

    return total_loss, losses
