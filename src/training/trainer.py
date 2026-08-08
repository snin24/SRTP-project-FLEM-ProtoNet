import torch

from .losses import flem_protonet_loss


def _compute_episode_loss(
    outputs,
    support_labels,
    query_labels,
    support_loss_weight,
    support_loss_type,
    flem_alpha,
    flem_beta,
    flem_method,
    flem_threshold,
    reduction,
):
    total_loss, losses = flem_protonet_loss(
        logits=outputs["logits"],
        query_labels=query_labels,
        support_label_logits=outputs["support_label_logits"],
        support_classifier_logits=outputs["support_classifier_logits"],
        support_features=outputs["support_features"],
        support_labels=support_labels,
        support_loss_weight=support_loss_weight,
        support_loss_type=support_loss_type,
        flem_alpha=flem_alpha,
        flem_beta=flem_beta,
        flem_method=flem_method,
        flem_threshold=flem_threshold,
        reduction=reduction,
    )
    return total_loss, losses


def train_episode(
    model,
    optimizer,
    support_images,
    support_labels,
    query_images,
    query_labels,
    support_loss_weight=0.0,
    support_loss_type="flem",
    flem_alpha=0.001,
    flem_beta=0.001,
    flem_method="ld",
    flem_threshold=0.0,
    reduction="mean",
):
    model.train()
    optimizer.zero_grad()

    outputs = model(
        support_images=support_images,
        support_labels=support_labels,
        query_images=query_images,
    )
    total_loss, losses = _compute_episode_loss(
        outputs=outputs,
        support_labels=support_labels,
        query_labels=query_labels,
        support_loss_weight=support_loss_weight,
        support_loss_type=support_loss_type,
        flem_alpha=flem_alpha,
        flem_beta=flem_beta,
        flem_method=flem_method,
        flem_threshold=flem_threshold,
        reduction=reduction,
    )
    total_loss.backward()
    optimizer.step()

    return outputs, losses


def evaluate_episode(
    model,
    support_images,
    support_labels,
    query_images,
    query_labels,
    support_loss_weight=0.0,
    support_loss_type="flem",
    flem_alpha=0.001,
    flem_beta=0.001,
    flem_method="ld",
    flem_threshold=0.0,
    reduction="mean",
):
    model.eval()
    with torch.no_grad():
        outputs = model(
            support_images=support_images,
            support_labels=support_labels,
            query_images=query_images,
        )
        _, losses = _compute_episode_loss(
            outputs=outputs,
            support_labels=support_labels,
            query_labels=query_labels,
            support_loss_weight=support_loss_weight,
            support_loss_type=support_loss_type,
            flem_alpha=flem_alpha,
            flem_beta=flem_beta,
            flem_method=flem_method,
            flem_threshold=flem_threshold,
            reduction=reduction,
        )

    return outputs, losses
