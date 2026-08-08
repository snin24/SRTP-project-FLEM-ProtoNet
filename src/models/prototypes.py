import torch


def build_label_prototypes(support_features, support_label_weights, eps=1e-8):
    if support_features.dim() != 2:
        raise ValueError("support_features must have shape [num_support, feature_dim].")
    if support_label_weights.dim() != 2:
        raise ValueError(
            "support_label_weights must have shape [num_support, num_labels]."
        )
    if support_features.size(0) != support_label_weights.size(0):
        raise ValueError("Support feature and label-weight counts must match.")

    weights = support_label_weights.transpose(0, 1)
    weight_sums = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return weights @ support_features / weight_sums


def score_query_by_prototypes(query_features, prototypes, metric="euclidean"):
    if query_features.dim() != 2:
        raise ValueError("query_features must have shape [num_query, feature_dim].")
    if prototypes.dim() != 2:
        raise ValueError("prototypes must have shape [num_labels, feature_dim].")
    if query_features.size(1) != prototypes.size(1):
        raise ValueError("Feature dimensions must match.")

    if metric == "euclidean":
        distances = torch.cdist(query_features, prototypes, p=2).pow(2)
        return -distances
    if metric == "cosine":
        query = torch.nn.functional.normalize(query_features, dim=1)
        labels = torch.nn.functional.normalize(prototypes, dim=1)
        return query @ labels.transpose(0, 1)

    raise ValueError("metric must be either 'euclidean' or 'cosine'.")
