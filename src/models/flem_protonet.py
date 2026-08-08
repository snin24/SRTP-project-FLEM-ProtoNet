import torch
from torch import nn

from .encoder import Conv4Encoder
from .feature_projector import FeatureProjector
from .label_enhancer import LabelEnhancer
from .prototypes import build_label_prototypes, score_query_by_prototypes


class FLEMProtoNet(nn.Module):
    def __init__(
        self,
        num_features,
        num_labels,
        encoder=None,
        input_is_feature=False,
        input_dim=None,
        feature_projector=None,
        label_enhancer=None,
        label_enhancer_hidden_dim=64,
        metric="euclidean",
        label_weight_mode="enhanced",
        use_bias=True,
    ):
        super().__init__()
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be either 'euclidean' or 'cosine'.")
        if label_weight_mode not in {"enhanced", "positive_gated", "binary"}:
            raise ValueError(
                "label_weight_mode must be one of: enhanced, positive_gated, binary."
            )

        self.num_features = num_features
        self.num_labels = num_labels
        self.metric = metric
        self.label_weight_mode = label_weight_mode
        self.input_is_feature = input_is_feature
        if input_is_feature:
            if feature_projector is not None:
                self.encoder = feature_projector
            elif input_dim is not None and input_dim != num_features:
                self.encoder = FeatureProjector(
                    input_dim=input_dim,
                    output_dim=num_features,
                )
            else:
                self.encoder = nn.Identity()
        else:
            self.encoder = encoder if encoder is not None else Conv4Encoder()
        if label_weight_mode in {"enhanced", "positive_gated"}:
            self.label_enhancer = (
                label_enhancer
                if label_enhancer is not None
                else LabelEnhancer(
                    num_features=num_features,
                    num_classes=num_labels,
                    hidden_dim=label_enhancer_hidden_dim,
                )
            )
            self.aux_classifier = nn.Linear(num_features, num_labels)
        else:
            self.label_enhancer = None
            self.aux_classifier = None
        self.bias = nn.Parameter(torch.zeros(num_labels)) if use_bias else None

    def forward(self, support_images, support_labels, query_images):
        if support_labels.dim() != 2:
            raise ValueError("support_labels must have shape [num_support, num_labels].")
        if support_labels.size(1) != self.num_labels:
            raise ValueError("support_labels num_labels does not match model config.")

        support_features = self.encoder(support_images)
        query_features = self.encoder(query_images)
        

        if support_features.size(1) != self.num_features:
            raise ValueError("Encoded support feature dimension does not match num_features.")
        if query_features.size(1) != self.num_features:
            raise ValueError("Encoded query feature dimension does not match num_features.")

        support_labels = support_labels.to(dtype=support_features.dtype)
        if self.label_weight_mode in {"enhanced", "positive_gated"}:
            support_label_logits = self.label_enhancer(
                support_features,
                support_labels,
            )
            enhanced_label_weights = torch.sigmoid(support_label_logits)
            if self.label_weight_mode == "positive_gated":
                support_label_weights = support_labels * enhanced_label_weights
            else:
                support_label_weights = enhanced_label_weights
            support_classifier_logits = self.aux_classifier(support_features)
        else:
            support_label_logits = None
            support_label_weights = support_labels
            support_classifier_logits = None
        prototypes = build_label_prototypes(
            support_features,
            support_label_weights,
        )
        logits = score_query_by_prototypes(
            query_features,
            prototypes,
            metric=self.metric,
        )
        if self.bias is not None:
            logits = logits + self.bias

        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "prototypes": prototypes,
            "support_features": support_features,
            "query_features": query_features,
            "support_label_logits": support_label_logits,
            "support_label_weights": support_label_weights,
            "support_classifier_logits": support_classifier_logits,
            "label_weight_mode": self.label_weight_mode,
        }
