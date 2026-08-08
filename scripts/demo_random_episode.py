from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import FLEMProtoNet
from src.training import evaluate_episode, train_episode


def make_random_multilabels(num_samples, num_labels, min_labels=1):
    labels = torch.zeros(num_samples, num_labels)
    for sample_idx in range(num_samples):
        label_count = torch.randint(min_labels, num_labels + 1, size=(1,)).item()
        label_indices = torch.randperm(num_labels)[:label_count]
        labels[sample_idx, label_indices] = 1.0
    return labels


def main():
    torch.manual_seed(7)

    num_support = 8
    num_query = 4
    num_labels = 5
    image_size = 16
    num_features = 64

    support_images = torch.randn(num_support, 3, image_size, image_size)
    query_images = torch.randn(num_query, 3, image_size, image_size)
    support_labels = make_random_multilabels(num_support, num_labels)
    query_labels = make_random_multilabels(num_query, num_labels)

    model = FLEMProtoNet(
        num_features=num_features,
        num_labels=num_labels,
        metric="euclidean",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    outputs, losses = train_episode(
        model=model,
        optimizer=optimizer,
        support_images=support_images,
        support_labels=support_labels,
        query_images=query_images,
        query_labels=query_labels,
        support_loss_weight=0.1,
    )
    eval_outputs, eval_losses = evaluate_episode(
        model=model,
        support_images=support_images,
        support_labels=support_labels,
        query_images=query_images,
        query_labels=query_labels,
        support_loss_weight=0.1,
    )

    print("demo_random_episode: ok")
    print(f"logits shape: {tuple(outputs['logits'].shape)}")
    print(f"probabilities shape: {tuple(outputs['probabilities'].shape)}")
    print(f"prototypes shape: {tuple(outputs['prototypes'].shape)}")
    print(f"total loss: {losses['total'].item():.6f}")
    print(f"prediction loss: {losses['prediction'].item():.6f}")
    print(
        "support label enhancement loss: "
        f"{losses['support_label_enhancement'].item():.6f}"
    )
    print(f"eval logits shape: {tuple(eval_outputs['logits'].shape)}")
    print(f"eval total loss: {eval_losses['total'].item():.6f}")


if __name__ == "__main__":
    main()
