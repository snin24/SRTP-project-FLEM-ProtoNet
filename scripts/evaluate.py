import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MultiLabelEpisodeSampler, MultiLabelFeatureDataset
from src.models import FLEMProtoNet
from src.training import evaluate_episode
from src.utils import resolve_device
from src.utils.metrics import evaluate_multilabel_predictions


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_episode_to_device(episode, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in episode.items()
    }


def build_model(config):
    return FLEMProtoNet(
        num_features=config["model"]["num_features"],
        num_labels=config["dataset"]["num_labels"],
        input_is_feature=True,
        input_dim=config["dataset"]["input_dim"],
        metric=config["model"]["metric"],
        label_weight_mode=config["model"].get("label_weight_mode", "enhanced"),
        label_enhancer_hidden_dim=config["model"].get("label_enhancer_hidden_dim", 64),
    )


def print_metrics(metrics):
    for name, value in sorted(metrics.items()):
        print(f"{name}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/voc2007_feature.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--phase", default=None, choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(args.device)

    phase = args.phase or config["evaluation"]["phase"]
    episodes = args.episodes or config["evaluation"]["episodes"]
    dataset = MultiLabelFeatureDataset(
        data_root=config["dataset"]["data_root"],
        dataset_name=config["dataset"]["name"],
        phase=phase,
        backbone=config["dataset"]["backbone"],
    )
    sampler = MultiLabelEpisodeSampler(
        dataset=dataset,
        num_episode_labels=config["episode"]["num_labels"],
        num_support_per_label=config["episode"]["num_support_per_label"],
        num_query=config["episode"]["num_query"],
        seed=config["seed"] + 2,
    )

    model = build_model(config).to(device)
    checkpoint_path = args.checkpoint or config["training"]["checkpoint_path"]
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded_checkpoint: {checkpoint_path}")
    else:
        print("loaded_checkpoint: none")

    all_probabilities = []
    all_labels = []
    losses = []
    for _ in range(episodes):
        episode = move_episode_to_device(sampler.sample_episode(), device)
        outputs, loss_parts = evaluate_episode(
            model=model,
            support_images=episode["support_features"],
            support_labels=episode["support_labels"],
            query_images=episode["query_features"],
            query_labels=episode["query_labels"],
            support_loss_weight=config["training"]["support_loss_weight"],
            support_loss_type=config["training"].get("support_loss_type", "flem"),
            flem_alpha=config["training"].get("flem_alpha", 0.001),
            flem_beta=config["training"].get("flem_beta", 0.001),
            flem_method=config["training"].get("flem_method", "ld"),
            flem_threshold=config["training"].get("flem_threshold", 0.0),
        )
        all_probabilities.append(outputs["probabilities"].cpu())
        all_labels.append(episode["query_labels"].cpu())
        losses.append(loss_parts["total"].item())

    probabilities = torch.cat(all_probabilities, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = evaluate_multilabel_predictions(
        labels,
        probabilities,
        threshold=config["evaluation"]["threshold"],
        threshold_search=config["evaluation"].get("threshold_search", False),
    )
    metrics["loss"] = float(np.mean(losses))
    print_metrics(metrics)


if __name__ == "__main__":
    main()
