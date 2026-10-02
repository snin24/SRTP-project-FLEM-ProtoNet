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
from src.utils.metrics import EPISODE_PROTOCOL
from scripts.train import evaluate_sampler


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
    parser.add_argument("--backbone", default=None, choices=["resnet50", "resnet101", "conv4"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.backbone is not None:
        config["dataset"]["backbone"] = args.backbone
        if args.backbone == "conv4":
            config["dataset"]["input_dim"] = 1600
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
        # Checkpoints produced by this trusted project include the training
        # config/metrics in addition to tensors; explicitly allow the full
        # pickle format on PyTorch >=2.6.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("evaluation_protocol") != EPISODE_PROTOCOL:
            raise ValueError("Checkpoint uses an older evaluation protocol; retrain before formal evaluation.")
        for key in ("dataset", "episode", "model"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Checkpoint/config mismatch: {key}")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded_checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    threshold = checkpoint["decision_threshold"]
    metrics = evaluate_sampler(
        model, sampler, config, device, episodes, threshold, threshold_search=False,
    )
    metrics["threshold-from-validation"] = threshold
    print_metrics(metrics)


if __name__ == "__main__":
    main()
