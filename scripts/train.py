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
from src.training import evaluate_episode, train_episode
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


def build_dataset(config, phase):
    dataset_config = config["dataset"]
    return MultiLabelFeatureDataset(
        data_root=dataset_config["data_root"],
        dataset_name=dataset_config["name"],
        phase=phase,
        backbone=dataset_config["backbone"],
    )


def build_sampler(config, dataset, seed):
    episode_config = config["episode"]
    return MultiLabelEpisodeSampler(
        dataset=dataset,
        num_episode_labels=episode_config["num_labels"],
        num_support_per_label=episode_config["num_support_per_label"],
        num_query=episode_config["num_query"],
        seed=seed,
    )


def build_model(config):
    dataset_config = config["dataset"]
    model_config = config["model"]
    return FLEMProtoNet(
        num_features=model_config["num_features"],
        num_labels=dataset_config["num_labels"],
        input_is_feature=True,
        input_dim=dataset_config["input_dim"],
        metric=model_config["metric"],
        label_weight_mode=model_config.get("label_weight_mode", "enhanced"),
        label_enhancer_hidden_dim=model_config.get("label_enhancer_hidden_dim", 64),
    )


def evaluate_sampler(model, sampler, config, device, episodes, threshold):
    all_probabilities = []
    all_labels = []
    losses = []
    training_config = config["training"]

    for _ in range(episodes):
        episode = move_episode_to_device(sampler.sample_episode(), device)
        outputs, loss_parts = evaluate_episode(
            model=model,
            support_images=episode["support_features"],
            support_labels=episode["support_labels"],
            query_images=episode["query_features"],
            query_labels=episode["query_labels"],
            support_loss_weight=training_config["support_loss_weight"],
            support_loss_type=training_config.get("support_loss_type", "flem"),
            flem_alpha=training_config.get("flem_alpha", 0.001),
            flem_beta=training_config.get("flem_beta", 0.001),
            flem_method=training_config.get("flem_method", "ld"),
            flem_threshold=training_config.get("flem_threshold", 0.0),
        )
        all_probabilities.append(outputs["probabilities"].cpu())
        all_labels.append(episode["query_labels"].cpu())
        losses.append(loss_parts["total"].item())

    probabilities = torch.cat(all_probabilities, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = evaluate_multilabel_predictions(
        labels,
        probabilities,
        threshold=threshold,
        threshold_search=config["evaluation"].get("threshold_search", False),
    )
    metrics["loss"] = float(np.mean(losses))
    return metrics


def print_metrics(prefix, metrics):
    metric_text = " ".join(
        f"{name}={value:.4f}" for name, value in sorted(metrics.items())
    )
    print(f"{prefix} {metric_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/voc2007_feature.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config["training"]["episodes"] = args.episodes
    if args.eval_every is not None:
        config["training"]["eval_every"] = args.eval_every
    if args.eval_episodes is not None:
        config["training"]["eval_episodes"] = args.eval_episodes
    if args.checkpoint is not None:
        config["training"]["checkpoint_path"] = args.checkpoint

    set_seed(config["seed"])
    device = resolve_device(args.device)

    train_dataset = build_dataset(config, phase="train")
    val_dataset = build_dataset(config, phase="val")
    train_sampler = build_sampler(config, train_dataset, seed=config["seed"])
    val_sampler = build_sampler(config, val_dataset, seed=config["seed"] + 1)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_micro_f1 = -1.0
    checkpoint_path = Path(config["training"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for episode_idx in range(1, config["training"]["episodes"] + 1):
        episode = move_episode_to_device(train_sampler.sample_episode(), device)
        _, losses = train_episode(
            model=model,
            optimizer=optimizer,
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

        if episode_idx % config["training"]["log_every"] == 0:
            # print(
            #     f"episode={episode_idx} "
            #     f"loss={losses['total'].item():.4f} "
            #     f"pred={losses['prediction'].item():.4f} "
            #     f"support={losses.get('support', torch.tensor(0.0)).item():.4f}"
            # )
            print(f"episode={episode_idx}")

        if episode_idx % config["training"]["eval_every"] == 0:
            metrics = evaluate_sampler(
                model=model,
                sampler=val_sampler,
                config=config,
                device=device,
                episodes=config["training"]["eval_episodes"],
                threshold=config["evaluation"]["threshold"],
            )
            # print_metrics(f"val episode={episode_idx}", metrics)
            print(f"val episode={episode_idx} mAP={metrics.get('Tuned-mAP', metrics['mAP']):.4f}")

            checkpoint_metric = metrics.get("Best-Micro-F1", metrics["Micro-F1"])
            if checkpoint_metric > best_micro_f1:
                best_micro_f1 = checkpoint_metric
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "metrics": metrics,
                        "episode": episode_idx,
                    },
                    checkpoint_path,
                )
                print(f"saved_checkpoint={checkpoint_path}")

    print("train: done")


if __name__ == "__main__":
    main()
