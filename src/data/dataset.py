from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz
from torch.utils.data import Dataset


DATASET_METADATA = {
    "VOC2007": {
        "num_labels": 20,
        "feature_dim": 2048,
    },
    "COCO": {
        "num_labels": 80,
        "feature_dim": 2048,
    },
    "NUSWIDE": {
        "num_labels": 21,
        "feature_dim": 2048,
    },
}

BACKBONE_FEATURE_DIM = {
    "resnet50": 2048,
    "resnet101": 2048,
    "conv4": 1600,
}


class MultiLabelFeatureDataset(Dataset):
    def __init__(
        self,
        data_root="datasets/VOC2007",
        dataset_name="VOC2007",
        phase="train",
        backbone="resnet50",
        mmap_features=True,
    ):
        if dataset_name not in DATASET_METADATA:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}.")
        if phase not in {"train", "val", "test"}:
            raise ValueError("phase must be one of: train, val, test.")

        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.phase = phase
        self.backbone = backbone
        self.meta = DATASET_METADATA[dataset_name]

        feature_path = (
            self.data_root
            / backbone
            / f"{dataset_name}_{phase}_features_{backbone}.npy"
        )
        label_path = (
            self.data_root
            / backbone
            / f"{dataset_name}_{phase}_labels.npz"
        )

        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        mmap_mode = "r" if mmap_features else None
        self.features = np.load(feature_path, mmap_mode=mmap_mode)
        self.labels = load_npz(label_path).toarray().astype(np.float32)

        if self.features.ndim != 2:
            raise ValueError("features must have shape [num_samples, feature_dim].")
        if self.labels.ndim != 2:
            raise ValueError("labels must have shape [num_samples, num_labels].")
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError("features and labels must have the same sample count.")
        expected_feature_dim = BACKBONE_FEATURE_DIM.get(
            self.backbone, self.meta["feature_dim"]
        )
        if self.features.shape[1] != expected_feature_dim:
            raise ValueError("feature dimension does not match dataset metadata.")
        if self.labels.shape[1] != self.meta["num_labels"]:
            raise ValueError("label dimension does not match dataset metadata.")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        feature = torch.from_numpy(np.asarray(self.features[index]).copy()).float()
        label = torch.from_numpy(self.labels[index]).float()
        sample_id = int(index)

        return {
            "feature": feature,
            "label": label,
            "sample_id": sample_id,
            "image": feature,
            "labels": label,
            "idx": sample_id,
        }
