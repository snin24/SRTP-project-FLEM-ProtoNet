#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


def split_trainval(base_dir: Path, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    base_dir = Path(base_dir)
    backbones = [p.name for p in base_dir.iterdir() if p.is_dir()]
    backbones.sort()

    rng = np.random.RandomState(seed)

    for backbone in backbones:
        backbone_dir = base_dir / backbone
        feature_path = backbone_dir / f"VOC2007_trainval_features_{backbone}.npy"
        label_path = backbone_dir / "VOC2007_trainval_labels.npz"

        if not feature_path.exists() or not label_path.exists():
            print(f"skip {backbone}: missing {feature_path.name} or {label_path.name}")
            continue

        feats = np.load(feature_path)
        labels = load_npz(label_path).toarray().astype(np.float32)

        n = len(feats)
        idx = rng.permutation(n)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for phase, ids in [
            ("train", idx[:n_train]),
            ("val", idx[n_train:n_train + n_val]),
            ("test", idx[n_train + n_val:]),
        ]:
            np.save(backbone_dir / f"VOC2007_{phase}_features_{backbone}.npy", feats[ids].astype(np.float32))
            save_npz(backbone_dir / f"VOC2007_{phase}_labels.npz", csr_matrix(labels[ids].astype(np.float32)))
            print(f"{backbone} {phase}: {len(ids)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Split VOC2007 trainval features and labels into train/val/test."
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/VOC2007",
        help="Feature root directory containing backbone subfolders.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train proportion.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation proportion.")
    args = parser.parse_args()

    split_trainval(Path(args.data_dir), seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
