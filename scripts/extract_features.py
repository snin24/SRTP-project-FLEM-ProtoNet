import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.sparse import csr_matrix, save_npz
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    resnet50,
    resnet101,
    ResNet50_Weights,
    ResNet101_Weights,
)
from tqdm import tqdm


# ——— backbone 配置 ———
BACKBONE_CONFIGS = {
    "resnet50": {
        "builder": lambda: resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        "feature_dim": 2048,
        "fc_attr": "fc",
        "transform": None,  # 使用 DEFAULT_TRANSFORM
    },
    "resnet101": {
        "builder": lambda: resnet101(weights=ResNet101_Weights.IMAGENET1K_V1),
        "feature_dim": 2048,
        "fc_attr": "fc",
        "transform": None,
    },
    "conv4": {
        "builder": lambda: _build_conv4(),
        "feature_dim": 1600,
        "fc_attr": None,
        "transform": "conv4",  # 84×84
    },
}


def _build_conv4():
    """构建标准 ProtoNet Conv4 编码器（84×84 输入 → 1600 维输出）。"""
    from torch import nn

    def conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    return nn.Sequential(
        conv_block(3, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        conv_block(64, 64),
        nn.Flatten(),
    )

CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

DEFAULT_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CONV4_TRANSFORM = T.Compose([
    T.Resize(84),
    T.CenterCrop(84),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VOC2007MultiLabelDataset(Dataset):
    def __init__(self, image_root, split_file, image_sets_dir=None, transform=None):
        self.image_root = Path(image_root)
        self.transform = transform or DEFAULT_TRANSFORM

        with open(split_file, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        self.labels = np.zeros((len(self.image_ids), len(CLASS_NAMES)), dtype=np.float32)
        id_to_idx = {img_id: idx for idx, img_id in enumerate(self.image_ids)}

        if image_sets_dir is None:
            image_sets_dir = Path(split_file).parent

        split_name = Path(split_file).stem
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_file = Path(image_sets_dir) / f"{cls_name}_{split_name}.txt"
            if not cls_file.exists():
                continue
            with open(cls_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_id, label_val = parts[0], int(parts[1])
                        if label_val == 1 and img_id in id_to_idx:
                            self.labels[id_to_idx[img_id], cls_idx] = 1.0

        valid_mask = self.labels.sum(axis=1) > 0
        self.image_ids = [img_id for img_id, v in zip(self.image_ids, valid_mask) if v]
        self.labels = self.labels[valid_mask]

        print(f"VOC2007 {split_name}: {len(self.image_ids)} images, "
              f"{self.labels.sum(axis=1).mean():.1f} labels/image")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        for ext in (".jpg", ".png", ".JPEG"):
            img_path = self.image_root / f"{self.image_ids[idx]}{ext}"
            if img_path.exists():
                break
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(self.labels[idx]), idx


class COCOMultiLabelDataset(Dataset):
    def __init__(self, image_root, ann_file, transform=None):
        from pycocotools.coco import COCO

        self.image_root = Path(image_root)
        self.transform = transform or DEFAULT_TRANSFORM
        self.coco = COCO(ann_file)

        self.cat_ids = sorted(self.coco.getCatIds())
        self.num_classes = len(self.cat_ids)
        self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = sorted(self.coco.getImgIds())

        self.labels = np.zeros((len(self.img_ids), self.num_classes), dtype=np.float32)
        for idx, img_id in enumerate(self.img_ids):
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
                cat_idx = self.cat_id_to_idx.get(ann["category_id"])
                if cat_idx is not None:
                    self.labels[idx, cat_idx] = 1.0

        valid_mask = self.labels.sum(axis=1) > 0
        self.img_ids = [img_id for img_id, v in zip(self.img_ids, valid_mask) if v]
        self.labels = self.labels[valid_mask]

        print(f"COCO: {len(self.img_ids)} images, {self.num_classes} classes, "
              f"{self.labels.sum(axis=1).mean():.1f} labels/image")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = self.image_root / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(self.labels[idx]), idx


class NUSWIDEDataset(Dataset):
    """NUS-WIDE 数据集，多标签。

    数据集结构：
        NUS-WIDE/
          images/                    # 所有图片（扁平存放）
          database_img.txt           # 训练集图片路径（相对路径，每行一个）
          database_label.txt         # 训练集标签（每行空格分隔的正类索引）
          test_img.txt               # 测试集图片路径
          test_label.txt             # 测试集标签

    phase:
      - train: database_img 按 80/20 切分取 80%
      - val:   database_img 按 80/20 切分取 20%
      - test:  test_img 全部
    """

    def __init__(self, image_root, phase="train", transform=None):
        self.image_root = Path(image_root)           # NUS-WIDE 根目录
        self.transform = transform or DEFAULT_TRANSFORM

        # ——— 1. 读取图片路径和标签 ———
        def _read_split(img_file, label_file):
            with open(self.image_root / img_file, "r") as f:
                paths = [line.strip() for line in f if line.strip()]
            labels_raw = []
            with open(self.image_root / label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        labels_raw.append([int(x) for x in line.split() if x])
                    else:
                        labels_raw.append([])
            return paths, labels_raw

        if phase == "test":
            img_paths, labels_raw = _read_split("test_img.txt", "test_label.txt")
        else:
            img_paths, labels_raw = _read_split("database_img.txt", "database_label.txt")

        # ——— 2. 确定类别数 ———
        all_indices = [idx for lbls in labels_raw for idx in lbls]
        self.num_classes = max(all_indices) + 1 if all_indices else 0

        # ——— 3. 转 multi-hot ———
        labels_full = np.zeros((len(img_paths), self.num_classes), dtype=np.float32)
        for i, lbls in enumerate(labels_raw):
            for idx in lbls:
                labels_full[i, idx] = 1.0

        # ——— 4. train/val 切分 ———
        if phase in ("train", "val"):
            rng = np.random.RandomState(42)
            indices = rng.permutation(len(img_paths))
            split_point = int(len(indices) * 0.8)
            if phase == "train":
                indices = sorted(indices[:split_point])
            else:
                indices = sorted(indices[split_point:])
        else:
            indices = list(range(len(img_paths)))

        self._img_paths = [img_paths[i] for i in indices]
        self.labels = labels_full[indices]

        # 过滤无标签样本
        valid_mask = self.labels.sum(axis=1) > 0
        self._img_paths = [p for p, v in zip(self._img_paths, valid_mask) if v]
        self.labels = self.labels[valid_mask]

        # 解析绝对路径
        self._abs_paths = []
        for p in self._img_paths:
            candidate = self.image_root / p
            if not candidate.exists():
                # 尝试去掉前缀，直接在 images/ 下找
                candidate = self.image_root / "images" / Path(p).name
            self._abs_paths.append(candidate if candidate.exists() else None)

        print(f"NUSWIDE {phase}: {len(self)} images, {self.num_classes} classes, "
              f"{self.labels.sum(axis=1).mean():.1f} labels/image")

    def __len__(self):
        return len(self._img_paths)

    def __getitem__(self, idx):
        img_path = self._abs_paths[idx]
        if img_path is None:
            raise FileNotFoundError(f"图片不存在: {self._img_paths[idx]}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(self.labels[idx]), idx


# ——— 特征提取核心 ———
def build_backbone_model(backbone_name, device):
    """根据 backbone 名称构建特征提取模型（去掉分类头）。"""
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(
            f"不支持的 backbone: {backbone_name}。可选: {list(BACKBONE_CONFIGS.keys())}"
        )
    cfg = BACKBONE_CONFIGS[backbone_name]
    model = cfg["builder"]()
    # 去掉分类头，输出特征向量（conv4 无 fc，跳过）
    fc_attr = cfg.get("fc_attr")
    if fc_attr is not None:
        setattr(model, fc_attr, torch.nn.Identity())
    model = model.to(device)
    model.eval()
    return model


def extract_features(
    dataset,
    output_dir,
    dataset_name,
    phase,
    backbone,
    batch_size=64,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  backbone: {backbone}")

    model = build_backbone_model(backbone, device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader):
            features = model(images.to(device))
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    output_dir = Path(output_dir) / backbone
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / f"{dataset_name}_{phase}_features_{backbone}.npy"
    label_path = output_dir / f"{dataset_name}_{phase}_labels.npz"

    np.save(feature_path, all_features.astype(np.float32))
    save_npz(label_path, csr_matrix(all_labels.astype(np.float32)))
    print(f"saved: {feature_path}  {all_features.shape}")
    print(f"saved: {label_path}    {all_labels.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["VOC2007", "COCO", "NUSWIDE"]
    )
    parser.add_argument("--image-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="datasets/VOC2007",
        help="基础输出目录，特征将保存到 {output_dir}/{backbone}/ 下",
    )
    parser.add_argument("--phase", default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="特征提取 backbone",
    )
    # VOC2007 专用
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--image-sets-dir", default=None)
    # COCO 专用
    parser.add_argument("--ann-file", default=None)
    # NUSWIDE 无额外参数
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # 根据 backbone 选择 transform
    backbone_cfg = BACKBONE_CONFIGS[args.backbone]
    if backbone_cfg.get("transform") == "conv4":
        transform = CONV4_TRANSFORM
    else:
        transform = DEFAULT_TRANSFORM

    if args.dataset == "VOC2007":
        if args.split_file is None:
            parser.error("VOC2007 需要 --split-file")
        dataset = VOC2007MultiLabelDataset(
            image_root=args.image_root,
            split_file=args.split_file,
            image_sets_dir=args.image_sets_dir,
            transform=transform,
        )
    elif args.dataset == "COCO":
        if args.ann_file is None:
            parser.error("COCO 需要 --ann-file")
        dataset = COCOMultiLabelDataset(
            image_root=args.image_root,
            ann_file=args.ann_file,
            transform=transform,
        )
    elif args.dataset == "NUSWIDE":
        dataset = NUSWIDEDataset(image_root=args.image_root, phase=args.phase, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    if len(dataset) == 0:
        raise RuntimeError("数据集为空！请检查路径参数。")

    extract_features(
        dataset=dataset,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        phase=args.phase,
        backbone=args.backbone,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
