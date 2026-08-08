# FLEM-ProtoNet 完整运行命令

> AutoDL 环境，数据集路径均为 AutoDL 公共数据集。

---

## 0. 环境准备

```bash
# 激活 conda 环境
conda activate AI_Intro

# 进入项目目录
cd /root/FLEM-ProtoNet-MultiLabel-FewShot

# 安装依赖（如未安装）
pip install torch torchvision pycocotools scipy tqdm pillow numpy

# 解压 VOC2007（首次运行前）
tar -xzf /root/autodl-pub/VOCdevkit/VOC2007.tar.gz -C /root/autodl-pub/VOCdevkit/
```

---

## 1. 特征提取

> 特征保存到 `datasets/{数据集名}/{backbone}/`，命名格式：`{数据集}_{phase}_features_{backbone}.npy` + `_labels.npz`

### 1.1 VOC2007（20 类，多标签）

VOC2007 只有 `trainval` 和 `test` 两个公开 split，需从 trainval 提取后手动划分 train/val。

```bash
# ---------- resnet50 ----------
# 提取 trainval（后续手动划分 train/val）
python scripts/extract_features.py \
  --dataset VOC2007 \
  --image-root /root/autodl-pub/VOCdevkit/VOC2007/JPEGImages \
  --split-file /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
  --image-sets-dir /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main \
  --backbone resnet50 \
  --phase trainval \
  --output-dir datasets/VOC2007 \
  --batch-size 128

# ---------- resnet101 ----------
python scripts/extract_features.py \
  --dataset VOC2007 \
  --image-root /root/autodl-pub/VOCdevkit/VOC2007/JPEGImages \
  --split-file /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
  --image-sets-dir /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main \
  --backbone resnet101 \
  --phase trainval \
  --output-dir datasets/VOC2007 \
  --batch-size 128

# ---------- conv4 ----------
python scripts/extract_features.py \
  --dataset VOC2007 \
  --image-root /root/autodl-pub/VOCdevkit/VOC2007/JPEGImages \
  --split-file /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
  --image-sets-dir /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main \
  --backbone conv4 \
  --phase trainval \
  --output-dir datasets/VOC2007 \
  --batch-size 256
```

**VOC2007 划分 train/val/test 脚本：**

```bash
# 将 trainval 特征按 8:1:1 划分为 train/val/test
python -c "
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from pathlib import Path

for backbone in ['resnet50', 'resnet101', 'conv4']:
    base = Path(f'datasets/VOC2007/{backbone}')
    feats = np.load(base / 'VOC2007_trainval_features_{}.npy'.format(backbone))
    labels = load_npz(base / 'VOC2007_trainval_labels.npz').toarray().astype(np.float32)

    rng = np.random.RandomState(42)
    n = len(feats)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    for phase, ids in [('train', idx[:n_train]),
                        ('val',   idx[n_train:n_train+n_val]),
                        ('test',  idx[n_train+n_val:])]:
        np.save(base / f'VOC2007_{phase}_features_{backbone}.npy', feats[ids].astype(np.float32))
        save_npz(base / f'VOC2007_{phase}_labels.npz', csr_matrix(labels[ids].astype(np.float32)))
        print(f'{backbone} {phase}: {len(ids)} samples')
"
```

### 1.2 COCO2017（80 类，多标签）

```bash
# ---------- resnet50 ----------
# train
python scripts/extract_features.py \
  --dataset COCO \
  --image-root /root/autodl-tmp/FLEM-ProtoNet/COCO2017/train2017 \
  --ann-file /root/autodl-tmp/FLEM-ProtoNet/COCO2017/annotations/instances_train2017.json \
  --backbone resnet50 \
  --phase train \
  --output-dir datasets/COCO17 \
  --batch-size 128

# val
python scripts/extract_features.py \
  --dataset COCO \
  --image-root /root/autodl-tmp/FLEM-ProtoNet/COCO2017/val2017 \
  --ann-file /root/autodl-tmp/FLEM-ProtoNet/COCO2017/annotations/instances_val2017.json \
  --backbone resnet50 \
  --phase val \
  --output-dir datasets/COCO17 \
  --batch-size 128

# ---------- resnet101 ----------
python scripts/extract_features.py \
  --dataset COCO \
  --image-root /root/autodl-tmp/FLEM-ProtoNet/COCO2017/train2017 \
  --ann-file /root/autodl-tmp/FLEM-ProtoNet/COCO2017/annotations/instances_train2017.json \
  --backbone resnet101 \
  --phase train \
  --output-dir datasets/COCO17 \
  --batch-size 128

python scripts/extract_features.py \
  --dataset COCO \
  --image-root /root/autodl-tmp/FLEM-ProtoNet/COCO2017/val2017 \
  --ann-file /root/autodl-tmp/FLEM-ProtoNet/COCO2017/annotations/instances_val2017.json \
  --backbone resnet101 \
  --phase val \
  --output-dir datasets/COCO17 \
  --batch-size 128

# ---------- conv4 ----------
python scripts/extract_features.py \
  --dataset COCO \
  --image-root /root/autodl-tmp/FLEM-ProtoNet/COCO2017/train2017 \
  --ann-file /root/autodl-tmp/FLEM-ProtoNet/COCO2017/annotations/instances_train2017.json \
  --backbone conv4 \
  --phase train \
  --output-dir datasets/COCO17 \
  --batch-size 256

python scripts/extract_features.py \
  --dataset COCO \
  --image-root /root/autodl-tmp/FLEM-ProtoNet/COCO2017/val2017 \
  --ann-file /root/autodl-tmp/FLEM-ProtoNet/COCO2017/annotations/instances_val2017.json \
  --backbone conv4 \
  --phase val \
  --output-dir datasets/COCO17 \
  --batch-size 256
```

**COCO 划分 val→val/test 脚本（val 按 1:1 划分）：**

```bash
# COCO val2017 按 1:1 划分为 val/test
python -c "
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from pathlib import Path

for backbone in ['resnet50', 'resnet101', 'conv4']:
    base = Path(f'datasets/COCO17/{backbone}')
    feats = np.load(base / f'COCO_val_features_{backbone}.npy')
    labels = load_npz(base / 'COCO_val_labels.npz').toarray().astype(np.float32)

    rng = np.random.RandomState(42)
    n = len(feats)
    idx = rng.permutation(n)
    half = n // 2

    for phase, ids in [('val',  idx[:half]),
                        ('test', idx[half:])]:
        np.save(base / f'COCO_{phase}_features_{backbone}.npy', feats[ids].astype(np.float32))
        save_npz(base / f'COCO_{phase}_labels.npz', csr_matrix(labels[ids].astype(np.float32)))
        print(f'{backbone} {phase}: {len(ids)} samples')
"
```

### 1.3 NUS-WIDE（21 类，多标签）

> 结构：`database_img.txt` + `database_label.txt`（train），`test_img.txt` + `test_label.txt`（test），
> train 按 80/20 自动切 train/val。图片在 `images/` 下。
> 本地路径：`D:\Downloads\NUS-WIDE`

```bash
# ---------- resnet50 ----------
python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone resnet50 \
  --phase train \
  --output-dir datasets/NUSWIDE \
  --batch-size 128

python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone resnet50 \
  --phase val \
  --output-dir datasets/NUSWIDE \
  --batch-size 128

python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone resnet50 \
  --phase test \
  --output-dir datasets/NUSWIDE \
  --batch-size 128

# ---------- resnet101 ----------
python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone resnet101 \
  --phase train \
  --output-dir datasets/NUSWIDE \
  --batch-size 128

python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone resnet101 \
  --phase val \
  --output-dir datasets/NUSWIDE \
  --batch-size 128

python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone resnet101 \
  --phase test \
  --output-dir datasets/NUSWIDE \
  --batch-size 128

# ---------- conv4 ----------
python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone conv4 \
  --phase train \
  --output-dir datasets/NUSWIDE \
  --batch-size 256

python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone conv4 \
  --phase val \
  --output-dir datasets/NUSWIDE \
  --batch-size 256

python scripts/extract_features.py \
  --dataset NUSWIDE \
  --image-root D:/Downloads/NUS-WIDE \
  --backbone conv4 \
  --phase test \
  --output-dir datasets/NUSWIDE \
  --batch-size 256
```

---

## 2. 训练

> 所有 backbone 均使用 resnet50 提取的特征（config 中 `backbone: "resnet50"`）。  
> 更换 backbone 特征时，修改 config 中 `dataset.backbone` 和 `dataset.input_dim`。

### 2.1 VOC2007

```bash
# ProtoNet-ML Baseline（binary 原型）
python scripts/train.py --config configs/voc2007_baseline.json --device cuda

# FLEM-ProtoNet（positive_gated 增强）
python scripts/train.py --config configs/voc2007_feature.json --device cuda
```

### 2.2 COCO2017

```bash
# ProtoNet-ML Baseline
python scripts/train.py --config configs/coco17_baseline.json --device cuda

# FLEM-ProtoNet
python scripts/train.py --config configs/coco17_feature.json --device cuda
```

### 2.3 NUS-WIDE

```bash
# ProtoNet-ML Baseline
python scripts/train.py --config configs/nuswide_baseline.json --device cuda

# FLEM-ProtoNet
python scripts/train.py --config configs/nuswide_feature.json --device cuda
```

---

## 3. 评估

### 3.1 VOC2007

```bash
# Baseline
python scripts/evaluate.py \
  --config configs/voc2007_baseline.json \
  --checkpoint experiments/checkpoints/voc2007_protonet_ml_best.pt \
  --phase test --device cuda

# FLEM-ProtoNet
python scripts/evaluate.py \
  --config configs/voc2007_feature.json \
  --checkpoint experiments/checkpoints/voc2007_feature_best.pt \
  --phase test --device cuda
```

### 3.2 COCO2017

```bash
# Baseline
python scripts/evaluate.py \
  --config configs/coco17_baseline.json \
  --checkpoint experiments/checkpoints/coco17_baseline_best.pt \
  --phase test --device cuda

# FLEM-ProtoNet
python scripts/evaluate.py \
  --config configs/coco17_feature.json \
  --checkpoint experiments/checkpoints/coco17_best.pt \
  --phase test --device cuda
```

### 3.3 NUS-WIDE

```bash
# Baseline
python scripts/evaluate.py \
  --config configs/nuswide_baseline.json \
  --checkpoint experiments/checkpoints/nuswide_baseline_best.pt \
  --phase test --device cuda

# FLEM-ProtoNet
python scripts/evaluate.py \
  --config configs/nuswide_feature.json \
  --checkpoint experiments/checkpoints/nuswide_best.pt \
  --phase test --device cuda
```

---

## 4. 一键脚本（可选）

将所有特征提取合并为一次性执行：

```bash
# ===== 全部特征提取（resnet50） =====
BACKBONE=resnet50

# VOC2007
python scripts/extract_features.py --dataset VOC2007 \
  --image-root /root/autodl-pub/VOCdevkit/VOC2007/JPEGImages \
  --split-file /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt \
  --image-sets-dir /root/autodl-pub/VOCdevkit/VOC2007/ImageSets/Main \
  --backbone $BACKBONE --phase trainval --output-dir datasets/VOC2007

# COCO
python scripts/extract_features.py --dataset COCO \
  --image-root /root/autodl-pub/COCO2017/train2017 \
  --ann-file /root/autodl-pub/COCO2017/annotations/instances_train2017.json \
  --backbone $BACKBONE --phase train --output-dir datasets/COCO17

python scripts/extract_features.py --dataset COCO \
  --image-root /root/autodl-pub/COCO2017/val2017 \
  --ann-file /root/autodl-pub/COCO2017/annotations/instances_val2017.json \
  --backbone $BACKBONE --phase val --output-dir datasets/COCO17

# ===== 划分 train/val/test（VOC2007 + COCO） =====
# 然后执行上面第 1 节中的划分脚本

# ===== 全部训练 + 评估 =====
for CONFIG in \
  configs/voc2007_baseline.json configs/voc2007_feature.json \
  configs/coco17_baseline.json configs/coco17_feature.json \
  configs/nuswide_baseline.json configs/nuswide_feature.json; do
  echo "=== Training: $CONFIG ==="
  python scripts/train.py --config $CONFIG --device cuda
  echo "=== Evaluating: $CONFIG ==="
  python scripts/evaluate.py --config $CONFIG --phase test --device cuda
done
```

---

## 附：更换 Backbone 训练

| backbone | input_dim | 说明 |
|----------|-----------|------|
| resnet50 | 2048 | ImageNet 预训练，224×224 输入 |
| resnet101 | 2048 | ImageNet 预训练，224×224 输入 |
| conv4 | 1600 | 随机初始化，84×84 输入 |

如需用非默认 backbone 特征训练，复制 config 后修改两处：

```json
{
  "dataset": {
    "backbone": "conv4",
    "input_dim": 1600
  }
}
```
