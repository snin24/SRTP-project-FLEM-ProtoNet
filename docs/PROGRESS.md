# 项目进度记录

最后更新日期：2026-07-15

## 项目路径

```text
D:\pycharm_Projects\PythonProject\SRTP\FLEM-ProtoNet-MultiLabel-FewShot
```

## 当前目标

新建一个独立项目，用于实现多标签少样本学习方法。该方法结合两篇论文的思想：

- Prototypical Networks for Few-shot Learning
- Fusion Label Enhancement for Multi-Label Learning

当前方法暂定名为：

```text
FLEM-ProtoNet
```

整体思路是：

1. 使用特征提取器将图像映射到嵌入空间。
2. 使用 FLEM 风格的标签增强模块，将原始 0/1 多标签转成连续标签权重。
3. 使用增强标签权重构造每个标签的加权原型。
4. 将 query 样本特征与各个标签原型进行距离比较。
5. 使用逐标签 `sigmoid` 做多标签预测，而不是使用 `softmax`。

## 已完成内容

### 2026-07-22 更新
- 创建 `scripts/extract_features.py`：通用 ResNet50 特征提取脚本，支持 VOC2007 / COCO 原始图片 → .npy + .npz。
- 更新 `src/data/dataset.py`：DATASET_METADATA 新增 COCO（80类）。
- 消融实验：Dropout+LayerNorm (↓7.5%)、加深 Projector (↓1.3%)，均不如基线，已撤销。

### 文档部分

- 创建了 `README.md`，写入完整方法设计。
- 将 `README.md` 中的数学公式全部改成 `$$...$$` 公式环境。
- 在 `README.md` 中追加了“当前工程框架”说明。
- 创建了 `docs/PROJECT_STRUCTURE.md`，用于说明工程目录职责。
- 创建了本文件 `docs/PROGRESS.md`，用于记录当前进度。
- 更新 `docs/PROJECT_STRUCTURE.md`，补充 `prototypes.py` 模块说明。
- 更新本文件，记录加权标签原型模块已完成，并调整后续任务顺序。
- 创建 `scripts/demo_random_episode.py`，用于用随机 episode 验证模型、loss 和优化步骤。
- 已将 VOC2007 预提取特征数据复制到 `datasets/data/`。
- 创建 `src/data/dataset.py`，用于读取 VOC2007 ResNet50 预提取特征和多标签稀疏标签。
- 创建 `src/data/sampler.py`，用于从多标签数据中采样 support/query episode。
- 创建 `src/models/feature_projector.py`，用于将 2048 维预提取特征投影到原型嵌入空间。
- 更新 `src/models/flem_protonet.py`，支持直接输入预提取特征。
- 创建 `configs/voc2007_feature.json`，作为 VOC2007 特征实验默认配置。
- 创建 `scripts/train.py` 和 `scripts/evaluate.py`，跑通真实 VOC2007 特征数据的训练与评估入口。
- 创建 `.gitignore`，忽略 Python 缓存和 checkpoint 产物。
- 更新 `src/utils/metrics.py`，增加预测概率统计和验证集阈值扫描。
- 增加 ProtoNet-ML baseline：通过 `label_weight_mode="binary"` 直接使用原始 multi-hot 标签构造标签原型。
- 创建 `configs/voc2007_feature_protonet_ml.json`，用于和 FLEM-ProtoNet 做同采样、同训练流程对比。
- 迁移 FLEM 原始风格损失函数，包括 `loss_enhanced`、`ld`、`threshold`
  和 `sim` 三种 LE-specific loss。
- 在 FLEM-ProtoNet 中加入 support auxiliary classifier，用于提供原 FLEM loss
  中的 `pred` 分支。
- FLEM-ProtoNet 的 support loss 默认从简化 BCE 改为 FLEM-style pred/LE 互教损失。

### 工程框架

已经创建如下目录结构：

```text
FLEM-ProtoNet-MultiLabel-FewShot/
  README.md
  configs/
  scripts/
  src/
    data/
    models/
    training/
    utils/
  docs/
    PROJECT_STRUCTURE.md
    PROGRESS.md
  experiments/
  datasets/
    data/
```

各目录当前职责：

- `configs/`：存放实验配置文件。
- `scripts/`：存放训练、评估和数据检查入口脚本。
- `src/data/`：存放数据集读取和 episode sampler。
- `src/models/`：存放模型相关模块。
- `src/training/`：后续存放训练循环、损失组合和 checkpoint 逻辑。
- `src/utils/`：存放通用工具和评价指标。
- `docs/`：存放说明文档、结构图和项目记录。
- `experiments/`：后续存放实验结果、配置快照和消融记录。
- `datasets/data/`：存放 VOC2007 预提取特征和标签文件。

### 已迁移的基础模块

这些模块属于“不需要大改，可以先迁移”的部分。

从 ProtoNet 项目迁移：

- `src/models/encoder.py`
  - 包含 `Conv4Encoder`。
  - 来源是原 ProtoNet 项目中的 Conv4 编码器。

- `src/models/distance.py`
  - 包含 `euclidean_dist`。
  - 包含 `cosine_similarity`。

从 FLEM 项目迁移：

- `src/models/label_enhancer.py`
  - 包含 `LabelEnhancer`。
  - 来源是 FLEM 项目中的 `LE` 模块。

- `src/utils/metrics.py`
  - 包含多标签评价指标。
  - 当前包括 Hamming Loss、Ranking Loss、Coverage、Micro-F1、Macro-F1 和 mAP。

- `src/models/prototypes.py`
  - 包含 `build_label_prototypes`。
  - 包含 `score_query_by_prototypes`。
  - 用增强标签权重构造标签原型，并支持 query 与标签原型的欧氏距离或余弦相似度打分。

- `src/models/flem_protonet.py`
  - 包含 `FLEMProtoNet`。
  - 串联 `Conv4Encoder`、`LabelEnhancer`、`build_label_prototypes`
    和 `score_query_by_prototypes`。
  - forward 输入 support 图像、support 标签和 query 图像，输出多标签 logits、probabilities、标签原型和中间特征。
  - 现在支持 `input_is_feature=True`，可直接接收 2048 维预提取特征。
  - 现在支持 `label_weight_mode="enhanced"` 和 `label_weight_mode="binary"`。
  - `enhanced` 是 FLEM-ProtoNet，使用 `LabelEnhancer` 输出的连续标签权重构造原型。
  - `positive_gated` 是保守增强版本，使用 `support_labels * sigmoid(LE)` 构造原型。
  - `binary` 是 ProtoNet-ML baseline，直接使用原始 multi-hot 标签构造原型。
  - enhanced/positive_gated 模式包含 `aux_classifier`，用于 support branch 的 FLEM-style loss。
  - 推理路径不使用 query 真实标签。

- `src/models/feature_projector.py`
  - 包含 `FeatureProjector`。
  - 默认将 VOC2007 ResNet50 的 2048 维特征投影到 `num_features` 维原型空间。

- `src/training/losses.py`
  - 包含 `multilabel_prototype_loss`。
  - 包含 `flem_protonet_loss`。
  - 主预测损失使用 `BCEWithLogitsLoss` 思路，直接接收 logits 和 query multi-hot 标签。
  - `flem_protonet_loss` 支持可选的 support 标签增强辅助损失。
  - 当前版本是可训练的最小损失组合，还不是 FLEM 原论文完整 `loss_enhanced` 的复刻。

- `src/training/trainer.py`
  - 包含 `train_episode`。
  - 包含 `evaluate_episode`。
  - `train_episode` 负责单个 episode 的 forward、loss、backward 和 optimizer step。
  - `evaluate_episode` 负责单个 episode 的无梯度 forward 和 loss 计算。
  - 当前训练器只处理已经构造好的 episode 张量，不包含数据读取和 episode 采样。

- `src/data/dataset.py`
  - 包含 `MultiLabelFeatureDataset`。
  - 读取 `VOC2007_{phase}_features_imagenet_resnet50.npy`。
  - 读取 `VOC2007_{phase}_labels.npz` 稀疏多标签矩阵，并转成 multi-hot 标签。
  - 单样本返回 `feature`、`label`、`sample_id`，同时保留 `image`、`labels`、`idx` 兼容字段。

- `src/data/sampler.py`
  - 包含 `MultiLabelEpisodeSampler`。
  - 每个 episode 采样若干标签。
  - 每个采样标签抽取指定数量的正样本进入 support set。
  - query set 从含有 episode 标签且不在 support set 中的样本中抽取。
  - support/query 都保留完整 20 维 multi-hot 标签。

### 配置和真实数据脚本

- `configs/voc2007_feature.json`
  - 默认使用 `datasets/data` 下的 VOC2007 ResNet50 预提取特征。
  - 默认设置为 5-label 1-shot，query 数为 16。
  - 默认模型嵌入维度为 128。
  - 默认 `label_weight_mode="enhanced"`。

- `configs/voc2007_feature_protonet_ml.json`
  - ProtoNet-ML baseline 配置。
  - 默认 `label_weight_mode="binary"`。
  - 默认 `support_loss_weight=0.0`，因为 baseline 不使用标签增强辅助损失。

- `scripts/train.py`
  - 读取配置。
  - 构造 train/val 数据集和 episode sampler。
  - 使用 `FLEMProtoNet(input_is_feature=True)` 训练。
  - 定期在 val split 上做 episode 评估。
  - 按 Micro-F1 保存 best checkpoint。

- `scripts/evaluate.py`
  - 读取配置。
  - 可加载训练保存的 checkpoint。
  - 在 train/val/test split 上按 episode 评估。
  - 输出 Hamming、Ranking、Coverage、Micro-F1、Macro-F1、mAP 和 loss。
  - 输出 `prob_min`、`prob_mean`、`prob_max`、`pred_positive_rate`
    和 `label_positive_rate`，用于判断固定阈值是否过高。
  - 支持 `threshold_search=true`，自动扫描阈值并输出 `Best-threshold`
    和 `Tuned-*` 指标。

### FLEM-style 损失迁移

当前 `src/training/losses.py` 已包含：

- `flem_loss_enhanced`
  - 对应原 FLEM 的 `loss_enhanced`。
  - 使用 teacher logits 对正负标签损失进行 one-sided weighting。

- `flem_original_style_loss`
  - 对应原 FLEM 的 `set_forward_loss` 主要逻辑。
  - 包含 `loss_pred_cls`、`loss_pred_ld`、`loss_le_cls`
    和 `loss_le_spec`。
  - 默认 `method="ld"`，并支持 `threshold` 和 `sim`。

在 FLEM-ProtoNet 中的适配方式：

```text
query_logits + query_labels
  -> query prototype BCE

support_classifier_logits + support_label_logits + support_labels
  -> FLEM-style support loss

total = query_loss + support_loss_weight * flem_support_loss
```

注意：加入 `aux_classifier` 后，旧的 enhanced checkpoint 和当前模型结构不完全一致。
需要重新训练 FLEM-ProtoNet checkpoint。

### Demo 脚本

- `scripts/demo_random_episode.py`
  - 使用随机生成的 support/query 图像和 multi-hot 标签。
  - 通过 `train_episode` 跑通 `FLEMProtoNet -> flem_protonet_loss -> backward -> optimizer.step()`。
  - 通过 `evaluate_episode` 跑通无梯度评估路径。
  - 打印 logits、probabilities、prototypes 的形状和 loss 分项。
  - 该脚本只用于验证最小训练链路，不替代正式训练器和真实数据集读取。

### Python 包结构

已经添加如下初始化文件：

```text
src/__init__.py
src/data/__init__.py
src/models/__init__.py
src/training/__init__.py
src/utils/__init__.py
```

这样后续可以用包导入方式组织代码。

### 语法检查

已经对新增 Python 文件执行过语法检查：

```text
python -m py_compile src\models\prototypes.py src\models\flem_protonet.py src\models\__init__.py src\training\losses.py src\training\trainer.py src\training\__init__.py scripts\demo_random_episode.py
python -m py_compile src\data\dataset.py src\data\sampler.py src\models\feature_projector.py scripts\train.py scripts\evaluate.py
```

结果：

```text
新增原型模块、FLEM-ProtoNet 主模型、loss 模块、训练器、demo 脚本和包导出均通过语法检查。
新增真实数据集读取、episode sampler、特征投影器、训练脚本和评估脚本均通过语法检查。
```

### 功能冒烟测试

已经对 `build_label_prototypes`、`score_query_by_prototypes`、`FLEMProtoNet`
和 loss 模块执行过张量级冒烟测试。

测试输入：

```text
features = [[1, 0], [3, 0], [0, 4]]
weights  = [[1, 0], [1, 1], [0, 1]]
query    = [[2, 0]]
```

验证结果：

```text
prototypes = [[2.0, 0.0], [1.5, 2.0]]
logits     = [[-0.0, -4.25]]
```

`FLEMProtoNet` 冒烟测试验证内容：

```text
logits:                [num_query, num_labels]
probabilities:         [num_query, num_labels]
prototypes:            [num_labels, num_features]
support_label_weights: [num_support, num_labels]
```

loss 冒烟测试验证内容：

```text
multilabel_prototype_loss 可以返回标量 loss。
flem_protonet_loss 可以返回 total、prediction 和 support_label_enhancement 分项。
FLEMProtoNet 输出 logits 后可以正常 backward。
train_episode 可以完成单个 episode 的参数更新。
evaluate_episode 可以完成单个 episode 的无梯度评估。
```

demo 脚本验证内容：

```text
python scripts\demo_random_episode.py
```

预期输出包括：

```text
demo_random_episode: ok
logits shape: (4, 5)
probabilities shape: (4, 5)
prototypes shape: (5, 64)
eval logits shape: (4, 5)
```

真实 VOC2007 特征链路验证内容：

```text
python -c "from src.data import MultiLabelFeatureDataset, MultiLabelEpisodeSampler; ..."
python -c "from src.models import FLEMProtoNet; ..."
python scripts\evaluate.py --episodes 2 --phase val --device cpu
python scripts\train.py --episodes 2 --eval-every 2 --eval-episodes 2 --checkpoint experiments\checkpoints\smoke.pt --device cpu
python scripts\evaluate.py --checkpoint experiments\checkpoints\voc2007_feature_best.pt --phase val --episodes 5 --device cpu
python scripts\train.py --config configs\voc2007_feature_protonet_ml.json --episodes 2 --eval-every 2 --eval-episodes 2 --checkpoint experiments\checkpoints\protonet_ml_smoke.pt --device cpu
python scripts\evaluate.py --config configs\voc2007_feature_protonet_ml.json --checkpoint experiments\checkpoints\protonet_ml_smoke.pt --phase val --episodes 2 --device cpu
python scripts\train.py --config configs\voc2007_feature.json --episodes 2 --eval-every 2 --eval-episodes 2 --checkpoint experiments\checkpoints\flem_loss_smoke.pt --device cpu
python scripts\evaluate.py --config configs\voc2007_feature.json --checkpoint experiments\checkpoints\flem_loss_smoke.pt --phase val --episodes 2 --device cpu
```

验证结果：

```text
dataset/sampler 可以返回 support_features、support_labels、query_features、query_labels。
FLEMProtoNet 可以直接接收 2048 维预提取特征并输出 [num_query, 20] logits。
evaluate.py 可以在 val split 上完成 episode 评估。
train.py 可以完成训练、验证和 checkpoint 保存。
evaluate.py 可以输出概率范围、预测正例比例和自动阈值扫描结果。
ProtoNet-ML baseline 可以完成训练、验证、checkpoint 保存和 checkpoint 加载评估。
FLEM-style loss 版本可以完成训练、验证、checkpoint 保存和 checkpoint 加载评估。
```

## 重要设计决策

### 使用标签原型，而不是类别原型

普通 ProtoNet 是单标签分类，构造的是类别原型。

本项目是多标签少样本学习，一个样本可以同时属于多个标签，因此应该构造“标签原型”：

$$
p_l = \frac{\sum_i q_{i,l} z_i}{\sum_i q_{i,l} + \varepsilon}
$$

其中：

- `z_i` 是 support 样本特征。
- `q_{i,l}` 是样本 `i` 对标签 `l` 的增强权重。
- `p_l` 是标签 `l` 的原型。

每个样本可以同时参与多个标签原型的构造。

### 最终预测使用 sigmoid，而不是 softmax

多标签任务中，各标签不是互斥关系。因此最终预测应该使用逐标签 `sigmoid`：

$$
\hat{y}_{j,l} = \sigma(-d_{j,l} + b_l)
$$

不要使用 `softmax` 作为最终多标签分类输出。

### query 标签不能在推理阶段使用

`D:\pycharm_Projects\PythonProject\SRTP\模型.png` 中画了 `y_query` 输入 LE。

这个设计在训练阶段可以接受，因为 query 标签已知，可以用于计算损失。

但在测试或推理阶段，真实 `y_query` 是未知的，不能作为模型输入。

因此推理流程应该是：

```text
support 标签 -> 标签增强 -> 标签原型
query 图像 -> query 特征 -> 与标签原型比较 -> 预测标签
```

## 已阅读的参考实现

### FLEM 项目

项目路径：

```text
D:\pycharm_Projects\PythonProject\SRTP\FLEM
```

已经阅读的关键文件：

- `FLEM\methods\flem.py`
- `FLEM\datasets\datasets.py`
- `FLEM\utils\metrics.py`
- `FLEM\run_flem.py`

有价值的部分：

- `LE` 标签增强模块。
- `loss_enhanced` 的思想。
- 多标签评价指标。
- 读取预提取特征和标签的 Dataset 结构。

不建议直接照搬的部分：

- FLEM 中普通线性分类器 `classifier`。
- FLEM 中部分 `CrossEntropyLoss + softmax` 的标签分布对齐逻辑，需要重新判断是否适合多标签少样本任务。

### ProtoNet 项目

项目路径：

```text
D:\pycharm_Projects\PythonProject\SRTP\Prototypical-Networks-for-Few-shot-Learning-PyTorch
```

已经阅读的关键文件：

- `Prototypical-Networks-for-Few-shot-Learning-PyTorch\protonet.py`
- `Prototypical-Networks-for-Few-shot-Learning-PyTorch\prototypical_loss.py`
- `Prototypical-Networks-for-Few-shot-Learning-PyTorch\prototypical_batch_sampler.py`
- `Prototypical-Networks-for-Few-shot-Learning-PyTorch\train.py`

有价值的部分：

- Conv4 特征提取器。
- 欧氏距离函数。
- episodic training 的整体组织方式。

必须重写的部分：

- 单标签 `PrototypicalBatchSampler`。
- 单标签 `prototypical_loss`。
- 最终 `log_softmax` 分类逻辑。

## 尚未实现

以下部分还没有实现：

1. 使用相同 seed、episodes、eval episodes 对比 ProtoNet-ML 和 FLEM-ProtoNet。
2. FLEM 原论文完整 `loss_enhanced` 的进一步适配。
3. 更严格的 episode 采样策略，例如保证 support/query 对每个 episode 标签都有正样本覆盖。
4. 正式实验记录和消融实验表格。
5. 可选：接入原始图像数据和更强图像 encoder。

## 推荐下一步

建议下一步运行一次较完整的 VOC2007 特征实验：

```text
python scripts\train.py --config configs\voc2007_feature.json --device cpu
python scripts\evaluate.py --config configs\voc2007_feature.json --phase test --device cpu
```

如果有可用 GPU，建议改用：

```text
--device cuda
```
