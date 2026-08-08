# FLEM-ProtoNet: 多标签少样本学习方案

本文档整理一种将 **Prototypical Networks for Few-shot Learning** 与 **Fusion Label Enhancement for Multi-Label Learning** 结合起来的方法，用于多标签少样本学习任务。

## 1. 核心思想

Prototypical Networks 的核心是：在少样本任务中，每个类别由 support set 中该类别样本的特征均值表示，query 样本根据到各类别原型的距离完成分类。

多标签学习的问题是：一个样本可以同时属于多个标签，因此不能直接使用单标签 ProtoNet 的互斥分类方式。需要做三个改造：

1. 将“类别原型”改为“标签原型”。
2. 将 `softmax` 分类改为逐标签 `sigmoid` 预测。
3. 将原始 0/1 多标签监督改为 FLEM 产生的增强标签分布，用软权重构造标签原型。

因此，本方案的基本思想是：

> 用 FLEM 从原始 multi-hot 标签中恢复更细粒度的增强标签分布，再用增强标签作为权重，为每个标签构造原型，并在 episodic meta-learning 框架下完成多标签少样本预测。

## 2. 整体框架

给定一个 episode：

- support set：少量带多标签标注的样本。
- query set：需要预测多标签的样本。
- label set：当前 episode 中涉及的标签集合。

整体流程如下：

```text
support images x_s, labels y_s
        |
        v
feature encoder f_theta
        |
        v
support embeddings z_s
        |
        +--------------------+
        |                    |
        v                    v
label enhancement g_phi   original labels y_s
        |
        v
enhanced labels q_s
        |
        v
label-wise prototypes p_l

query images x_q
        |
        v
query embeddings z_q
        |
        v
distance to each label prototype
        |
        v
sigmoid multi-label prediction
```

## 3. 特征编码

使用编码器将输入样本映射到嵌入空间：

$$
z_i = f_\theta(x_i)
$$

其中：

- $x_i$ 是输入样本。
- $z_i$ 是样本嵌入。
- $f_\theta$ 可以是 CNN、ResNet、Transformer 或已有项目中的特征提取网络。

## 4. 标签增强

原始多标签标注通常是二值形式：

$$
y_{i,l} \in \{0, 1\}
$$

但二值标签只能表示标签是否存在，不能表示标签对样本的重要程度。FLEM 的作用是根据样本特征、标签相关性和原始逻辑标签，恢复连续的增强标签：

$$
q_{i,l} \in [0, 1]
$$

其中：

- `q_i,l` 越大，表示标签 `l` 对样本 `i` 越重要。
- `q_i,l` 不只是监督目标，也参与原型构造。

可以抽象为：

$$
Q = g_\phi(Z, Y)
$$

其中：

- `Z` 是样本特征矩阵。
- `Y` 是原始 multi-hot 标签矩阵。
- `Q` 是增强标签矩阵。
- `g_phi` 是 label enhancement 模块。

## 5. 标签原型构造

传统 ProtoNet 对单个类别取特征均值：

$$
c_k = \frac{1}{|S_k|}\sum_{i \in S_k} z_i,\quad S_k = \{i \mid y_i = k\}
$$

多标签场景中，一个样本可以同时参与多个标签原型的构造。因此对每个标签 `l` 使用增强标签 `q_i,l` 作为权重：

$$
p_l = \frac{\sum_i q_{i,l} z_i}{\sum_i q_{i,l} + \varepsilon}
$$

其中：

- `p_l` 是标签 `l` 的原型。
- `q_i,l` 是样本 `i` 对标签 `l` 的增强权重。
- `eps` 用于避免少样本情况下分母为 0。

这个设计的优势是：标签越能代表该样本，该样本对该标签原型的贡献越大。

## 6. Query 多标签预测

对 query 样本 `x_j`，先得到嵌入：

$$
z_j = f_\theta(x_j)
$$

然后计算它到每个标签原型的距离：

$$
d_{j,l} = \|z_j - p_l\|_2^2
$$

将距离转换为逐标签预测概率：

$$
\hat{y}_{j,l} = \sigma(-d_{j,l} + b_l)
$$

这里必须使用 `sigmoid`，不能使用 `softmax`。原因是多标签任务中多个标签可以同时为真，标签之间不是互斥关系。

## 7. 损失函数设计

总损失可以由三部分组成：

$$
L_{\mathrm{total}} = L_{\mathrm{pred}} + \alpha L_{\mathrm{soft}} + \beta L_{\mathrm{enhance}}
$$

### 7.1 原始多标签预测损失

使用原始 multi-hot 标签监督最终预测：

$$
L_{\mathrm{pred}} = \mathrm{BCE}(\hat{y}, y)
$$

这保证模型仍然学习真实的二值多标签目标。

### 7.2 增强标签软监督损失

使用 FLEM 产生的增强标签监督预测分布：

$$
L_{\mathrm{soft}} = \mathrm{BCE}(\hat{y}, q)
$$

这使模型学习标签强弱关系，而不只是学习标签是否存在。

### 7.3 标签增强损失

FLEM 模块自身需要保留原始标签信息，并利用特征和标签相关性恢复合理的增强标签：

$$
L_{\mathrm{enhance}} = L_{\mathrm{FLEM}}(q, y, z)
$$

实际实现中，可以沿用 FLEM 代码中的优化目标，再和 ProtoNet 的 episodic loss 联合训练。

## 8. 训练流程

推荐使用 episodic meta-learning 训练方式。

每个 episode 可以按如下方式构造：

1. 采样 `N` 个标签作为当前 episode 的标签集合。
2. 对每个标签采样 `K` 个正样本进入 support set。
3. 采样若干 query 样本。
4. 保留所有样本的完整 multi-hot 标签，不要把样本强行转换成单标签类别。
5. 用 FLEM 生成 support/query 的增强标签。
6. 用 support 的增强标签构造标签原型。
7. 用 query 到标签原型的距离做多标签预测。
8. 联合优化预测损失和标签增强损失。

伪代码如下：

```text
for episode in episodes:
    S, Q = sample_multilabel_episode(dataset)

    z_s = f_theta(S.x)
    z_q = f_theta(Q.x)

    q_s = g_phi(z_s, S.y)
    q_q = g_phi(z_q, Q.y)

    for label l in episode_labels:
        p_l = weighted_mean(z_s, weight=q_s[:, l])

    logits = -distance(z_q, prototypes) + bias
    y_hat = sigmoid(logits)

    loss = BCE(y_hat, Q.y) + alpha * BCE(y_hat, q_q) + beta * FLEM_loss
    update(theta, phi)
```

## 9. 可加入的标签相关性建模

多标签任务中，标签之间常有共现关系。例如：

- `person` 和 `bicycle` 常共同出现。
- `car` 和 `road` 常共同出现。
- 某些标签之间可能互斥或弱相关。

可以构造标签相关性矩阵：

$$
A_{l,m} = \mathrm{co\_occurrence}(l, m)
$$

然后在标签原型上做图传播：

$$
P' = \mathrm{GCN}(P, A)
$$

最终用传播后的原型 `P_prime` 做 query 预测。这样可以让稀疏标签从相关标签中获得补充信息。

## 10. 推荐消融实验

建议至少比较以下模型：

1. `ProtoNet-ML`：只做多标签 ProtoNet，不使用 FLEM。
2. `ProtoNet + offline LE`：先离线生成增强标签，再训练多标签 ProtoNet。
3. `FLEM-ProtoNet joint`：联合训练 FLEM 和多标签 ProtoNet。
4. `FLEM-ProtoNet + label graph`：在联合模型上增加标签图传播。

推荐评估指标：

- Micro-F1
- Macro-F1
- mAP
- Hamming Loss
- Ranking Loss
- Coverage
- One-error

推荐少样本设置：

```text
5-label 1-shot
5-label 5-shot
10-label 1-shot
10-label 5-shot
```

## 11. 可能的实现路线

可以基于当前目录中的两个已有项目做整合：

1. 从 `Prototypical-Networks-for-Few-shot-Learning-PyTorch` 中复用 episodic training、encoder 和 prototype 距离计算逻辑。
2. 从 `FLEM` 中复用 label enhancement 模块和对应损失。
3. 新增一个多标签 episode sampler，保证 support/query 保留完整 multi-hot 标签。
4. 将 prototype 构造函数从按类别均值改为按增强标签加权均值。
5. 将分类头从 `softmax` 改为逐标签 `sigmoid`。
6. 将 loss 从单标签交叉熵改为多标签 BCE 与 FLEM loss 的联合目标。

## 12. 方法名称

可以将该方法命名为：

```text
FLEM-ProtoNet
```

或者更完整地命名为：

```text
Label-Enhanced Prototypical Network for Few-Shot Multi-Label Learning
```

核心贡献可以表述为：

> 本方法将原型网络扩展到多标签少样本学习场景，利用标签增强得到的连续标签分布对 support 样本进行加权，从而构造标签级原型，并通过联合优化标签增强目标和原型预测目标，实现端到端的多标签少样本学习。
