# 消融实验结果

种子：42, 123, 2024；backbone：resnet50, resnet101；设备：cpu。
各消融仅改变表中对应字段，其余参数来自同一 OurModel 配置。mAP 为逐 episode 平均；± 为跨随机种子的样本标准差。

## enhanced vs positive_gated

| 数据集 | Backbone | enhanced mAP | positive_gated mAP | positive_gated - enhanced |
|---|---|---:|---:|---:|
| voc2007 | resnet50 | 0.882656 ± 0.009909 | 0.892239 ± 0.007225 | +0.009583 |
| voc2007 | resnet101 | 0.884289 ± 0.003883 | 0.894664 ± 0.006062 | +0.010375 |
| coco17 | resnet50 | 0.750602 ± 0.002100 | 0.765780 ± 0.013376 | +0.015179 |
| coco17 | resnet101 | 0.753776 ± 0.002042 | 0.773517 ± 0.004935 | +0.019741 |
| nuswide | resnet50 | 0.724461 ± 0.020820 | 0.742027 ± 0.015759 | +0.017566 |
| nuswide | resnet101 | 0.716104 ± 0.021511 | 0.743008 ± 0.018609 | +0.026903 |

## flem vs bce

| 数据集 | Backbone | flem mAP | bce mAP | bce - flem |
|---|---|---:|---:|---:|
| voc2007 | resnet50 | 0.890878 ± 0.003232 | 0.892239 ± 0.007225 | +0.001361 |
| voc2007 | resnet101 | 0.894254 ± 0.008798 | 0.894664 ± 0.006062 | +0.000410 |
| coco17 | resnet50 | 0.769835 ± 0.005846 | 0.765780 ± 0.013376 | -0.004054 |
| coco17 | resnet101 | 0.765983 ± 0.010134 | 0.773517 ± 0.004935 | +0.007534 |
| nuswide | resnet50 | 0.737447 ± 0.017207 | 0.742027 ± 0.015759 | +0.004580 |
| nuswide | resnet101 | 0.737313 ± 0.009100 | 0.743008 ± 0.018609 | +0.005694 |

## cosine vs euclidean

| 数据集 | Backbone | cosine mAP | euclidean mAP | euclidean - cosine |
|---|---|---:|---:|---:|
| voc2007 | resnet50 | 0.848295 ± 0.001586 | 0.892239 ± 0.007225 | +0.043944 |
| voc2007 | resnet101 | 0.865448 ± 0.009645 | 0.894664 ± 0.006062 | +0.029216 |
| coco17 | resnet50 | 0.714014 ± 0.010295 | 0.765780 ± 0.013376 | +0.051766 |
| coco17 | resnet101 | 0.717411 ± 0.011007 | 0.773517 ± 0.004935 | +0.056106 |
| nuswide | resnet50 | 0.708623 ± 0.012947 | 0.742027 ± 0.015759 | +0.033404 |
| nuswide | resnet101 | 0.699548 ± 0.007923 | 0.743008 ± 0.018609 | +0.043460 |

## 单次运行

| 数据集 | 消融项 | 设置 | Backbone | Seed | mAP |
|---|---|---|---|---:|---:|
| coco17 | label_weight_mode | enhanced | resnet101 | 42 | 0.751420 |
| coco17 | label_weight_mode | enhanced | resnet101 | 123 | 0.754869 |
| coco17 | label_weight_mode | enhanced | resnet101 | 2024 | 0.755038 |
| coco17 | label_weight_mode | positive_gated | resnet101 | 42 | 0.777268 |
| coco17 | label_weight_mode | positive_gated | resnet101 | 123 | 0.775357 |
| coco17 | label_weight_mode | positive_gated | resnet101 | 2024 | 0.767926 |
| coco17 | label_weight_mode | enhanced | resnet50 | 42 | 0.750641 |
| coco17 | label_weight_mode | enhanced | resnet50 | 123 | 0.752682 |
| coco17 | label_weight_mode | enhanced | resnet50 | 2024 | 0.748482 |
| coco17 | label_weight_mode | positive_gated | resnet50 | 42 | 0.760935 |
| coco17 | label_weight_mode | positive_gated | resnet50 | 123 | 0.755502 |
| coco17 | label_weight_mode | positive_gated | resnet50 | 2024 | 0.780904 |
| nuswide | label_weight_mode | enhanced | resnet101 | 42 | 0.739284 |
| nuswide | label_weight_mode | enhanced | resnet101 | 123 | 0.712244 |
| nuswide | label_weight_mode | enhanced | resnet101 | 2024 | 0.696785 |
| nuswide | label_weight_mode | positive_gated | resnet101 | 42 | 0.763400 |
| nuswide | label_weight_mode | positive_gated | resnet101 | 123 | 0.726944 |
| nuswide | label_weight_mode | positive_gated | resnet101 | 2024 | 0.738679 |
| nuswide | label_weight_mode | enhanced | resnet50 | 42 | 0.743824 |
| nuswide | label_weight_mode | enhanced | resnet50 | 123 | 0.727120 |
| nuswide | label_weight_mode | enhanced | resnet50 | 2024 | 0.702440 |
| nuswide | label_weight_mode | positive_gated | resnet50 | 42 | 0.760154 |
| nuswide | label_weight_mode | positive_gated | resnet50 | 123 | 0.731575 |
| nuswide | label_weight_mode | positive_gated | resnet50 | 2024 | 0.734353 |
| voc2007 | label_weight_mode | enhanced | resnet101 | 42 | 0.882678 |
| voc2007 | label_weight_mode | enhanced | resnet101 | 123 | 0.881470 |
| voc2007 | label_weight_mode | enhanced | resnet101 | 2024 | 0.888718 |
| voc2007 | label_weight_mode | positive_gated | resnet101 | 42 | 0.889977 |
| voc2007 | label_weight_mode | positive_gated | resnet101 | 123 | 0.901510 |
| voc2007 | label_weight_mode | positive_gated | resnet101 | 2024 | 0.892504 |
| voc2007 | label_weight_mode | enhanced | resnet50 | 42 | 0.893831 |
| voc2007 | label_weight_mode | enhanced | resnet50 | 123 | 0.874940 |
| voc2007 | label_weight_mode | enhanced | resnet50 | 2024 | 0.879198 |
| voc2007 | label_weight_mode | positive_gated | resnet50 | 42 | 0.883915 |
| voc2007 | label_weight_mode | positive_gated | resnet50 | 123 | 0.896892 |
| voc2007 | label_weight_mode | positive_gated | resnet50 | 2024 | 0.895910 |
| coco17 | metric | cosine | resnet101 | 42 | 0.708932 |
| coco17 | metric | cosine | resnet101 | 123 | 0.713452 |
| coco17 | metric | cosine | resnet101 | 2024 | 0.729850 |
| coco17 | metric | euclidean | resnet101 | 42 | 0.777268 |
| coco17 | metric | euclidean | resnet101 | 123 | 0.775357 |
| coco17 | metric | euclidean | resnet101 | 2024 | 0.767926 |
| coco17 | metric | cosine | resnet50 | 42 | 0.702130 |
| coco17 | metric | cosine | resnet50 | 123 | 0.719682 |
| coco17 | metric | cosine | resnet50 | 2024 | 0.720230 |
| coco17 | metric | euclidean | resnet50 | 42 | 0.760935 |
| coco17 | metric | euclidean | resnet50 | 123 | 0.755502 |
| coco17 | metric | euclidean | resnet50 | 2024 | 0.780904 |
| nuswide | metric | cosine | resnet101 | 42 | 0.703603 |
| nuswide | metric | cosine | resnet101 | 123 | 0.704623 |
| nuswide | metric | cosine | resnet101 | 2024 | 0.690418 |
| nuswide | metric | euclidean | resnet101 | 42 | 0.763400 |
| nuswide | metric | euclidean | resnet101 | 123 | 0.726944 |
| nuswide | metric | euclidean | resnet101 | 2024 | 0.738679 |
| nuswide | metric | cosine | resnet50 | 42 | 0.705457 |
| nuswide | metric | cosine | resnet50 | 123 | 0.722859 |
| nuswide | metric | cosine | resnet50 | 2024 | 0.697553 |
| nuswide | metric | euclidean | resnet50 | 42 | 0.760154 |
| nuswide | metric | euclidean | resnet50 | 123 | 0.731575 |
| nuswide | metric | euclidean | resnet50 | 2024 | 0.734353 |
| voc2007 | metric | cosine | resnet101 | 42 | 0.855962 |
| voc2007 | metric | cosine | resnet101 | 123 | 0.865138 |
| voc2007 | metric | cosine | resnet101 | 2024 | 0.875244 |
| voc2007 | metric | euclidean | resnet101 | 42 | 0.889977 |
| voc2007 | metric | euclidean | resnet101 | 123 | 0.901510 |
| voc2007 | metric | euclidean | resnet101 | 2024 | 0.892504 |
| voc2007 | metric | cosine | resnet50 | 42 | 0.848340 |
| voc2007 | metric | cosine | resnet50 | 123 | 0.849858 |
| voc2007 | metric | cosine | resnet50 | 2024 | 0.846687 |
| voc2007 | metric | euclidean | resnet50 | 42 | 0.883915 |
| voc2007 | metric | euclidean | resnet50 | 123 | 0.896892 |
| voc2007 | metric | euclidean | resnet50 | 2024 | 0.895910 |
| coco17 | support_loss_type | bce | resnet101 | 42 | 0.777268 |
| coco17 | support_loss_type | bce | resnet101 | 123 | 0.775357 |
| coco17 | support_loss_type | bce | resnet101 | 2024 | 0.767926 |
| coco17 | support_loss_type | flem | resnet101 | 42 | 0.759010 |
| coco17 | support_loss_type | flem | resnet101 | 123 | 0.761330 |
| coco17 | support_loss_type | flem | resnet101 | 2024 | 0.777608 |
| coco17 | support_loss_type | bce | resnet50 | 42 | 0.760935 |
| coco17 | support_loss_type | bce | resnet50 | 123 | 0.755502 |
| coco17 | support_loss_type | bce | resnet50 | 2024 | 0.780904 |
| coco17 | support_loss_type | flem | resnet50 | 42 | 0.763474 |
| coco17 | support_loss_type | flem | resnet50 | 123 | 0.771057 |
| coco17 | support_loss_type | flem | resnet50 | 2024 | 0.774973 |
| nuswide | support_loss_type | bce | resnet101 | 42 | 0.763400 |
| nuswide | support_loss_type | bce | resnet101 | 123 | 0.726944 |
| nuswide | support_loss_type | bce | resnet101 | 2024 | 0.738679 |
| nuswide | support_loss_type | flem | resnet101 | 42 | 0.747785 |
| nuswide | support_loss_type | flem | resnet101 | 123 | 0.731318 |
| nuswide | support_loss_type | flem | resnet101 | 2024 | 0.732837 |
| nuswide | support_loss_type | bce | resnet50 | 42 | 0.760154 |
| nuswide | support_loss_type | bce | resnet50 | 123 | 0.731575 |
| nuswide | support_loss_type | bce | resnet50 | 2024 | 0.734353 |
| nuswide | support_loss_type | flem | resnet50 | 42 | 0.757291 |
| nuswide | support_loss_type | flem | resnet50 | 123 | 0.726649 |
| nuswide | support_loss_type | flem | resnet50 | 2024 | 0.728402 |
| voc2007 | support_loss_type | bce | resnet101 | 42 | 0.889977 |
| voc2007 | support_loss_type | bce | resnet101 | 123 | 0.901510 |
| voc2007 | support_loss_type | bce | resnet101 | 2024 | 0.892504 |
| voc2007 | support_loss_type | flem | resnet101 | 42 | 0.897697 |
| voc2007 | support_loss_type | flem | resnet101 | 123 | 0.900809 |
| voc2007 | support_loss_type | flem | resnet101 | 2024 | 0.884255 |
| voc2007 | support_loss_type | bce | resnet50 | 42 | 0.883915 |
| voc2007 | support_loss_type | bce | resnet50 | 123 | 0.896892 |
| voc2007 | support_loss_type | bce | resnet50 | 2024 | 0.895910 |
| voc2007 | support_loss_type | flem | resnet50 | 42 | 0.888780 |
| voc2007 | support_loss_type | flem | resnet50 | 123 | 0.894600 |
| voc2007 | support_loss_type | flem | resnet50 | 2024 | 0.889255 |
