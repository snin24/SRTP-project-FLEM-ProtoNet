from torch import nn
import torch


class LabelEnhancer(nn.Module):
    """FLEM 标签增强模块，针对少样本场景优化：

    - 用 LayerNorm 替代 BatchNorm1d，避免小 batch（5~8样本）下统计不稳定。
    - 默认 hidden_dim=64，减少参数量防止过拟合。
    """

    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()
        self.feature_encoder_1 = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.feature_encoder_2 = nn.Linear(hidden_dim, hidden_dim)
        self.label_encoder_1 = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.label_encoder_2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, num_classes),
            nn.LeakyReLU(),
            nn.LayerNorm(num_classes),
        )
        self.decoder_2 = nn.Linear(num_classes, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if "weight" in name and len(parameter.shape) == 2:
                nn.init.kaiming_normal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, features, labels):
        feature_hidden = self.feature_encoder_1(features)
        feature_hidden = feature_hidden + self.feature_encoder_2(feature_hidden)
        label_hidden = self.label_encoder_1(labels)
        label_hidden = label_hidden + self.label_encoder_2(label_hidden)
        fused = torch.cat([feature_hidden, label_hidden], dim=-1)
        enhanced = self.decoder_1(fused)
        return enhanced + self.decoder_2(enhanced)

    def predict_distribution(self, features, labels):
        return torch.sigmoid(self.forward(features, labels))
