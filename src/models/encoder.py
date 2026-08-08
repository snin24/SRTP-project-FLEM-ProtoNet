from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class Conv4Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, hidden_dim),
            conv_block(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        features = self.encoder(inputs)
        return features.view(features.size(0), -1)
