"""Independent research baselines; these classes are not paper reproductions."""
import torch
from torch import nn


def dsnt(logits):
    """Expectation on a [0,1] endpoint grid, output [B,K,2] in x,y order."""
    b, k, h, w = logits.shape
    probabilities = logits.flatten(2).softmax(-1).reshape(b, k, h, w)
    xs = torch.linspace(0, 1, w, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0, 1, h, device=logits.device, dtype=logits.dtype)
    return torch.stack(((probabilities.sum(2)*xs).sum(-1),
                        (probabilities.sum(3)*ys).sum(-1)), dim=-1)


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 8)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                     nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.channel(x)
        pooled = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        return x * self.spatial(pooled)


class MultiResolutionFusion(nn.Module):
    """Learn weights for high/low resolution branches, then channel/spatial attention."""
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(2))
        self.attention = ChannelSpatialAttention(channels)

    def forward(self, high, low):
        low = nn.functional.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        weights = self.weights.softmax(0)
        return self.attention(high * weights[0] + low * weights[1])


class KeypointNet(nn.Module):
    """Small trainable multi-resolution CNN + DSNT baseline for arbitrary K."""
    def __init__(self, num_keypoints, channels=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, channels, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU())
        self.low = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.fusion = MultiResolutionFusion(channels)
        self.head = nn.Conv2d(channels, num_keypoints, 1)

    def forward(self, image):
        high = self.encoder(image)
        return dsnt(self.head(self.fusion(high, self.low(high))))


def masked_coordinate_loss(prediction, target, visible):
    mask = visible.to(prediction.dtype)
    per_point = (prediction-target).square().sum(-1)
    return (per_point * mask).sum() / mask.sum().clamp_min(1)


class MambaTransformerBlock(nn.Module):
    """Optional genuine Mamba + Transformer experiment; requires mamba-ssm/CUDA."""
    def __init__(self, channels=128, heads=4):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.transformer = nn.TransformerEncoderLayer(channels, heads, batch_first=True)

    def forward(self, tokens):
        return self.transformer(tokens + self.mamba(self.norm(tokens)))
