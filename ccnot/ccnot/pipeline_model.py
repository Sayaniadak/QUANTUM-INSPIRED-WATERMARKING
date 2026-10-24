"""CNN + SE + compression + angle encoding, moved out of main_ccnot.py so other
scripts (run_dataset.py) can import it. Same layers, same seed (42), same maths,
so it gives the same 8 values as main_ccnot.py for the same watermark.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEAttention(nn.Module):
    def __init__(self, feature_size=64, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(feature_size, feature_size // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feature_size // reduction, feature_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        weights = self.sigmoid(self.fc2(self.relu(self.fc1(features))))
        return features * weights, weights


class WatermarkFeaturePipeline(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False
        self.projection = nn.Linear(512, 64)
        self.attention = SEAttention(feature_size=64, reduction=16)
        self.compressor = nn.Sequential(nn.Linear(64, 8), nn.Sigmoid())

    def forward(self, image):
        with torch.no_grad():
            feature_maps = self.cnn(image)
            pooled = F.adaptive_avg_pool2d(feature_maps, (1, 1)).flatten(1)
        cnn_features = F.relu(self.projection(pooled))
        attended, attention_weights = self.attention(cnn_features)
        compressed = self.compressor(attended)
        angles = compressed * math.pi
        return cnn_features, attended, attention_weights, compressed, angles


def build_model():
    """Seed first, then build, exactly like main_ccnot.py."""
    torch.manual_seed(42)
    model = WatermarkFeaturePipeline(pretrained=True)
    model.eval()
    return model


def watermark_to_angle_u8(scrambled_128, model):
    """128x128 0/1 scrambled watermark -> eight 8-bit angle values."""
    x = torch.from_numpy(np.asarray(scrambled_128, dtype=np.float32))
    x = x.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        _, _, _, _, angles = model(x)
    return torch.round(angles[0] / math.pi * 255).to(torch.uint8).cpu().numpy()