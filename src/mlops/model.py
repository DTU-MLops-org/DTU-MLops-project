from torch import nn
import torch
import torchvision

# Define model
# Load dataset
from torchvision.models import MobileNet_V3_Small_Weights
weights = MobileNet_V3_Small_Weights.DEFAULT
preset = weights.transforms() # ImageNet preset with mean/std etc.
mean, std = preset.mean, preset.std

from torchvision.models import mobilenet_v3_small
model = mobilenet_v3_small(weights=weights)
for p in model.features.parameters():
    p.requires_grad = False # freeze features first

# change just the last layer of default classifier
in_f: int = model.classifier[-1].in_features  # type: ignore
model.classifier[-1] = nn.Linear(in_f, 17) # 4 suits and 13 numbers


# if __name__ == "__main__":
#     x = torch.rand(1)
#     print(f"Output shape of model: {model(x).shape}")
