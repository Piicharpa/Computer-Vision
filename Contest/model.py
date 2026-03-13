import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class FoodComparator(nn.Module):

    def __init__(self):
        super().__init__()

        # backbone CNN
        self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # remove classifier
        self.cnn.classifier = nn.Identity()

        # MobileNet feature size = 1280
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, img1, img2):

        # extract features
        f1 = self.cnn(img1)
        f2 = self.cnn(img2)

        # siamese difference
        x = torch.cat([f1, f2], dim=1)

        # classify
        out = self.classifier(x)

        return out