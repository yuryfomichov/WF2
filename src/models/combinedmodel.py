import torch.nn as nn
import torchvision.models as models
import math
import torch


class CombinedModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CombinedModel, self).__init__()
        # vgg = models.vgg16(pretrained=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self._require_grad_false()

        self.classifier = nn.Sequential(
            nn.Linear(192 * 10 * 10, 2024),
            nn.BatchNorm1d(2024),
            nn.ReLU(True),
            nn.Linear(2024, 2024),
            nn.BatchNorm1d(2024),
            nn.ReLU(True),
            nn.Linear(2024, num_classes),
        )

        self.secondNet = nn.Sequential(
            nn.Linear(25, 2024),
            nn.BatchNorm1d(2024),
            nn.ReLU(True),
            nn.Linear(2024, 2024),
            nn.BatchNorm1d(2024),
            nn.ReLU(True),
            nn.Linear(2024, 2024),
            nn.BatchNorm1d(2024),
            nn.ReLU(True),
            nn.Linear(2024, 2024),
            nn.BatchNorm1d(2024),
            nn.ReLU(True),
            nn.Linear(2024, num_classes),
        )

        self._initialize_weights()

    def forward(self, x, x1):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x1 = self.secondNet(x1)

        y = x * 0.2 + x1 * 0.8

        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _require_grad_false(self):
        for p in self.features.parameters():
            p.requires_grad = False
