import torch.nn as nn
import torchvision.models as models
import math
import torch
from .combinedmodel import CombinedModel
from .featuresmodel import FeaturesModel
from .imagemodel import ImageModel
from .postermodel import PosterModel


class StackingModel(nn.Module):
    def __init__(self, num_classes=2):
        super(StackingModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(18, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes)
        )
        self.model1 = self._get_model(CombinedModel, "model1-1.pt")
        self.model2 = self._get_model(CombinedModel, "model1-2.pt")
        self.model3 = self._get_model(FeaturesModel, "model2-1.pt")
        self.model4 = self._get_model(FeaturesModel, "model2-2.pt")
        self.model5 = self._get_model(FeaturesModel, "model2-3.pt")
        self.model6 = self._get_model(ImageModel, "model3-1.pt")
        self.model7 = self._get_model(ImageModel, "model3-2.pt")
        self.model8 = self._get_model(PosterModel, "model4-1.pt")
        self.model9 = self._get_model(PosterModel, "model4-2.pt")
        self._initialize_weights()

    def forward(self, x, x1):
        scores1 = self.model1(x, x1)
        scores2 = self.model2(x, x1)
        scores3 = self.model3(x, x1)
        scores4 = self.model4(x, x1)
        scores5 = self.model5(x, x1)
        scores6 = self.model6(x, x1)
        scores7 = self.model7(x, x1)
        scores8 = self.model8(x, x1)
        scores9 = self.model9(x, x1)

        scores = torch.cat((scores1, scores2, scores3, scores4, scores5, scores6, scores7, scores8, scores9), 1)
        y = self.classifier(scores)
        return y

    def _get_model(self, Model, file_name):
        model = Model()
        model = model.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
        try:
            model.load_state_dict(torch.load(file_name))
        except:
            pass
        model.eval()
        return model

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
