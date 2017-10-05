import torch.nn as nn
import torchvision.models as models
import math
import torch
from .combinedmodel import CombinedModel
from .featuresmodel import FeaturesModel
from .imagemodel import ImageModel
from .postermodel import PosterModel
from torch.autograd import Variable

class StackingLayer(nn.Module):
    def __init__(self):
        super(StackingLayer, self).__init__()
        self.data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.layer = nn.Parameter(torch.ones(4).type(self.data_type))

    def forward(self, x):
        return torch.bmm(self.layer.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, len(self.layer)), x)

class StackingModel(nn.Module):
    def __init__(self, num_classes=2):
        super(StackingModel, self).__init__()
        self.data_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.classifier = nn.Sequential(
            StackingLayer()
        )
        self.model1 = self._get_model(CombinedModel, "model1.pt")
        self.model2 = self._get_model(FeaturesModel, "model2.pt")
        self.model3 = self._get_model(ImageModel, "model3.pt")
        self.model4 = self._get_model(PosterModel, "model4.pt")
        #self.model5 = self._get_model(FeaturesModel, "model2-3.pt")
        #self.model6 = self._get_model(ImageModel, "model3-1.pt")
        #self.model7 = self._get_model(ImageModel, "model3-2.pt")
        #self.model8 = self._get_model(PosterModel, "model4-1.pt")
        #self.model9 = self._get_model(PosterModel, "model4-2.pt")

    def forward(self, x, x1):
        scores1 = nn.Softmax()(self.model1(x, x1)).unsqueeze(1)
        scores2 = nn.Softmax()(self.model2(x, x1)).unsqueeze(1)
        scores3 = nn.Softmax()(self.model3(x, x1)).unsqueeze(1)
        scores4 = nn.Softmax()(self.model4(x, x1)).unsqueeze(1)
        #scores5 = nn.Softmax()(self.model5(x, x1)).unsqueeze(1)
        #scores6 = nn.Softmax()(self.model6(x, x1)).unsqueeze(1)
        #scores7 = nn.Softmax()(self.model7(x, x1)).unsqueeze(1)
        #scores8 = nn.Softmax()(self.model8(x, x1)).unsqueeze(1)
        #scores9 = nn.Softmax()(self.model9(x, x1)).unsqueeze(1)

        scores = torch.cat((scores1, scores2, scores3, scores4), 1)
        y = self.classifier(scores)
        return y.view(y.size(0), -1)

    def _get_model(self, Model, file_name):
        model = Model()
        model = model.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
        try:
            model.load_state_dict(torch.load(file_name))
        except:
            pass
        model.eval()
        self._require_grad_false(model)
        return model

    def _require_grad_false(self, model):
        for p in model.parameters():
            p.requires_grad = False
