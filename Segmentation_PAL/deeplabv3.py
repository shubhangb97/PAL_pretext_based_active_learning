from collections import OrderedDict
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import mobilenet_v2
from torch.nn import functional as F
import torch


class DeepLab(nn.Module):

    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        # define model
        self.model = deeplabv3_resnet50(num_classes = num_classes, progress = True)

    def forward(self, x):
        x = self.model(x)
        return x


class DeepLabScoring(nn.Module):

    def __init__(self, num_classes):
        super(DeepLabScoring, self).__init__()
        # define model
        self.model = deeplabv3_resnet50(num_classes = num_classes, progress = True)


    def forward(self, x,y1):
        x1 = self.model(x)
        x2= self.model.backbone(y1)
        return x1,x2

class DeepLabMobile(nn.Module):

    def __init__(self, num_classes):
        super(DeepLabMobile, self).__init__()
        # define model
        self.model = deeplabv3_resnet50(num_classes = num_classes, progress = True)
        self.MobileNet=mobilenet_v2(pretrained=True)
        self.model.backbone=self.MobileNet.features
        self.model.classifier=DeepLabHead(in_channels=1280,num_classes=num_classes)


    def forward(self, x):
        input_shape = x.shape[-2:]
        result = OrderedDict()
        x1=self.model.backbone(x)
        #x1=x1['out']
        x1=self.model.classifier(x1)
        x1 = F.interpolate(x1, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x1
        #x = self.model(x)
        return result

class DeepLabScoringMobile(nn.Module):

    def __init__(self, num_classes):
        super(DeepLabScoringMobile, self).__init__()
        # define model
        self.model = deeplabv3_resnet50(num_classes = num_classes, progress = True)
        self.MobileNet=mobilenet_v2(pretrained=True)
        self.model.backbone=self.MobileNet.features
        #self.classifier=self.model.classifier
        self.model.classifier=DeepLabHead(in_channels=1280,num_classes=num_classes)


    def forward(self, x,y1):
        input_shape = x.shape[-2:]
        result = OrderedDict()
        result2 = OrderedDict()
        x1=self.model.backbone(x)
        #x1=x1['out']
        x1=self.model.classifier(x1)
        x1 = F.interpolate(x1, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x1
        x2=self.model.backbone(y1)
        result2["out"]=x2
        #x = self.model(x)
        return result, result2



class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
