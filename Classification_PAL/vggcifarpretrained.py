import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class vgg16_pretrained(nn.Module):
    def __init__(self,num_classes):
        super(vgg16_pretrained,self).__init__()
        self.vgg16_1=models.vgg16(pretrained=True);
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self,x):
        p1=self.vgg16_1.features(x)
        p1=self.avgpool(p1)
        p1=torch.flatten(p1,1)
        p1=p1.squeeze()
        x=self.classifier(p1)
        return x
