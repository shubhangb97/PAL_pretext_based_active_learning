import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Has definitions of scoring networks used

class RotNetMulti(nn.Module):
    def __init__(self,num_classes,num_rotations):
        super(RotNetMulti,self).__init__()
        self.resnet1=models.resnet18(pretrained=False);
        self.resnet1.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        num_fea=self.resnet1.fc.in_features
        self.layer1=nn.Linear(num_fea,num_rotations)
        self.layer2=nn.Linear(num_fea,num_classes)
        self.features=nn.Sequential(*list(self.resnet1.children())[:-1])

    def forward(self,x):
        p1=self.features(x)
        p1=p1.squeeze()
        x=self.layer1(p1)
        x2=self.layer2(p1)
        return x,x2

class RotNetMultiPretrained(nn.Module):
    def __init__(self,num_classes,num_rotations):
        super(RotNetMultiPretrained,self).__init__()
        self.resnet1=models.resnet18(pretrained=True);
        self.resnet1.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        num_fea=self.resnet1.fc.in_features
        self.layer1=nn.Linear(num_fea,num_rotations)
        self.layer2=nn.Linear(num_fea,num_classes)
        self.features=nn.Sequential(*list(self.resnet1.children())[:-1])

    def forward(self,x):
        p1=self.features(x)
        p1=p1.squeeze()
        x=self.layer1(p1)
        x2=self.layer2(p1)
        return x,x2
