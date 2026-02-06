import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class PSPNet(nn.Module):
    def __init__(self, n_classes=2, layers=50, bins=(1, 2, 3, 6), dropout=0.1, zoom_factor=8, use_ppm=True, pretrained=False):
        super(PSPNet, self).__init__()
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        
        # Backbone (ResNet)
        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        else:
            resnet = models.resnet101(pretrained=pretrained)
            
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Modify strides for dilated convolution (Output Stride = 8 like standard PSPNet)
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
            fea_dim *= 2

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, n_classes, kernel_size=1)
        )
        
        self.orient_head = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 37, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_ppm:
            x = self.ppm(x)

        road = self.cls(x)
        orient = self.orient_head(x)

        if self.zoom_factor != 1:
            road = F.interpolate(road, size=(h, w), mode='bilinear', align_corners=True)
            orient = F.interpolate(orient, size=(h, w), mode='bilinear', align_corners=True)

        return [road], [orient]
