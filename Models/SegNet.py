import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SegNet(nn.Module):
    def __init__(self, n_classes=2, in_channels=3):
        super(SegNet, self).__init__()

        # Load VGG16 bn weights
        vgg = models.vgg16_bn(pretrained=False)
        features = list(vgg.features.children())

        # Encoder
        self.enc1 = nn.Sequential(*features[0:6])
        self.enc2 = nn.Sequential(*features[7:13])
        self.enc3 = nn.Sequential(*features[14:23])
        self.enc4 = nn.Sequential(*features[24:33])
        self.enc5 = nn.Sequential(*features[34:43])

        # Maxpool with indices
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        # Decoder
        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.dec5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        )
        
        # Orientation Head (Simple adapter)
        self.orient_head = nn.Conv2d(64, 37, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1, i1 = self.pool(x1)
        
        x2 = self.enc2(p1)
        p2, i2 = self.pool(x2)
        
        x3 = self.enc3(p2)
        p3, i3 = self.pool(x3)
        
        x4 = self.enc4(p3)
        p4, i4 = self.pool(x4)
        
        x5 = self.enc5(p4)
        p5, i5 = self.pool(x5)

        # Decoder
        d5 = self.unpool(p5, i5)
        d5 = self.dec5(d5)

        d4 = self.unpool(d5, i4)
        d4 = self.dec4(d4)

        d3 = self.unpool(d4, i3)
        d3 = self.dec3(d3)

        d2 = self.unpool(d3, i2)
        d2 = self.dec2(d2)

        d1 = self.unpool(d2, i1)
        # Separate heads
        road = self.dec1(d1)
        orient = self.orient_head(d1)

        return [road], [orient]
