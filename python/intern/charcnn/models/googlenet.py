'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1),
                                nn.BatchNorm2d(n1x1),
                                nn.ReLU(True))

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1),
                                nn.BatchNorm2d(n3x3red),
                                nn.ReLU(True),
                                nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
                                nn.BatchNorm2d(n3x3),
                                nn.ReLU(True))

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1),
                                nn.BatchNorm2d(n5x5red),
                                nn.ReLU(True),
                                nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
                                nn.BatchNorm2d(n5x5),
                                nn.ReLU(True))

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                nn.Conv2d(in_planes, pool_planes, kernel_size=1),
                                nn.BatchNorm2d(pool_planes),
                                nn.ReLU(True))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GoogLeNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2),
                                        nn.ReLU(True),
                                        nn.MaxPool2d(3, stride=2),
                                        LRN(local_size=5, alpha=0.0001, beta=0.75),
                                        nn.Conv2d(64, 64, kernel_size=1, stride=1),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(True),
                                        LRN(local_size=5, alpha=0.0001, beta=0.75),
                                        nn.MaxPool2d(3, stride=2))

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2)

        self.a4 = Inception(480, 160, 112, 224, 24, 64, 64)
        self.b4 = Inception(512, 256, 160, 320, 32, 128, 128)
        self.avgpool = nn.AvgPool2d(5, stride=3)
        self.conv = nn.Conv2d(832, 128, kernel_size=1)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Sequential(nn.Linear(128, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(1024, num_classes))

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.avgpool(out)
        out = self.relu(self.conv(out))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


    def get_class_name(self):
        return self.__class__.__name__

