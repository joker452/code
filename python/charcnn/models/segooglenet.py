import torch
import torch.nn as nn


class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def conv_wrapper(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.PReLU())


class Inception3(nn.Module):

    def __init__(self, in_channels, out1x1, out3x3, out3x3double, outpool, outfc1, outfc2):
        super(Inception3, self).__init__()
        self.out_channels = out1x1 + out3x3 + out3x3double + outpool
        self.b1 = conv_wrapper(in_channels, out1x1, 1)
        self.b2 = nn.Sequential(conv_wrapper(in_channels, out1x1, 1),
                                conv_wrapper(out1x1, out3x3double, 3, 1, 1))
        self.b3 = nn.Sequential(conv_wrapper(in_channels, out1x1, 1),
                                conv_wrapper(out1x1, out3x3, 3, 1, 1),
                                conv_wrapper(out3x3, out3x3, 3, 1, 1))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, 1, 1),
                                conv_wrapper(in_channels, outpool, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(nn.Linear(self.out_channels, outfc1, bias=False),
                                nn.Linear(outfc1, outfc2, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        #print(y1.size(), y2.size(), y3.size(), y4.size())
        y = torch.cat([y1, y2, y3, y4], 1)
        scale_factor = self.SE(self.pool(y).squeeze_())[:, :, None, None]
        #print(y.size(), scale_factor.size())
        return y * scale_factor


class Inception3c(nn.Module):

    def __init__(self, in_channels, outfc1, outfc2):
        super(Inception3c, self).__init__()
        self.out_channels = in_channels + 160 + 96
        self.b1 = nn.Sequential(conv_wrapper(in_channels, 128, 1),
                                conv_wrapper(128, 160, 3, 2, 1))
        self.b2 = nn.Sequential(conv_wrapper(in_channels, 64, 1),
                                conv_wrapper(64, 96, 3, 1, 1),
                                conv_wrapper(96, 96, 3, 2, 1))
        self.b3 = nn.MaxPool2d(3, 2, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(nn.Linear(self.out_channels, outfc1, bias=False),
                                nn.Linear(outfc1, outfc2, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        #print(y1.size(), y2.size(), y3.size())
        y = torch.cat([y1, y2, y3], 1)
        scale_factor = self.SE(self.pool(y).squeeze_())[:, :, None, None]
        #print(y.size(), scale_factor.size())
        return y * scale_factor


class Inception4(nn.Module):

    def __init__(self, in_channels, out1x1, out1x1double, out1x1reduce, out3x3, out3x3double, outpool, outfc1, outfc2):
        super(Inception4, self).__init__()
        self.out_channels = out1x1 + out3x3 + out3x3double + outpool
        self.b1 = conv_wrapper(in_channels, out1x1, 1)
        self.b2 = nn.Sequential(conv_wrapper(in_channels, out1x1double, 1),
                                conv_wrapper(out1x1double, out3x3double, 3, 1, 1))
        self.b3 = nn.Sequential(conv_wrapper(in_channels, out1x1reduce, 1),
                                conv_wrapper(out1x1reduce, out3x3, 3, 1, 1),
                                conv_wrapper(out3x3, out3x3, 3, 1, 1))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, 1, 1),
                                conv_wrapper(in_channels, outpool, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(nn.Linear(self.out_channels, outfc1, bias=False),
                                nn.Linear(outfc1, outfc2, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        #print(y1.size(), y2.size(), y3.size(), y4.size())
        y = torch.cat([y1, y2, y3, y4], 1)
        scale_factor = self.SE(self.pool(y).squeeze_())[:, :, None, None]
        #print(y.size(), scale_factor.size())
        return y * scale_factor


class Inception4e(nn.Module):

    def __init__(self, in_channels, outfc1, outfc2):
        super(Inception4e, self).__init__()
        self.out_channels = in_channels + 192 + 256
        self.b1 = nn.Sequential(conv_wrapper(in_channels, 128, 1),
                                conv_wrapper(128, 192, 3, 2, 1))
        self.b2 = nn.Sequential(conv_wrapper(in_channels, 192, 1),
                                conv_wrapper(192, 256, 3, 1, 1),
                                conv_wrapper(256, 256, 3, 2, 1))
        self.b3 = nn.MaxPool2d(3, 2, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(nn.Linear(self.out_channels, outfc1, bias=False),
                                nn.Linear(outfc1, outfc2, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        #print(y1.size(), y2.size(), y3.size())
        y = torch.cat([y1, y2, y3], 1)
        scale_factor = self.SE(self.pool(y).squeeze_())[:, :, None, None]
        #print(y.size(), scale_factor.size())
        return y * scale_factor


class Inception5b(nn.Module):

    def __init__(self, in_channels, out1x1, out1x1double, out1x1reduce, out3x3, out3x3double, outpool):
        super(Inception5b, self).__init__()
        self.out_channels = out1x1 + out3x3 + out3x3double + outpool
        self.b1 = conv_wrapper(in_channels, out1x1, 1)
        self.b2 = nn.Sequential(conv_wrapper(in_channels, out1x1double, 1),
                                conv_wrapper(out1x1double, out3x3double, 3, 1, 1))
        self.b3 = nn.Sequential(conv_wrapper(in_channels, out1x1reduce, 1),
                                conv_wrapper(out1x1reduce, out3x3, 3, 1, 1),
                                conv_wrapper(out3x3, out3x3, 3, 1, 1))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1),
                                conv_wrapper(in_channels, outpool, 1))
        self.pool = nn.Sequential(nn.AvgPool2d(7, 1),
                                  nn.Dropout(0.4))


    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        #print(y1.size(), y2.size(), y3.size(), y4.size())
        y = self.pool(torch.cat([y1, y2, y3, y4], 1))
        feature_vector = y.view(y.size()[0], -1)
        #print(feature_vector.size(), y.size())
        return feature_vector / torch.norm(feature_vector, 2, 1, True)



class SEGoogleNet(nn.Module):

    def __init__(self, num_classes):
        super(SEGoogleNet, self).__init__()
        self.pre_layers = nn.Sequential(conv_wrapper(1, 64, 5, 2, 2),
                                        nn.MaxPool2d(2, 2),
                                        conv_wrapper(64, 64, 1),
                                        conv_wrapper(64, 192, 3, 1, 1),
                                        nn.MaxPool2d(3, 2))
        self.inception3a = Inception3(192, 64, 64, 96, 32, 16, 256)
        self.inception3b = Inception3(256, 64, 96, 96, 64, 20, 320)
        self.inception3c = Inception3c(320, 36, 576)
        self.inception4a = Inception4(576, 224, 64, 196, 128, 96, 128, 36, 576)
        self.inception4b = Inception4(576, 192, 96, 96, 128, 128, 128, 36, 576)
        self.inception4c = Inception4(576, 160, 128, 128, 160, 160, 96, 36, 576)
        self.inception4d = Inception4(576, 96, 128, 160, 192, 192, 96, 36, 576)
        self.inception4e = Inception4e(576, 64, 1024)
        self.inception5a = Inception4(1024, 352, 192, 160, 224, 320, 128, 64, 1024)
        self.inception5b = Inception5b(1024, 352, 192, 192, 224, 320, 128)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.inception3c(out)
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        out = self.inception5a(out)
        feat = self.inception5b(out)
        return self.fc(feat), feat
    def get_class_name(self):
        return self.__class__.__name__


if __name__ == '__main__':
    a = torch.FloatTensor(3, 1, 224, 224)
    n = SEGoogleNet(3755)
    print(n(a).size())
    # (b, 576, 1/2, 1/2)
    # n1 = Inception3(192, 64, 64, 96, 32, 16, 256)
    # n2 = Inception3(256, 64, 96, 96, 64, 20, 320)
    # n3 = Inception3c(320, 36, 576)
    # n4 = Inception4(576, 224, 64, 196, 128, 96, 128, 36, 576)
    # n5 = Inception4(576, 192, 96, 96, 128, 128, 128, 36, 576)
    # n6 = Inception4(576, 160, 128, 128, 160, 160, 96, 36, 576)
    # n7 = Inception4(576, 96, 128, 160, 192, 192, 96, 36, 576)
    # n8 = Inception4e(576, 64, 1024)
    # n9 = Inception4(1024, 352, 192, 160, 224, 320, 128, 64, 1024)
    # n10 = Inception5b(1024, 352, 192, 192, 224, 320, 128)
    # b = n1(a)
    # print(b.size())
    # c = n2(b)
    # print(c.size())
    # d = n3(c)
    # print(d.size())
    # e = n4(d)
    # print(e.size())
    # f = n5(e)
    # print(f.size())
    # g = n6(f)
    # print(f.size())
    # h = n7(g)
    # print(h.size())
    # i = n8(h)
    # print(i.size())
    # j = n9(i)
    # print(j.size())
    # k = n10(j)
    # print(k.size())
