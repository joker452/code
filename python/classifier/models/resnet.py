# --------------------------------------------------------
# CGNL Network
# Copyright (c) 2018 Kaiyu Yue
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Functions for model building.
   Based on https://github.com/pytorch/vision
"""

import math
import torch
import torch.nn as nn

from termcolor import cprint
from collections import OrderedDict

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


def model_hub(arch, char_class, model_path='pretrained/resnet50-19c8e357.pth', pretrained=True, nl_type=None,
              nl_nums=None,
              pool_size=7):
    """Model hub.
    """

    if arch == '50':
        return resnet50(char_class,
                        model_path,
                        pretrained=pretrained,
                        nl_type=nl_type,
                        nl_nums=nl_nums,
                        pool_size=pool_size)
    elif arch == '101':
        return resnet101(char_class,
                         model_path,
                         pretrained=pretrained,
                         nl_type=nl_type,
                         nl_nums=nl_nums,
                         pool_size=pool_size)
    elif arch == '152':
        return resnet152(char_class,
                         model_path,
                         pretrained=pretrained,
                         nl_type=nl_type,
                         nl_nums=nl_nums,
                         pool_size=pool_size)
    else:
        raise NameError("The arch '{}' is not supported yet in this repo. \
                You can add it by yourself.".format(arch))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """

    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            cprint("=> WARN: SpatialCGNL block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            cprint("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
                   'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c * h * w) ** 0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 nl_type=None, nl_nums=None, pool_size=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if not nl_nums:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           nl_type=nl_type, nl_nums=nl_nums)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(pool_size, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if nl_nums == 1:
            for name, m in self._modules['layer3'][-2].named_modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, nl_type=None, nl_nums=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if (i == 5 and blocks == 6) or \
                    (i == 22 and blocks == 23) or \
                    (i == 35 and blocks == 36):
                if nl_type == 'cgnl':
                    layers.append(SpatialCGNL(
                        self.inplanes,
                        int(self.inplanes / 2),
                        use_scale=False,
                        groups=8))
                else:
                    pass

            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def load_partial_weight(model, pretrained, nl_nums, nl_layer_id):
    """Loads the partial weights for NL/CGNL network.
    """
    _pretrained = pretrained
    _model_dict = model.state_dict()
    _pretrained_dict = OrderedDict()
    for k, v in _pretrained.items():
        ks = k.split('.')
        if ks[0] == 'module':
            # remove wrapper
            ks = ks[1:]
            k = k[7:]
        layer_name = '.'.join(ks[0:2])
        if nl_nums == 1 and \
                layer_name == 'layer3.{}'.format(nl_layer_id):
            ks[1] = str(int(ks[1]) + 1)
            k = '.'.join(ks)
        _pretrained_dict[k] = v
    if _model_dict['fc.weight'].shape != _pretrained_dict['fc.weight'].shape:
        # different fc shape
        del _pretrained_dict['fc.weight']
        del _pretrained_dict['fc.bias']
        torch.nn.init.kaiming_normal_(_model_dict['fc.weight'],
                                      mode='fan_out', nonlinearity='relu')
    _model_dict.update(_pretrained_dict)
    return _model_dict


def resnet50(char_class, model_path='pretrained/resnet50-19c8e357.pth', pretrained=False, nl_type=None, nl_nums=None,
             **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   nl_type=nl_type, nl_nums=nl_nums, **kwargs)
    # change the fc layer
    model._modules['fc'] = torch.nn.Linear(in_features=2048,
                                           out_features=char_class)
    if pretrained:
        _pretrained = torch.load(model_path)['state_dict']

        _model_dict = load_partial_weight(model, _pretrained, nl_nums, 5)
        model.load_state_dict(_model_dict)

    return model


def resnet101(char_class, model_path='pretrained/resnet50-19c8e357.pth', pretrained=False, nl_type=None, nl_nums=None,
              **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   nl_type=nl_type, nl_nums=nl_nums, **kwargs)
    # change the fc layer
    model._modules['fc'] = torch.nn.Linear(in_features=2048,
                                           out_features=char_class)
    if pretrained:
        _pretrained = torch.load(model_path)['state_dict']
        _model_dict = load_partial_weight(model, _pretrained, nl_nums, 22)
        model.load_state_dict(_model_dict)

    return model


def resnet152(char_class, model_path='pretrained/resnet50-19c8e357.pth', pretrained=False, nl_type=None, nl_nums=None,
              **kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3],
                   nl_type=nl_type, nl_nums=nl_nums, **kwargs)
    # change the fc layer
    model._modules['fc'] = torch.nn.Linear(in_features=2048,
                                           out_features=char_class)
    if pretrained:
        _pretrained = torch.load(model_path)['state_dict']
        _model_dict = load_partial_weight(model, _pretrained, nl_nums, 35)
        model.load_state_dict(_model_dict)
    return model
