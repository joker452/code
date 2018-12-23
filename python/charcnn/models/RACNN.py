import torch
import torch.nn as nn
import torch.functional as F
from .charcnn import CharCnn

class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1 / (1 + torch.exp(-10 * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size)
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()

        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            # 28~81
            tx = 54 + int(locs[i][0] * 27 + 0.5)
            ty = 54 + int(locs[i][1] * 27 + 0.5)
            # 15~28
            tl = 21 + int(locs[i][2] * 7 + 0.5)
            #tx = tx if tx > (in_size / 3) else in_size / 3
            # tx = tx if (in_size / 3 * 2) > tx else (in_size / 3 * 2)
           # ty = ty if ty > (in_size / 3) else in_size / 3
            # ty = ty if (in_size / 3 * 2) > ty else (in_size / 3 * 2)
            #tl = tl if tl > (in_size / 3) else in_size / 3

            w_off = int(tx - tl) if (tx - tl) > 0 else 0
            h_off = int(ty - tl) if (ty - tl) > 0 else 0
            w_end = int(tx + tl) if (tx + tl) < in_size else in_size
            h_end = int(ty + tl) if (ty + tl) < in_size else in_size

            mk = (h(x - w_off) - h(x - w_end)) * (h(y - h_off) - h(y - h_end))
            xatt = images[i] * mk

            xatt_cropped = xatt[:, w_off: w_end, h_off: h_end]
            before_upsample = xatt_cropped.unsqueeze(0)
            xamp = F.upsample(before_upsample, size=(108, 108), mode='bilinear', align_corners=True)
            ret.append(xamp.data.squeeze())

        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 108
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size / 4 * 3)
        short_size = (in_size / 4)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x < short_size) + (x >= long_size) + (y < short_size) + (y >= long_size)) > 0).float() * 2 - 1

        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))

        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()

        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret


class AttentionCropLayer(nn.Module):
    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)


class RACNN(nn.Module):
    def __init__(self, num_classes, cnn_path):
        super(RACNN, self).__init__()
        checkpoint = torch.load(cnn_path)
        self.cnn = CharCnn(num_classes)
        self.cnn.apply_parallel()
        self.cnn.load_state_dict(checkpoint['state_dict'])
        self.cnn = self.cnn.features


        self.classifier1 = nn.Sequential(
            nn.Linear(18432, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes))
        self.classifier2 = nn.Sequential(
            nn.Linear(18432, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

        self.apn = nn.Sequential(
            nn.Linear(18432, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 3),
            nn.Tanh())

        self.cropper = AttentionCropLayer()

    def forward(self, inputs):
        after_cnn1 = self.cnn(inputs).view(inputs.size(0), -1)
        apn_out = self.apn(after_cnn1)
        p1 = self.classifier1(after_cnn1)
        xamp_tensor = self.cropper(inputs, apn_out)
        after_cnn2 = self.cnn(xamp_tensor).view(inputs.size(0), -1)
        p2 = self.classifier2(after_cnn2)
        return p1, p2
