import os
import torch
import torch.nn as nn
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model  # module that actually defined

    def forward(self, x):
        return self.module(x)


def mkdir(dir_name):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            print('Can not make directory for {}'.format(dir_name))
            raise OSError
        else:
            print("Make directory for {}".format(dir_name))
    else:
        print("{} already exists".format(dir_name))


def compare(c1, c2):
    p1 = c1['coordinates']
    p2 = c2['coordinates']
    # x1, y1, x2, y2
    # x1 bigger -> come first
    # y1 smaller -> come first
    if p1[0] > p2[0] + 100:
        return -1
    elif p1[0] < p2[0] - 100:
        return 1
    else:
        # in the same column
        if p1[1] < p2[1]:
            return -1
        else:
            return 1


def load_model(cnn, model_path):
    cnn = WrappedModel(cnn)
    d = torch.load(model_path, map_location={"cuda": "cpu"})
    cnn.load_state_dict(d['state_dict'])
    cnn = cnn.to(device)
    return cnn


def get_class(cnn, trans, img, x1, y1, x2, y2):
    char_img = img[y1: y2, x1: x2]
    net_in = trans(Image.fromarray(char_img)).to(device).unsqueeze(0)
    outputs = cnn(net_in)
    _, predicts = outputs.topk(k=1, dim=1)
    return predicts.item()
