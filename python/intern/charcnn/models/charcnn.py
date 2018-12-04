import torch.nn as nn


class CharCnn(nn.Module):

    def __init__(self, num_classes):
        super(CharCnn, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(64, 128, 3, 1, 1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(128, 256, 3, 1, 1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(256, 512, 3, 1, 1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 512, 3, 1, 1),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2))

        # change the first line below if the input size change
        self.classifer = nn.Sequential(nn.Linear(18432, 4096),
                                       nn.BatchNorm1d(4096),
                                       nn.ReLU(),
                                       nn.Linear(4096, num_classes))

    def forward(self, input):
        input = self.features(input)
        out = self.classifer(input.view(input.size()[0], -1))
        return out

    def get_class_name(self):
        return self.__class__.__name__

    def apply_parallel(self):
        self.features = nn.DataParallel(self.features)
