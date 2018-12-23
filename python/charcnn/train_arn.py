import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from models.RACNN import RACNN
from datasets.casia_charcnn import CASIA


def pretrain_apn(network, criterion, optimizer, scheduler, trainLoader, n_epochs=10):
    network = network.cuda()

    for epoch in range(n_epochs):
        cum_loss = 0.0
        network.train()

        for (i, (inputs, labels)) in enumerate(trainLoader):
            inputs = inputs.cuda()
            features = network.cnn(inputs)
            outputs = network.apn(features.view(inputs.size(0), -1))

            # search regions with the highest response value in conv5
            loc_label = torch.ones([inputs.size(0), 3]) * 21  # tl = 0.25, fixed
            loc_label = loc_label.cuda()
            features = F.upsample(features, (108, 108))
            target = torch.zeros([inputs.size(0), 3])
            target[:, :2] = outputs[:, : 2].mul(27).add(54)
            target[:, 2] = outputs[:, 2].mul(7).add(21)
            print(target)
            target = target.cuda()
            for i in range(inputs.size(0)):
                response_map = features[i]
                response_map = response_map.mean(0)
                rawmaxidx = response_map.view(-1).max(0)[1]
                idx = []
                for d in list(response_map.size())[::-1]:
                    idx.append(rawmaxidx % d)
                    rawmaxidx = rawmaxidx / d
                # idx[0] col, idx[1] row
                loc_label[i, 0] = idx[1].float()
                loc_label[i, 1] = idx[0].float()
            print('*'*40)
            print(loc_label)
            loss = F.smooth_l1_loss(target, loc_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.data[0]

        scheduler.step()


if __name__ == '__main__':
    char_classes = 7356
    network = RACNN(char_classes, '/mnt/data1/dengbowen/code/python/charcnn/output/CharCnn.pth.tar')

    criterion = nn.L1Loss()
    optimizer = optim.SGD(network.apn.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1 if e < 2 else 0.1)
    train_image_dir = "/mnt/data1/dengbowen/character/images/train"
    train_label_dir = "/mnt/data1/dengbowen/character/labels/train"
    label_files = ['./datasets/Char4037-list.txt', './datasets/Char3319-list.txt']
    char_class = 7356
    train_set = CASIA(train_image_dir, train_label_dir, label_files, char_class)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    pretrain_apn(network, criterion, optimizer, scheduler, train_loader, n_epochs=6)
    with open('apn.pt', 'wb') as f:
        torch.save(network.apn, f)
