import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from datasets.casia_charcnn import CASIA
from models.RACNN import RACNN
from tqdm import tqdm as tqdm

def alternative_train(network, c_entropy1, c_entropy2, c_ranking, oc1, oc2, oa, trainLoader, n_epochs=10):
    network = network.cuda()
    c_entropy1 = c_entropy1.cuda()
    c_entropy2 = c_entropy2.cuda()
    c_ranking = c_ranking.cuda()

    for epoch in range(n_epochs):
        correct1 = correct2 = 0.0
        cum_loss1 = cum_loss2 = 0.0
        counter = 0
        t = tqdm(trainLoader, desc='Classifier epoch %d' % epoch)
        network.train()
        for step in range(30):
            for i, (inputs, labels) in enumerate(t):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs1, outputs2 = network(inputs)
                loss1 = c_entropy1(outputs1, labels)
                oc1.zero_grad()
                loss1.backward()
                oc1.step()

                loss2 = c_entropy2(outputs2, labels)
                oc2.zero_grad()
                loss2.backward()
                oc2.step()

                counter += inputs.size(0)
                cum_loss1 += loss1.data[0]
                max_scores, max_labels = outputs1.data.topk(5, dim=1)
                for j in range(5):
                    correct1 += (max_labels[:, j] == labels.data).sum()

                cum_loss2 += loss2.data[0]
                max_scores, max_labels = outputs2.data.topk(5, dim=1)
                for j in range(5):
                    correct2 += (max_labels[:, j] == labels.data).sum()
                t.set_postfix(loss1=cum_loss1 / (1 + i), acc1=100 * correct1 / counter,
                              loss2=cum_loss2 / (1 + i), acc2=100 * correct2 / counter)

        cum_loss = 0
        t = tqdm(trainLoader, desc='Attention epoch %d, step %d' % (epoch, step))

        for i, (inputs, labels) in enumerate(t):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs1, outputs2 = network(inputs)

            p1 = torch.Tensor(outputs1.size(1))._zero()
            p2 = torch.Tensor(outputs2.size(1))._zero()
            for cnt, idx in enumerate(labels.data):
                p1[cnt] = outputs1.data[cnt, idx]
                p2[cnt] = outputs2.data[cnt, idx]

            p1 = p1.cuda()
            p2 = p2.cuda()
            y = torch.Tensor(outputs1.size(1)).fill_(-1).cuda()
            loss = c_ranking(p1, p2, y)
            oa.zero_grad()
            loss.backward()
            oa.step()

            cum_loss += loss.data[0]
            t.set_postfix(loss=cum_loss / (1 + i))

# converged_network = torch.load("./racnn/racnn81.pt")
# network.classifier1 = converged_network.classifier1
# network.classifier2 = converged_network.classifier2
train_image_dir = "/mnt/data1/dengbowen/character/images/train"
train_label_dir = "/mnt/data1/dengbowen/character/labels/train"
test_image_dir = "/mnt/data1/dengbowen/cmp/images"
test_label_dir = "/mnt/data1/dengbowen/cmp/labels"
label_files = ['./datasets/Char4037-list.txt', './datasets/Char3319-list.txt']
char_class = 7356
train_set = CASIA(train_image_dir, train_label_dir, label_files, char_class)
test_set = CASIA(test_image_dir, test_label_dir, label_files, char_class)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
network = RACNN(char_class, '/mnt/data1/dengbowen/code/python/charcnn/output/CharCnn.pth.tar')
network.apn = torch.load('apn.pt')
c_entropy1 = nn.CrossEntropyLoss()
c_entropy2 = nn.CrossEntropyLoss()
c_ranking = nn.MarginRankingLoss(margin=0.05)
oc1 = optim.SGD(network.classifier1.parameters(), lr=0.001)
oc2 = optim.SGD(network.classifier2.parameters(), lr=0.001)
oa = optim.SGD(network.apn.parameters(), lr=1e-6)
alternative_train(network, c_entropy1, c_entropy2, c_ranking, oc1, oc2, oa, train_loader, n_epochs=10)
