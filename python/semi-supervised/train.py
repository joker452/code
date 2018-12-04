import torch
import torch.nn as nn
from models import VariationalAutoencoder
from datasets.casia_googlenet import CASIA
from torch.utils.data.dataloader import DataLoader

train_image_dir = "/mnt/data1/dengbowen/character/images/train"
train_label_dir = "/mnt/data1/dengbowen/character/labels/train"
label_files = ['./Char4037-list.txt', './Char3319-list.txt']
char_class = 7356


def binary_cross_entropy(r, x):
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)


train_set = CASIA(train_image_dir, train_label_dir, label_files, char_class)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
class CNN(nn.Module):
    def __init__(self):
        self.l1 = nn.Conv2d(1, 1, kernel_size=3, stride=2)
        self.l2 = nn.Conv2d(1, 1, kernel_size=3, stride=2)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        return out

cnn = CNN()
model = VariationalAutoencoder([729, 32, [256, 128]])

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

for epoch in range(50):
    model.train()
    total_loss = 0
    for batch_id, (image, label) in enumerate(train_loader):

        u = cnn(image)
        reconstruction = model(u)
        likelihood = -binary_cross_entropy(reconstruction, u)
        elbo = likelihood - model.kl_divergence

        L = -torch.mean(elbo)

        L.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += L.data[0]

    m = len(train_loader)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}\tL: {total_loss/m:.2f}")
