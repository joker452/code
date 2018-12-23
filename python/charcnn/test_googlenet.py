import torch
import pickle
import torch.nn as nn
from models.googlenet import GoogLeNet
from datasets.casia_googlenet import CASIA
from torch.utils.data.dataloader import DataLoader


def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")


def get_char(code):
    return bytes.fromhex(code[4:] + code[2: 4]).decode('gbk')


with open('./utils/char2index.pkl', 'rb') as f:
    dictionary = pickle.load(f)


def evaluate_cnn(cnn, dataset_loader, device):
    # set the CNN in eval mode
    cnn.eval()
    top1 = 0.0
    top5 = 0.0
    total = 0
    with torch.no_grad():
        for sample_idx, (image, label) in enumerate(dataset_loader):
            total += label.size()[0]
            image = image.to(device)
            label = label.to(device).view(1, -1)
            outputs = cnn(image)
            _, predicts = outputs.topk(k=5, dim=1)
            tmp = predicts.cpu().numpy()[0]
            codes = []
            for index in tmp:
                codes.append(get_nth_key(dictionary, index))
            print(list(map(get_char, codes)))
            predicts = predicts.t()
            print(get_char(get_nth_key(dictionary, label[0].item())))
            correct = predicts.eq(label.expand_as(predicts))

            top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
            top5 += correct[:5].view(-1).float().sum(0, keepdim=True).item()
        # set the CNN in train model
    return 100 * top1 / total, 100 * top5 / total


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('./output/model.pth.tar')
char_class = 7356
cnn = GoogLeNet(char_class)
cnn = nn.DataParallel(cnn)
cnn.load_state_dict(checkpoint['state_dict'])
cnn.to(device)
test_image_dir = "/home/dengbowen/images"
test_label_dir = "/home/dengbowen/labels"
label_files = ['/mnt/data1/dengbowen/charcnn/datasets/Char4037-list.txt',
               '/mnt/data1/dengbowen/charcnn/datasets/Char3319-list.txt']
test_set = CASIA(test_image_dir, test_label_dir, label_files, char_class)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
evaluate_cnn(cnn, test_loader, device)
