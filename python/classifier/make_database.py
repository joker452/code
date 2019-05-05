import numpy as np
import os
import torch.cuda
import argparse
import torch.nn as nn
import torch.optim
import torchvision
from PIL import Image, ImageDraw
from torchvision.transforms import transforms
from models import resnet
import cv2


class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model  # module that actually defined

    def forward(self, x):
        return self.module(x)


def draw(img, boxes, i):
    d = ImageDraw.Draw(img)
    for words in boxes:
        for word in words:
            x1, y1, x2, y2 = word
            d.rectangle([x1, y1, x2, y2], outline='white', width=4)
    img.save("/mnt/data1/dengbowen/resnet/{}.png".format(i))


def load_model(model_path, arch='50', nl_type='cgnl', nl_nums=1, pool_size=7, char_class=5601):
    cnn = resnet.model_hub(arch, pretrained=False, nl_type=nl_type, nl_nums=nl_nums,
                           pool_size=pool_size)

    # change the fc layer
    cnn._modules['fc'] = torch.nn.Linear(in_features=2048,
                                         out_features=char_class)
    cnn = WrappedModel(cnn)
    d = torch.load(model_path, map_location={"cuda": "cpu"})
    cnn.load_state_dict(d['state_dict'])
    cnn = cnn.cuda()
    return cnn


# parser = argparse.ArgumentParser()
# parser.add_argument('--words', '-w', required=True)
# args = parser.parse_args()

img_dir = "c:/Users/Deng/Desktop/bw_out"
res_dir = "c:/Users/Deng/Desktop/res_txt"
out_dir = "c:/Users/Deng/out"
images = [f.name for f in os.scandir(img_dir) if f.name.endswith(".jpg")]
cnn.eval()
trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
with torch.no_grad():
    for image in images:
        img_name = image.split('.')[0]
        with open(os.path.join(res_dir, img_name + '.txt'), 'r', encoding='utf-8') as f:
            positions = f.readlines()
        img = cv2.imread(os.path.join(img_dir, image))
        with open(os.path.join(out_dir, img_name + '.txt'), 'w', encoding='utf-8') as f:
            for pos in positions:
                x1, y1, x2, y2 = list(map(int, pos.split()))
                char_img = img[y1: y2, x1: x2]
                net_in = trans(Image.fromarray(char_img)).to(device).unsqueeze(0)
                outputs = cnn(net_in)
                _, predicts = outputs.topk(k=1, dim=1)
                f.write(pos[: -1] + " " + str(predicts.item()) + "\n")

# img = np.asarray(Image.open("355.jpg"))
# im1 = Image.open("0.jpg")
# im2 = Image.open("1.jpg")
# with torch.no_grad():
#     im1 = trans(im1)
#     im1 = im1.to(device)
#     image = im1.unsqueeze(0)
#     outputs = cnn(image)
#     _, predicts = outputs.topk(k=5, dim=1)
#     key_words.append(predicts.cpu().numpy()[0][0])
#     im2 = trans(im2)
#     im2 = im2.to(device)
#     image = im2.unsqueeze(0)
#     outputs = cnn(image)
#     _, predicts = outputs.topk(k=5, dim=1)
#     key_words.append(predicts.cpu().numpy()[0][0])
#     print(dd[str(key_words[-1])])
# print(key_words)
# with torch.no_grad():
#     for sample_idx, (image, label) in enumerate(test_loader):
#         image = image.to(device)
#         label = label.to(device).view(1, -1)
#         outputs = cnn(image)
#         _, predicts = outputs.topk(k=5, dim=1)
#         for j in range(0, 5):
#             print(dd[str(predicts[0][j].item())], end="")
#         print("")
# with open("355.txt", "r", encoding="utf-8") as f:
#     result = []
#     temp = []
#     begin = False
#     lines = f.readlines()
#     lines = lines[::-1]
#     with torch.no_grad():
#         for i, line in enumerate(lines):
#             x1, y1, x2, y2 = list(map(int, line.split()))
#             image = Image.fromarray(img[y1: y2, x1: x2, :])
#             image = trans(image)
#             image = image.to(device)
#             image = image.unsqueeze(0)
#             outputs = cnn(image)
#             _, predicts = outputs.topk(k=5, dim=1)
#             if not begin:
#                 if key_words[-1] in predicts.cpu().numpy().tolist()[0]:
#                     print(1)
#                     begin = True
#                     temp.append([(x1, y1, x2, y2)])
#             else:
#                 if key_words[0] in predicts.cpu().numpy().tolist()[0]:
#                     temp[-1].append((x1, y1, x2, y2))
#                     result.append(temp[-1])
#                 del temp[-1]
#                 begin = False
#     draw(Image.fromarray(img), result, 1111)
