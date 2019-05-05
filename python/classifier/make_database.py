import os
import cv2
import sqlite3
import torch.optim
import torch.nn as nn
from models import resnet
from utils.makedir import mkdir
from PIL import Image, ImageDraw
from torchvision.transforms import transforms


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


def load_model(model_path, device, arch='50', nl_type='cgnl', nl_nums=1, pool_size=7, char_class=5601):
    cnn = resnet.model_hub(arch, pretrained=False, nl_type=nl_type, nl_nums=nl_nums,
                           pool_size=pool_size)
    cnn._modules['fc'] = torch.nn.Linear(in_features=2048,
                                         out_features=char_class)
    cnn = WrappedModel(cnn)
    d = torch.load(model_path, map_location={"cuda": "cpu"})
    cnn.load_state_dict(d['state_dict'])
    cnn = cnn.to(device)
    return cnn


def get_class(images, img_dir, res_dir, out_dir):
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


def create_db(db_path, out_dir):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE RESULT(
                      ID INTEGER,
                      LOCATION TEXT,
                      CLASS INTEGER NOT NULL,
                      PRIMARY KEY(ID, LOCATION));''')
    indexes = [f.name for f in os.scandir(out_dir)]
    rows = []
    for index in indexes:
        page = index.split('.')[0]
        with open(os.path.join(out_dir, index), 'r', encoding='utf-8') as f:
            for line in f:
                # split location and class
                split_point = line.rfind(' ') + 1
                rows.append((int(page), line[: split_point], int(line[split_point: -1])))
    cursor.executemany('INSERT INTO RESULT VALUES (?, ?, ?);', rows)
    conn.commit()
    cursor.close()
    conn.close()


# parser = argparse.ArgumentParser()
# parser.add_argument('--words', '-w', required=True)
# args = parser.parse_args()

if __name__ == '__main__':
    db_path = "./index.db"
    img_dir = "c:/Users/Deng/Desktop/bw_out"
    # detection result dir
    res_dir = "c:/Users/Deng/Desktop/res_txt"
    # output txt dir
    out_dir = "c:/Users/Deng/Desktop/out"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = './output/_ResNet03-30-14-33_89.96.pth.tar'
    if not os.path.exists(db_path):
        mkdir(out_dir)
        cnn = load_model(model_path, device)
        images = [f.name for f in os.scandir(img_dir) if f.name.endswith(".jpg")]
        cnn.eval()
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        get_class(images, img_dir, res_dir, out_dir)
        create_db(db_path, out_dir)
