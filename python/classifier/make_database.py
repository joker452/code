import os
import cv2
import sqlite3
import torch.optim
from models import resnet
from utils.util import mkdir, compare, load_model, get_class
from PIL import ImageDraw
from functools import cmp_to_key
from torchvision.transforms import transforms


def draw(img, boxes, i):
    d = ImageDraw.Draw(img)
    for words in boxes:
        for word in words:
            x1, y1, x2, y2 = word
            d.rectangle([x1, y1, x2, y2], outline='white', width=4)
    img.save("/mnt/data1/dengbowen/resnet/{}.png".format(i))


def write_char(f, x1, y1, x2, y2, cur_class, next_class, nnext_class):
    f.write(str(x1) + " " + str(y1) + " " +
            str(x2) + " " + str(y2) + " " +
            str(cur_class) + " " +
            str(next_class) + " " + str(nnext_class) + "\n")


def get_table_row(images, img_dir, res_dir, out_dir):
    with torch.no_grad():
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        for image in images:
            img_name = image.split('.')[0]
            with open(os.path.join(res_dir, img_name + '.txt'), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            positions = []
            for line in lines:
                positions.append({'coordinates': list(map(int, line.split()[0: 4]))})
            positions.sort(key=cmp_to_key(compare))
            img = cv2.imread(os.path.join(img_dir, image))
            out_file_name = os.path.join(out_dir, img_name + '.txt')
            if not os.path.exists(out_file_name):
                with open(out_file_name, 'w', encoding='utf-8') as f:
                    char_num = len(positions)
                    cur_class = next_class = nnext_class = -1
                    for k in range(char_num - 2):
                        if cur_class == -1:
                            x1, y1, x2, y2 = positions[k]['coordinates']
                            cur_class = get_class(cnn, trans, img, x1, y1, x2, y2)
                            x1, y1, x2, y2 = positions[k + 1]['coordinates']
                            next_class = get_class(cnn, trans, img, x1, y1, x2, y2)
                            x1, y1, x2, y2 = positions[k + 2]['coordinates']
                            nnext_class = get_class(cnn, trans, img, x1, y1, x2, y2)

                        else:
                            cur_class = next_class
                            next_class = nnext_class
                            x1, y1, x2, y2 = positions[k + 2]['coordinates']
                            nnext_class = get_class(cnn, trans, img, x1, y1, x2, y2)
                        write_char(f, x1, y1, x2, y2, cur_class, next_class, nnext_class)
                    x1, y1, x2, y2 = positions[char_num - 2]['coordinates']
                    write_char(f, x1, y1, x2, y2, next_class, nnext_class, -1)
                    x1, y1, x2, y2 = positions[char_num - 1]['coordinates']
                    write_char(f, x1, y1, x2, y2, nnext_class, -1, -1)


def create_table(db_path, table_name, src_dir):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE {}(
                      PAGE INTEGER,
                      LOCATION TEXT NOT NULL,
                      CLASS INTEGER NOT NULL,
                      NEXT INTEGER NOT NULL,
                      NNEXT INTEGER NOT NULL,
                      PRIMARY KEY(PAGE, LOCATION));'''.format(table_name))
    indexes = [f.name for f in os.scandir(out_dir) if f.name.endswith('.txt')]
    rows = []
    for index in indexes:
        page = index.split('.')[0]
        with open(os.path.join(src_dir, index), 'r', encoding='utf-8') as f:
            for line in f:
                # split location and class
                x1, y1, x2, y2, class_id, next_class, nnext_class, = line.split()
                location = '{} {} {} {}'.format(x1, y1, x2, y2)
                rows.append((int(page), location, int(class_id), int(next_class), int(nnext_class)))
    cursor.executemany('INSERT INTO {} VALUES (?, ?, ?, ?, ?);'.format(table_name), rows)
    cursor.execute("CREATE INDEX BI_CLASS_IDX ON {} (CLASS, NEXT);".format(table_name))
    cursor.execute("CREATE INDEX TRI_CLASS_IDX ON {} (CLASS, NEXT, NNEXT);".format(table_name))
    conn.commit()
    cursor.close()
    conn.close()


# parser = argparse.ArgumentParser()
# parser.add_argument('--words', '-w', required=True)
# args = parser.parse_args()

if __name__ == '__main__':
    # get the class of each detection result, create table for both detection and gt
    db_path = "./index.db"
    img_dir = "./bw_out"
    # detection result dir
    res_dir = "./res_txt"
    # output txt dir, add class and id to detectioon result
    out_dir = "./out"
    gt_dir = "./bw_out"
    model_path = './output/_ResNet03-30-14-33_89.96.pth.tar'
    if not os.path.exists(db_path):
        mkdir(out_dir)
        arch = '50'
        nl_type = 'cgnl'
        nl_nums = 1
        pool_size = 7
        char_class = 5601
        cnn = resnet.model_hub(arch, pretrained=False, nl_type=nl_type, nl_nums=nl_nums,
                               pool_size=pool_size)
        cnn._modules['fc'] = torch.nn.Linear(in_features=2048,
                                             out_features=char_class)
        cnn = load_model(cnn, model_path)
        images = [f.name for f in os.scandir(img_dir) if f.name.endswith(".jpg")]
        cnn.eval()
        get_table_row(images, img_dir, res_dir, out_dir)
        create_table(db_path, "DETECTION", out_dir)
        create_table(db_path, "GT", gt_dir)
