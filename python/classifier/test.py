# use CTE, t2.order = t1.order + 1
# order by order
# select next from prev
# support keyword length to 2
import cv2
import ast
import torch
import pickle
import sqlite3
import argparse
from models import resnet
from utils.util import load_model, get_class
from torchvision.transforms import transforms


def search(cursor, keywords_idx):
    char_num = len(keywords_idx)
    if char_num == 1:
        sql_command = "SELECT PAGE, LOCATION FROM {} WHERE CLASS = ?"
        cursor.execute(sql_command.format("DETECTION"), (keywords_idx[0], ))
        detection_result = cursor.fetchall()
        cursor.execute(sql_command.format("GT"), (keywords_idx[0], ))
        gt_result = cursor.fetchall()
    elif char_num == 2:
        sql_command = "SELECT PAGE, LOCATION FROM {} WHERE (CLASS, NEXT) =(?, ?) AND NEXT <> -1"
        cursor.execute(sql_command.format("DETECTION"), (keywords_idx[0], keywords_idx[1]))
        detection_result = cursor.fetchall()
        cursor.execute(sql_command.format("GT"), (keywords_idx[0], keywords_idx[1]))
        gt_result = cursor.fetchall()
    else:
        sql_command = "SELECT PAGE, LOCATION FROM {} WHERE (CLASS, NEXT, NNEXT) = (?, ?, ?) AND NEXT <> -1"
        cursor.execute(sql_command.format("DETECTION"), (keywords_idx[0], keywords_idx[1], keywords_idx[2]))
        detection_result = cursor.fetchall()
        cursor.execute(sql_command.format("GT"), (keywords_idx[0], keywords_idx[1], keywords_idx[2]))
        gt_result = cursor.fetchall()
    return detection_result, gt_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['qbe', 'qbs'], required=True, help='Query method')
    parser.add_argument('--words', '-w', help='Keyword used in query by string')
    parser.add_argument('--image_path', nargs='+', help='Example image path used in query by example')
    parser.add_argument('--gpu', type=ast.literal_eval, default=True, help='Whether to use gpu or not')
    arg = parser.parse_args()
    # query by string
    conn = sqlite3.connect('c:/users/Deng/Desktop/index.db')
    cursor = conn.cursor()
    if arg.method == 'qbs':
        with open('utils/char2index.pkl', 'rb') as f:
            char2index = pickle.load(f)
        keywords_idx = arg.words
        assert len(keywords_idx) < 4, "only keyword of length up to 3 is supported!"
        keywords_idx = list(map(lambda w: char2index[w], keywords_idx))
    else:
        assert type(arg.image_path) == list
        arch = '50'
        nl_type = 'cgnl'
        nl_nums = 1
        pool_size = 7
        char_class = 5601
        model_path = './output/_ResNet03-30-14-33_89.96.pth.tar'
        cnn = resnet.model_hub(arch, pretrained=False, nl_type=nl_type, nl_nums=nl_nums,
                               pool_size=pool_size)
        cnn._modules['fc'] = torch.nn.Linear(in_features=2048,
                                             out_features=char_class)
        cnn = load_model(cnn, model_path)
        cnn.eval()
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        keywords_idx = []
        for image_path in arg.image_path:
            img = cv2.imread(image_path)
            h, w = img.shape
            class_id = get_class(cnn, trans, img, 0, 0, w, h)
            keywords_idx.append(class_id)

    detection_result, gt_result = search(cursor, keywords_idx)
    cursor.close()
    conn.close()