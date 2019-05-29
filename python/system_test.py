import cv2
import ast
import torch
import pickle
import sqlite3
import argparse
import numpy as np
from models import resnet
from evaluate_box import bbox_overlaps
from utils.util import load_model, get_class
from torchvision.transforms import transforms


def AP(detection_result, gt_result):
    relevance = np.zeros(len(detection_result))
    n_relevant = len(gt_result)
    covered = []
    for i in range(len(detection_result)):
        gt_same_page = []
        dt = detection_result[i]
        for j in range(len(gt_result)):
            # same page
            if gt_result[j][0] > dt[0]:
                break
            gt_coordinates = list(map(int, gt_result[j][1].split()))
            if dt[0] == gt_result[j][0] and (gt_result[j][0], gt_coordinates) not in covered:
                gt_same_page.append(gt_coordinates)
        if len(gt_same_page) > 0:
            ious = bbox_overlaps(np.array(list(map(int, dt[1].split())), ndmin=2), np.array(gt_same_page)).view(1, len(
                gt_same_page))
            max_iou, idx = torch.max(ious, dim=1)
            if max_iou > 0.5:
                relevance[i] = 1.0
                covered.append((dt[0], gt_same_page[idx]))
    rel_cumsum = np.cumsum(relevance, dtype=float)
    precision = rel_cumsum / np.arange(1, relevance.size + 1)
    ap = (precision * relevance).sum() / n_relevant
    return ap


def search(table_name, cursor, keywords_idx):
    char_num = len(keywords_idx)
    if char_num == 1:
        sql_command = "SELECT PAGE, LOCATION FROM {} WHERE CLASS = ? ORDER BY PAGE"
        cursor.execute(sql_command.format(table_name), (keywords_idx[0],))
        detection_result = cursor.fetchall()
        cursor.execute(sql_command.format("GT"), (keywords_idx[0],))
        gt_result = cursor.fetchall()
    elif char_num == 2:
        sql_command = "SELECT PAGE, LOCATION FROM {} WHERE (CLASS, NEXT) =(?, ?) AND NEXT <> -1 ORDER BY PAGE"
        cursor.execute(sql_command.format(table_name), (keywords_idx[0], keywords_idx[1]))
        detection_result = cursor.fetchall()
        cursor.execute(sql_command.format("GT"), (keywords_idx[0], keywords_idx[1]))
        gt_result = cursor.fetchall()
    else:
        sql_command = "SELECT PAGE, LOCATION FROM {} WHERE (CLASS, NEXT, NNEXT) = (?, ?, ?) AND NEXT <> -1 ORDER BY PAGE"
        cursor.execute(sql_command.format(table_name), (keywords_idx[0], keywords_idx[1], keywords_idx[2]))
        detection_result = cursor.fetchall()
        cursor.execute(sql_command.format("GT"), (keywords_idx[0], keywords_idx[1], keywords_idx[2]))
        gt_result = cursor.fetchall()
    return detection_result, gt_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['qbe', 'qbs'], required=True, help='Query method')
    parser.add_argument('--words', '-w', help='Keyword used in query by string')
    parser.add_argument('--image_path', nargs='+', help='Example image path used in query by example')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--gpu', type=ast.literal_eval, default=True, help='Whether to use gpu or not')
    arg = parser.parse_args()
    # query by string
    conn = sqlite3.connect('/mnt/data1/dengbowen/python/classifier/index.db')
    cursor = conn.cursor()
    with open('./classifier/utils/index2char.pkl', 'rb') as f:
        index2char = pickle.load(f)
    if arg.method == 'qbs':
        with open('./classifier/utils/char2index.pkl', 'rb') as f:
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
        model_path = './classifier/output/_ResNet03-30-14-33_89.98.pth.tar'
        cnn = resnet.model_hub(arch, char_class, pretrained=False, nl_type=nl_type, nl_nums=nl_nums,
                               pool_size=pool_size)
        cnn._modules['fc'] = torch.nn.Linear(in_features=2048,
                                             out_features=char_class)
        cnn = load_model(cnn, model_path)
        cnn.eval()
        trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        keywords_idx = []
        for image_path in arg.image_path:
            img = cv2.imread(image_path)
            h, w, _ = img.shape
            class_id = get_class(cnn, trans, img, 0, 0, w, h)
            keywords_idx.append(class_id)
        if arg.verbose:
            # print query keywords
            with open('./classifier/utils/index2char.pkl', 'rb') as f:
                index2char = pickle.load(f)
            print(list(map(lambda x: index2char[str(x)], keywords_idx)))
    detection_result, gt_result = search("RCNN", cursor, keywords_idx)
    cursor.close()
    conn.close()
    print(AP(detection_result, gt_result))
