# -*- coding: utf-8 -*-
import os
import cv2
import torch
import logging
import random
import pickle
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from models.alexnet import AlexNet
from models.charcnn import CharCnn
from models.resnetx import ResNetX
from models.googlenet import GoogLeNet
from torchvision.transforms import transforms
from utils.segmentation import segment
from utils.gen_printed_char import generate_character


def IOU(bbox, gt):
    x_11, y_11, x_12, y_12 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[0] + bbox[2]
    iou = []
    print(len(gt))
    for i in range(0, len(gt), 4):
        x_21, y_21, x_22, y_22 = gt[i + 1], gt[i], gt[i + 1] + gt[i + 3], gt[i] + gt[i + 2]
        inter_x1 = max(x_11, x_21)
        inter_y1 = max(y_11, y_21)
        inter_x2 = min(x_12, x_22)
        inter_y2 = min(y_12, y_22)
        inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
        union_area = bbox[2] ** 2 + (x_22 - x_21) * (y_22 - y_21) - inter_area
        iou.append(inter_area / union_area)
    return max(iou)


def get_image(image, char_size, image_size):
    '''
    :param image: ndarray of the image
    :return: resized PIL image
    '''
    # crop the character from the original image
    black_index = np.where(image < 255)
    try:
        min_x = min(black_index[0])
        max_x = max(black_index[0])
        min_y = min(black_index[1])
        max_y = max(black_index[1])
    except ValueError:
        img = cv2.resize(image, dsize=(char_size, char_size), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(image[min_x: max_x + 1, min_y: max_y + 1], dsize=(char_size, char_size),
                         interpolation=cv2.INTER_CUBIC)
    if image_size != char_size:
        h, w = img.shape
        img_bg = np.full((image_size, image_size), 255, dtype='uint8')
        row_start = (img_bg.shape[0] - h) // 2
        col_start = (img_bg.shape[1] - w) // 2
        img_bg[row_start: row_start + h, col_start: col_start + w] = img
    else:
        img_bg = img
    return Image.fromarray(img_bg)


def feature_hook(m, input, output):
    inter_feature.append(input)


def draw_bb(img, row_start, column_start, width, height, class_score):
    color = random.choice(colors)
    cv2.rectangle(img, (column_start, row_start), (column_start + width, row_start + height), color, 1)
    text_size, _ = cv2.getTextSize(class_score, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    cv2.rectangle(img, (column_start, row_start), (column_start + text_size[0], row_start + text_size[1]), color, -1)
    cv2.putText(img, class_score, (column_start, row_start + text_size[1]), cv2.FONT_HERSHEY_PLAIN, 1,
                [255, 255, 255], 1)
    return img


def parse_arg():
    parser = argparse.ArgumentParser(description="Handwritten character recognition in text")

    parser.add_argument("--method", "-m", default='QBS', type=str, required=True, help="Query method")
    parser.add_argument("--net", default='charcnn', required=True, help='net architecture used')
    parser.add_argument("--model_path", required=True, help="path to load the pre-trained model")
    parser.add_argument("--character", "-char", required=True, help="The query character or the one you think" +
                                                                    "most likely to be true")
    parser.add_argument("--image_path", "-im", help="The query example image")
    parser.add_argument("--font_path", "-font", default='./utils/fonts/fangzheng_heiti.TTF', help="Path to the font used to generate" +
                                                                                                  "the character image only in effect when qbs")
    parser.add_argument("--image_dir", required=True, help="Directory of the images to test on")
    parser.add_argument("--out_dir", default='./result', help="Directory to store the result images")
    parser.add_argument("--label_dir", required=True, help="Directory of the corresponding labels")
    return parser.parse_args()


if __name__ == '__main__':
    # sanity check
    args = parse_arg()
    args.method = args.method.lower()
    args.net = args.net.lower()
    if args.method not in ['qbe', 'qbs']:
        raise NotImplementedError("Only qbs and qbe are supported!")
    if args.net not in ['charcnn', 'alexnet', 'googlenet', 'resnetx']:
        raise NotImplementedError("Please choose among charcnn, alexnet, googlenet and resnetx")
    if not os.path.exists(args.model_path):
        raise OSError("Cannot load model because {} doesn't exist!".format(args.model_path))
    # set character size and image size for net input
    size = {'charcnn': {'char_size': 108, 'image_size': 108},
            'alexnet': {'char_size': 108, 'image_size': 120},
            'googlenet': {'char_size': 112, 'image_size': 120},
            'resnetx': {'char_size': 108, 'image_size': 120}}
    # load colors used for bounding box, char2index dictionary and character frequency dictionary
    with open('./utils/pallete', 'rb') as f:
        colors = pickle.load(f)
    with open('./utils/char2index.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    with open('./utils/freq.pkl', 'rb') as f:
        freq_dict = pickle.load(f)
    # get the logger for printing
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('CASIA handwritten character recognition system')
    # make output directory

    if not os.path.exists(args.out_dir):
        try:
            os.makedirs(args.out_dir)
        except OSError:
            logger.error("Cann't make directory {:s}".format(args.out_dir))
        else:
            logger.info("Make output directory")
    else:

        logger.info("Output directory already exits")
    inter_feature = []
    # generate a query word
    query_word = args.character
    # get gbk code
    char_code = query_word.encode('gbk')
    # convert to the form in the char2index dictionary, and get the corresponding index
    char_code = hex(char_code[1]) + hex(char_code[0])[2:]
    try:
        char_index = dictionary[char_code]
    except KeyError as e:
        raise NotImplementedError("Don't support OOV now!") from e
    # load checkpoint
    checkpoint = torch.load(args.model_path)
    # init the net
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    char_class = 7356
    if args.net == 'charcnn':
        cnn = CharCnn(char_class)
        cnn.classifer.register_forward_hook(feature_hook)
        cnn.apply_parallel()
    elif args.net == 'alexnet':
        cnn = AlexNet(char_class)
        cnn.classifier.register_forward_hook(feature_hook)
        cnn.apply_parallel()
    elif args.net == 'googlenet':
        cnn = GoogLeNet(char_class)
        cnn.classifier.register_forward_hook(feature_hook)
        cnn = nn.DataParallel(cnn)
    else:
        cnn = ResNetX(num_blocks=[3, 3, 3], cardinality=2, bottleneck_width=64, num_classes=char_class)
        cnn.classifier.register_forward_hook(feature_hook)
        cnn = nn.DataParallel(cnn)
    cnn.load_state_dict(checkpoint['state_dict'])
    cnn.to(device)
    cnn.eval()
    # get the query image according to the query method
    if args.method == 'qbs':
        query_img = generate_character(query_word, size[args.net]['image_size'], size[args.net]['image_size'],  0,
                                       args.font_path, 0)
    else:
        query_img = get_image(cv2.imread(args.image_path, 0), size[args.net]['char_size'],
                              size[args.net]['image_size'])
    query_img = transforms.ToTensor()(query_img).unsqueeze_(0)
    query_img.to(device)
    # get the document image names and the corresponding label names
    image_list = [file for file in os.listdir(args.image_dir) if file.endswith('.png')]
    label_list = [file for file in os.listdir(args.label_dir) if file.endswith('.txt')]
    sorted(image_list)
    sorted(label_list)
    # true positive, true positive plus false positive, true positive plus false negative
    TP = 0
    TP_FP = 0
    try:
        TP_FN = freq_dict[char_code]
    except KeyError as e:
        TP_FN = 0
    for img_index, image_name in enumerate(image_list):
        # segment the image into lines
        segmented = segment(os.path.join(args.image_dir, image_name))
        doc, image_lines, row_starts = segmented['image'], segmented['image_lines'], segmented['row_start']
        logger.info("Total lines:{}".format(len(image_lines)))
        # get the gt for a particular character
        gt = {}
        with open(os.path.join(args.label_dir, label_list[img_index]), 'r', encoding='utf-8') as f:
            for line_index, line in enumerate(f):
                one_line = line.strip().split(' ')
                gt[str(line_index)] = []
                for i in range(0, len(one_line), 5):
                    if one_line[i] == char_code:
                        gt[str(line_index)] += list(map(int, one_line[i + 1: i + 5]))
        for index, image_line in enumerate(image_lines):
            if index < len(gt):
                h, w = image_line.shape
                logger.info("Now: {} row".format(index))
                start = False
                threshold = 0.0
                pre_column = 0
                for column_start in range(w - h + 1):
                    doc_line = get_image(image_line[:, column_start: column_start + h], size[args.net]['char_size'],
                                         size[args.net]['image_size'])
                    with torch.no_grad():
                        # get the feature map for the query image and normalize it
                        cnn(query_img)
                        feature_query = inter_feature[0][0]
                        feature_query = feature_query.view(1, -1)
                        # get the feature map for the document image and normalize it
                        doc_line = transforms.ToTensor()(doc_line).unsqueeze_(0)
                        doc_line.to(device)
                        probility = F.sigmoid(cnn(doc_line)[0, char_index])
                        feature_doc_line = inter_feature[1][0]
                        feature_doc_line = feature_doc_line.view(1, -1)
                        # compute the distance between the two feature maps
                        dis = F.pairwise_distance(feature_doc_line, feature_query)
                        inter_feature.clear()
                        if not start and probility > 0.5:
                            start = True
                            threshold = dis
                        if dis < threshold and column_start - pre_column > h and probility.item() > 0.97:
                            TP_FP += 1
                            pre_column = column_start
                            draw_bb(doc, row_starts[index], column_start, h, h, '{:.2%}'.format(probility.item()))
                            if gt[str(index)]:
                                bbox = (column_start, row_starts[index], h)
                                if IOU(bbox, gt[str(index)]) > 0.5:
                                    TP += 1
        Image.fromarray(doc).save(args.out_dir + os.sep + image_name)
    if TP_FN == 0:
        if TP_FP == 0:
            logger.info("The query character doesn't appear in the doc, so recall doesn't make sense," +
                        " but the system doesn't predict any positive results so the precision is 100%")
        else:
            logger.info("The query character doesn't appear in the doc, so recall doesn't make sense," +
                        " Precision:{:.2%}".format(TP / TP_FP))
    else:
        if TP_FP == 0:
            logger.info("Recall:0%, Precision:100%")
        else:
            logger.info("Recall:{:.2%}, Precision:{:.2%}".format(TP / TP_FN, TP / TP_FP))